//! Production-hardening regression tests for the `torsh-nn` functional loss
//! family (campaign item W4-L).
//!
//! Every loss in this file used to be *severed* from the autograd graph: it read
//! its operands out with `to_vec()` and rebuilt the answer through
//! `Tensor::from_data` / `Tensor::from_vec`, which returns a fresh detached leaf,
//! or it routed the branch selection through `le()` + `where_tensor()`, neither
//! of which records. The forward numbers were right and `backward()` was
//! impossible — the worst possible failure mode for a loss function, because the
//! only thing a loss is *for* is being differentiated.
//!
//! Measured on the pre-W4-L tree (`requires_grad` of the returned loss):
//!
//! | loss                  | before | note                                   |
//! |-----------------------|--------|----------------------------------------|
//! | `l1_loss`             | true   | reconnected for free by recording `abs` |
//! | `binary_cross_entropy`| true   | reconnected for free by `maximum`/`minimum` |
//! | `smooth_l1_loss`      | false  | `to_vec` + `from_vec`                  |
//! | `huber_loss`          | false  | `le` + `where_tensor`                  |
//! | `wing_loss`           | false  | `to_vec` + `from_data`                 |
//! | `dice_loss`           | false  | `to_vec` + `from_data`                 |
//! | `tversky_loss`        | false  | `to_vec` + `from_data`                 |
//! | `focal_loss`          | false  | recorded `log_softmax`, then `to_vec`  |
//! | `center_loss`         | false  | `to_vec` + `from_data`                 |
//! | `infonce_loss`        | false  | `to_vec` + `from_data`                 |
//!
//! Each loss gets four checks: (a) the result joins the graph, (b) the analytic
//! gradient matches central finite differences, (c) the forward values still
//! equal what the detached kernel computed (pinned from a probe run against the
//! old implementations, so a rewrite cannot quietly change the formula), and
//! (d) every reduction mode keeps its previous output shape.

use torsh_core::error::Result;
use torsh_nn::functional;
use torsh_tensor::Tensor;

/// Serializes every test in this file against the process-global grad-mode
/// `AtomicBool` (`torsh-core/src/grad_mode.rs`). Under plain `cargo test` a
/// whole binary's tests share one process across a thread pool, so a `no_grad`
/// window opened by any test transiently suppresses recording for every other
/// test's tensor ops. Every test here asserts recording behaviour, so every test
/// here takes the lock for its full body. Pattern copied from
/// `torsh-tensor/tests/hardening_autograd_unary.rs`.
static GRAD_MODE_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes [`GRAD_MODE_GUARD`], recovering the guard if a previous test panicked
/// while holding it. Without the recovery a single genuine assertion failure
/// poisons the lock and every remaining test in the file reports
/// `PoisonError` instead of its own verdict.
fn grad_mode_guard() -> std::sync::MutexGuard<'static, ()> {
    GRAD_MODE_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Finite-difference step mandated for the campaign's gradient checks.
const FD_STEP: f32 = 1e-2;

/// Relative tolerance for the *forward-value* pins. Loose enough to absorb the
/// re-association that comes with expressing a hand-rolled scalar loop as a
/// composition of tensor ops (`sum()/count` instead of a sequential `f32` sum,
/// `matmul` instead of a manual dot product), tight enough that any change to
/// the formula itself shows up immediately.
const FORWARD_TOLERANCE: f32 = 1e-5;

/// Campaign tolerance for a single finite-difference comparison: relative for
/// large gradients, absolute for small ones.
fn fd_tolerance(numeric: f32) -> f32 {
    2e-2 * numeric.abs().max(1.0)
}

/// Central finite differences of a scalar objective w.r.t. every element of
/// `base`.
fn numeric_gradient<F>(base: &[f32], objective: F) -> Vec<f32>
where
    F: Fn(&[f32]) -> f32,
{
    (0..base.len())
        .map(|i| {
            let mut plus = base.to_vec();
            let mut minus = base.to_vec();
            plus[i] += FD_STEP;
            minus[i] -= FD_STEP;
            (objective(&plus) - objective(&minus)) / (2.0 * FD_STEP)
        })
        .collect()
}

fn assert_matches_finite_differences(label: &str, analytic: &[f32], numeric: &[f32]) {
    assert_eq!(
        analytic.len(),
        numeric.len(),
        "{label}: analytic gradient has {} entries, {} inputs were perturbed",
        analytic.len(),
        numeric.len()
    );
    for (i, (&got, &want)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let tol = fd_tolerance(want);
        assert!(
            (got - want).abs() <= tol,
            "{label}: gradient[{i}] = {got}, finite differences gave {want} \
             (tolerance {tol})\nanalytic = {analytic:?}\nnumeric  = {numeric:?}"
        );
    }
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: produced {} values, pinned {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = FORWARD_TOLERANCE * w.abs().max(1.0);
        assert!(
            (g - w).abs() <= tol,
            "{label}: value[{i}] = {g}, pinned {w} (tolerance {tol})\n\
             got    = {got:?}\npinned = {want:?}"
        );
    }
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec().expect("loss values")
}

fn leaf(data: &[f32], dims: &[usize]) -> Tensor {
    Tensor::from_vec(data.to_vec(), dims).expect("tensor")
}

fn param(data: &[f32], dims: &[usize]) -> Tensor {
    leaf(data, dims).requires_grad_(true)
}

fn gradient_of(tensor: &Tensor, label: &str) -> Vec<f32> {
    tensor
        .grad()
        .unwrap_or_else(|| panic!("{label}: backward did not populate a gradient"))
        .to_vec()
        .expect("gradient values")
}

/// The regression operands every element-wise loss below is measured on. The
/// differences `input - target` are `[0.5, -0.7, 1.2, 0.65, -1.5, -0.6]`, chosen
/// so that no element sits within `FD_STEP` of a kink at `|d| = 0`, `|d| = 0.5`
/// or `|d| = 1.0` — the finite-difference check would otherwise straddle the
/// kink and compare against a slope neither side actually has.
const INPUT: [f32; 6] = [0.6, -0.3, 1.4, 0.05, -1.2, 0.9];
const TARGET: [f32; 6] = [0.1, 0.4, 0.2, -0.6, 0.3, 1.5];
const DIMS: [usize; 2] = [2, 3];

/// Probabilities and binary labels for the losses that consume a `sigmoid`
/// output rather than raw values.
const PROBS: [f32; 6] = [0.2, 0.7, 0.45, 0.9, 0.05, 0.6];
const LABELS: [f32; 6] = [0.0, 1.0, 1.0, 1.0, 0.0, 0.0];

/// Logits for the classification losses.
const LOGITS: [f32; 6] = [0.5, -1.0, 2.0, 0.25, 1.5, -0.5];

// ---------------------------------------------------------------------------
// l1_loss — reconnected for free by `Tensor::abs` recording (W4-T)
// ---------------------------------------------------------------------------

#[test]
fn ls_l1_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let input = param(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let loss = functional::l1_loss(&input, &target, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "l1_loss must stay attached to its input"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&input, "l1_loss");

    let numeric = numeric_gradient(&INPUT, |x| {
        let probe = leaf(x, &DIMS);
        let target = leaf(&TARGET, &DIMS);
        values(&functional::l1_loss(&probe, &target, Reduction::Mean).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("l1_loss", &analytic, &numeric);
}

#[test]
fn ls_l1_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let input = leaf(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let none = functional::l1_loss(&input, &target, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &DIMS);
    assert_close(
        "l1_loss/none",
        &values(&none),
        &[0.5, 0.70000005, 1.1999999, 0.65000004, 1.5, 0.6],
    );

    let mean = functional::l1_loss(&input, &target, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("l1_loss/mean", &values(&mean), &[0.85833335]);

    let sum = functional::l1_loss(&input, &target, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("l1_loss/sum", &values(&sum), &[5.15]);
}

// ---------------------------------------------------------------------------
// binary_cross_entropy — reconnected for free by `maximum`/`minimum` recording.
//
// This is also the load-bearing check for every rewrite below: `clamped_input`
// feeds *two* consumers (`log()` and `ones.sub()`), so a correct gradient here
// proves the engine accumulates into a fan-out node instead of overwriting it.
// `smooth_l1_loss`, `huber_loss` and `wing_loss` all reuse `abs_diff` twice and
// depend on exactly that property.
// ---------------------------------------------------------------------------

#[test]
fn ls_binary_cross_entropy_is_differentiable() {
    let _guard = grad_mode_guard();
    let probs = param(&PROBS, &DIMS);
    let labels = leaf(&LABELS, &DIMS);

    let loss = functional::binary_cross_entropy(&probs, &labels, None, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "binary_cross_entropy must stay attached to its input"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&probs, "binary_cross_entropy");

    let numeric = numeric_gradient(&PROBS, |x| {
        let probe = leaf(x, &DIMS);
        let labels = leaf(&LABELS, &DIMS);
        values(&functional::binary_cross_entropy(&probe, &labels, None, Reduction::Mean).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("binary_cross_entropy", &analytic, &numeric);
}

#[test]
fn ls_binary_cross_entropy_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let probs = leaf(&PROBS, &DIMS);
    let labels = leaf(&LABELS, &DIMS);

    let none = functional::binary_cross_entropy(&probs, &labels, None, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &DIMS);
    assert_close(
        "binary_cross_entropy/none",
        &values(&none),
        &[
            0.22314353,
            0.35667497,
            0.79850775,
            0.105360545,
            0.051293306,
            0.9162908,
        ],
    );

    let mean = functional::binary_cross_entropy(&probs, &labels, None, Reduction::Mean).expect("mean");
    assert_close("binary_cross_entropy/mean", &values(&mean), &[0.40854514]);

    let sum = functional::binary_cross_entropy(&probs, &labels, None, Reduction::Sum).expect("sum");
    assert_close("binary_cross_entropy/sum", &values(&sum), &[2.4512708]);
}

// ---------------------------------------------------------------------------
// smooth_l1_loss
// ---------------------------------------------------------------------------

#[test]
fn ls_smooth_l1_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    for beta in [1.0f32, 0.5] {
        let input = param(&INPUT, &DIMS);
        let target = leaf(&TARGET, &DIMS);

        let loss = functional::smooth_l1_loss(&input, &target, beta, Reduction::Mean).expect("forward");
        assert!(
            loss.requires_grad(),
            "smooth_l1_loss (beta={beta}) must stay attached to its input; the \
             to_vec/from_vec kernel returned a detached leaf"
        );
        loss.backward().expect("backward");
        let analytic = gradient_of(&input, "smooth_l1_loss");

        let numeric = numeric_gradient(&INPUT, |x| {
            let probe = leaf(x, &DIMS);
            let target = leaf(&TARGET, &DIMS);
            values(&functional::smooth_l1_loss(&probe, &target, beta, Reduction::Mean).expect("probe"))
                .iter()
                .sum()
        });
        assert_matches_finite_differences(
            &format!("smooth_l1_loss(beta={beta})"),
            &analytic,
            &numeric,
        );
    }
}

#[test]
fn ls_smooth_l1_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let input = leaf(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let none = functional::smooth_l1_loss(&input, &target, 1.0, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &DIMS);
    assert_close(
        "smooth_l1_loss/none/beta=1",
        &values(&none),
        &[0.125, 0.24500003, 0.6999999, 0.21125002, 1.0, 0.18],
    );

    let half = functional::smooth_l1_loss(&input, &target, 0.5, Reduction::None).expect("beta=0.5");
    assert_close(
        "smooth_l1_loss/none/beta=0.5",
        &values(&half),
        &[0.25, 0.45000005, 0.9499999, 0.40000004, 1.25, 0.35000002],
    );

    let mean = functional::smooth_l1_loss(&input, &target, 1.0, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("smooth_l1_loss/mean", &values(&mean), &[0.41020834]);

    let sum = functional::smooth_l1_loss(&input, &target, 1.0, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("smooth_l1_loss/sum", &values(&sum), &[2.46125]);
}

#[test]
fn ls_smooth_l1_loss_zero_beta_degenerates_to_l1() {
    let _guard = grad_mode_guard();
    let input = param(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let loss = functional::smooth_l1_loss(&input, &target, 0.0, Reduction::None).expect("beta=0");
    assert_close(
        "smooth_l1_loss/beta=0",
        &values(&loss),
        &[0.5, 0.70000005, 1.1999999, 0.65000004, 1.5, 0.6],
    );
    assert!(
        loss.requires_grad(),
        "the beta == 0 branch must stay on the graph too"
    );
    loss.sum().expect("sum").backward().expect("backward");
    assert_close(
        "smooth_l1_loss/beta=0/grad",
        &gradient_of(&input, "smooth_l1_loss"),
        &[1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
    );
}

// ---------------------------------------------------------------------------
// huber_loss
// ---------------------------------------------------------------------------

#[test]
fn ls_huber_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    for delta in [1.0f32, 0.5] {
        let input = param(&INPUT, &DIMS);
        let target = leaf(&TARGET, &DIMS);

        let loss = functional::huber_loss(&input, &target, delta, Reduction::Mean).expect("forward");
        assert!(
            loss.requires_grad(),
            "huber_loss (delta={delta}) must stay attached to its input; \
             le()/where_tensor() do not record"
        );
        loss.backward().expect("backward");
        let analytic = gradient_of(&input, "huber_loss");

        let numeric = numeric_gradient(&INPUT, |x| {
            let probe = leaf(x, &DIMS);
            let target = leaf(&TARGET, &DIMS);
            values(&functional::huber_loss(&probe, &target, delta, Reduction::Mean).expect("probe"))
                .iter()
                .sum()
        });
        assert_matches_finite_differences(
            &format!("huber_loss(delta={delta})"),
            &analytic,
            &numeric,
        );
    }
}

#[test]
fn ls_huber_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let input = leaf(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let none = functional::huber_loss(&input, &target, 1.0, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &DIMS);
    assert_close(
        "huber_loss/none/delta=1",
        &values(&none),
        &[0.125, 0.24500003, 0.6999999, 0.21125002, 1.0, 0.18],
    );

    let half = functional::huber_loss(&input, &target, 0.5, Reduction::None).expect("delta=0.5");
    assert_close(
        "huber_loss/none/delta=0.5",
        &values(&half),
        &[0.125, 0.22500002, 0.47499996, 0.20000002, 0.625, 0.17500001],
    );

    let mean = functional::huber_loss(&input, &target, 1.0, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("huber_loss/mean", &values(&mean), &[0.41020834]);

    let sum = functional::huber_loss(&input, &target, 0.5, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("huber_loss/sum", &values(&sum), &[1.825]);
}

#[test]
fn ls_huber_loss_zero_delta_is_identically_zero() {
    let _guard = grad_mode_guard();
    let input = leaf(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let loss = functional::huber_loss(&input, &target, 0.0, Reduction::None).expect("delta=0");
    assert_close("huber_loss/delta=0", &values(&loss), &[0.0; 6]);
}

// ---------------------------------------------------------------------------
// wing_loss
// ---------------------------------------------------------------------------

#[test]
fn ls_wing_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let input = param(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let loss = functional::wing_loss(&input, &target, 1.0, 0.5, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "wing_loss must stay attached to its input"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&input, "wing_loss");

    let numeric = numeric_gradient(&INPUT, |x| {
        let probe = leaf(x, &DIMS);
        let target = leaf(&TARGET, &DIMS);
        values(&functional::wing_loss(&probe, &target, 1.0, 0.5, Reduction::Mean).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("wing_loss", &analytic, &numeric);
}

#[test]
fn ls_wing_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let input = leaf(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let none = functional::wing_loss(&input, &target, 1.0, 0.5, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &DIMS);
    assert_close(
        "wing_loss/none",
        &values(&none),
        &[
            0.6931472, 0.8754688, 1.2986122, 0.8329092, 1.5986123, 0.7884574,
        ],
    );

    let mean = functional::wing_loss(&input, &target, 1.0, 0.5, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("wing_loss/mean", &values(&mean), &[1.0145345]);

    let sum = functional::wing_loss(&input, &target, 1.0, 0.5, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("wing_loss/sum", &values(&sum), &[6.087207]);
}

#[test]
fn ls_wing_loss_zero_width_degenerates_to_l1() {
    let _guard = grad_mode_guard();
    let input = leaf(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    let loss = functional::wing_loss(&input, &target, 0.0, 0.5, Reduction::None).expect("width=0");
    assert_close(
        "wing_loss/width=0",
        &values(&loss),
        &[0.5, 0.70000005, 1.1999999, 0.65000004, 1.5, 0.6],
    );
}

// ---------------------------------------------------------------------------
// dice_loss
// ---------------------------------------------------------------------------

#[test]
fn ls_dice_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let probs = param(&PROBS, &DIMS);
    let labels = leaf(&LABELS, &DIMS);

    let loss = functional::dice_loss(&probs, &labels, 1.0, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "dice_loss must stay attached to its input"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&probs, "dice_loss");

    let numeric = numeric_gradient(&PROBS, |x| {
        let probe = leaf(x, &DIMS);
        let labels = leaf(&LABELS, &DIMS);
        values(&functional::dice_loss(&probe, &labels, 1.0, Reduction::Mean).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("dice_loss", &analytic, &numeric);
}

#[test]
fn ls_dice_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let probs = leaf(&PROBS, &DIMS);
    let labels = leaf(&LABELS, &DIMS);

    let none = functional::dice_loss(&probs, &labels, 1.0, Reduction::None).expect("none");
    assert_eq!(
        none.shape().dims(),
        &[1],
        "dice_loss reduces the whole batch to one number; 'none' kept shape [1]"
    );
    assert_close("dice_loss/none", &values(&none), &[0.26086956]);

    let mean = functional::dice_loss(&probs, &labels, 1.0, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("dice_loss/mean", &values(&mean), &[0.26086956]);

    let sum = functional::dice_loss(&probs, &labels, 1.0, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("dice_loss/sum", &values(&sum), &[0.26086956]);
}

// ---------------------------------------------------------------------------
// tversky_loss
// ---------------------------------------------------------------------------

#[test]
fn ls_tversky_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let probs = param(&PROBS, &DIMS);
    let labels = leaf(&LABELS, &DIMS);

    let loss = functional::tversky_loss(&probs, &labels, 0.3, 0.7, 1.0, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "tversky_loss must stay attached to its input"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&probs, "tversky_loss");

    let numeric = numeric_gradient(&PROBS, |x| {
        let probe = leaf(x, &DIMS);
        let labels = leaf(&LABELS, &DIMS);
        values(&functional::tversky_loss(&probe, &labels, 0.3, 0.7, 1.0, Reduction::Mean).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("tversky_loss", &analytic, &numeric);
}

#[test]
fn ls_tversky_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let probs = leaf(&PROBS, &DIMS);
    let labels = leaf(&LABELS, &DIMS);

    let none = functional::tversky_loss(&probs, &labels, 0.3, 0.7, 1.0, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &[1]);
    assert_close("tversky_loss/none", &values(&none), &[0.23173803]);

    let mean = functional::tversky_loss(&probs, &labels, 0.3, 0.7, 1.0, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("tversky_loss/mean", &values(&mean), &[0.23173803]);

    let sum = functional::tversky_loss(&probs, &labels, 0.3, 0.7, 1.0, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("tversky_loss/sum", &values(&sum), &[0.23173803]);
}

#[test]
fn ls_tversky_loss_rejects_out_of_range_weights() {
    let _guard = grad_mode_guard();
    let probs = leaf(&PROBS, &DIMS);
    let labels = leaf(&LABELS, &DIMS);
    assert!(
        functional::tversky_loss(&probs, &labels, 0.8, 0.7, 1.0, Reduction::Mean).is_err(),
        "alpha + beta > 1 must stay an error"
    );
}

// ---------------------------------------------------------------------------
// focal_loss
// ---------------------------------------------------------------------------

fn focal_classes() -> Tensor<i64> {
    Tensor::<i64>::from_vec(vec![2i64, 0], &[2]).expect("classes")
}

#[test]
fn ls_focal_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let logits = param(&LOGITS, &DIMS);
    let classes = focal_classes();

    let loss = functional::focal_loss(&logits, &classes, Some(0.25), 2.0, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "focal_loss calls the recording log_softmax and then threw the graph \
         away through to_vec(); it must stay attached"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&logits, "focal_loss");

    let numeric = numeric_gradient(&LOGITS, |x| {
        let probe = leaf(x, &DIMS);
        let classes = focal_classes();
        values(&functional::focal_loss(&probe, &classes, Some(0.25), 2.0, Reduction::Mean).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("focal_loss", &analytic, &numeric);
}

#[test]
fn ls_focal_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let logits = leaf(&LOGITS, &DIMS);
    let classes = focal_classes();

    let none = functional::focal_loss(&logits, &classes, Some(0.25), 2.0, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &[2]);
    assert_close(
        "focal_loss/none",
        &values(&none),
        &[0.0027731883, 0.25535023],
    );

    let mean = functional::focal_loss(&logits, &classes, Some(0.25), 2.0, Reduction::Mean).expect("mean");
    assert_eq!(
        mean.shape().dims(),
        &[1],
        "focal_loss builds its reduced output with shape [1]"
    );
    assert_close("focal_loss/mean", &values(&mean), &[0.12906171]);

    let sum = functional::focal_loss(&logits, &classes, Some(0.25), 2.0, Reduction::Sum).expect("sum");
    assert_eq!(sum.shape().dims(), &[1]);
    assert_close("focal_loss/sum", &values(&sum), &[0.25812343]);
}

#[test]
fn ls_focal_loss_rejects_out_of_range_class() {
    let _guard = grad_mode_guard();
    let logits = leaf(&LOGITS, &DIMS);
    let classes = Tensor::<i64>::from_vec(vec![7i64, 0], &[2]).expect("classes");
    assert!(
        functional::focal_loss(&logits, &classes, None, 2.0, Reduction::Mean).is_err(),
        "a class index outside [0, num_classes) must stay an error"
    );
}

/// Documented consequence of the graph-preserving rewrite: the target
/// log-probability is now selected with a one-hot multiply and a row sum, so a
/// non-finite entry *anywhere* in a row poisons that row (`0.0 * -inf == NaN`).
/// Same precedent as `nll_loss`'s `# Non-finite inputs` section.
#[test]
fn ls_focal_loss_non_finite_row_entry_yields_nan() {
    let _guard = grad_mode_guard();
    let logits = leaf(&[0.5, f32::NEG_INFINITY, 2.0, 0.25, 1.5, -0.5], &DIMS);
    let classes = focal_classes();

    let loss = functional::focal_loss(&logits, &classes, None, 2.0, Reduction::None).expect("none");
    let observed = values(&loss);
    assert!(
        observed[0].is_nan(),
        "a -inf logit elsewhere in row 0 must make row 0 NaN, got {observed:?}"
    );
    assert!(
        observed[1].is_finite(),
        "row 1 has no non-finite entry and must stay finite, got {observed:?}"
    );
}

// ---------------------------------------------------------------------------
// center_loss
// ---------------------------------------------------------------------------

const FEATURES: [f32; 4] = [0.3, -0.7, 1.1, 0.4];
const CENTERS: [f32; 6] = [0.0, 0.5, -1.0, 0.25, 2.0, -0.5];

fn center_labels() -> Tensor<i64> {
    Tensor::<i64>::from_vec(vec![1i64, 2], &[2]).expect("labels")
}

#[test]
fn ls_center_loss_is_differentiable_in_features() {
    let _guard = grad_mode_guard();
    let features = param(&FEATURES, &[2, 2]);
    let centers = leaf(&CENTERS, &[3, 2]);
    let labels = center_labels();

    let loss = functional::center_loss(&features, &labels, &centers, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "center_loss must stay attached to its features"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&features, "center_loss/features");

    let numeric = numeric_gradient(&FEATURES, |x| {
        let probe = leaf(x, &[2, 2]);
        let centers = leaf(&CENTERS, &[3, 2]);
        let labels = center_labels();
        values(&functional::center_loss(&probe, &labels, &centers, Reduction::Mean).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("center_loss/features", &analytic, &numeric);
}

#[test]
fn ls_center_loss_is_differentiable_in_centers() {
    let _guard = grad_mode_guard();
    let features = leaf(&FEATURES, &[2, 2]);
    let centers = param(&CENTERS, &[3, 2]);
    let labels = center_labels();

    let loss = functional::center_loss(&features, &labels, &centers, Reduction::Sum).expect("forward");
    assert!(
        loss.requires_grad(),
        "center_loss must reach the centers too"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&centers, "center_loss/centers");

    let numeric = numeric_gradient(&CENTERS, |x| {
        let features = leaf(&FEATURES, &[2, 2]);
        let probe = leaf(x, &[3, 2]);
        let labels = center_labels();
        values(&functional::center_loss(&features, &labels, &probe, Reduction::Sum).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("center_loss/centers", &analytic, &numeric);
    assert_eq!(
        &analytic[0..2],
        &[0.0, 0.0],
        "class 0 is unused by this batch and must receive no gradient"
    );
}

#[test]
fn ls_center_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let features = leaf(&FEATURES, &[2, 2]);
    let centers = leaf(&CENTERS, &[3, 2]);
    let labels = center_labels();

    let none = functional::center_loss(&features, &labels, &centers, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &[2]);
    assert_close("center_loss/none", &values(&none), &[1.2962499, 0.80999994]);

    let mean = functional::center_loss(&features, &labels, &centers, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("center_loss/mean", &values(&mean), &[1.0531249]);

    let sum = functional::center_loss(&features, &labels, &centers, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("center_loss/sum", &values(&sum), &[2.1062498]);
}

#[test]
fn ls_center_loss_rejects_out_of_range_label() {
    let _guard = grad_mode_guard();
    let features = leaf(&FEATURES, &[2, 2]);
    let centers = leaf(&CENTERS, &[3, 2]);
    let labels = Tensor::<i64>::from_vec(vec![1i64, 9], &[2]).expect("labels");
    assert!(
        functional::center_loss(&features, &labels, &centers, Reduction::Mean).is_err(),
        "a label outside [0, num_classes) must stay an error"
    );
}

/// The class centre is now selected by a one-hot `matmul` instead of an
/// element-wise read, so a non-finite value in *any* centre row poisons every
/// sample (`0.0 * inf == NaN`). Documented on `center_loss`, precedent set by
/// `nll_loss`.
#[test]
fn ls_center_loss_non_finite_unused_center_yields_nan() {
    let _guard = grad_mode_guard();
    let features = leaf(&FEATURES, &[2, 2]);
    let centers = leaf(&[f32::INFINITY, 0.5, -1.0, 0.25, 2.0, -0.5], &[3, 2]);
    let labels = center_labels();

    let loss = functional::center_loss(&features, &labels, &centers, Reduction::None).expect("none");
    assert!(
        values(&loss).iter().all(|v| v.is_nan()),
        "an inf in the unused centre row 0 must make the selection NaN, got {:?}",
        values(&loss)
    );
}

// ---------------------------------------------------------------------------
// infonce_loss
// ---------------------------------------------------------------------------

const ANCHOR: [f32; 4] = [0.3, -0.7, 1.1, 0.4];
const POSITIVE: [f32; 4] = [0.25, -0.6, 0.9, 0.5];
const NEGATIVES: [f32; 6] = [-0.4, 0.8, 1.2, -0.2, 0.05, 0.35];

#[test]
fn ls_infonce_loss_is_differentiable_in_anchor() {
    let _guard = grad_mode_guard();
    let anchor = param(&ANCHOR, &[2, 2]);
    let positive = leaf(&POSITIVE, &[2, 2]);
    let negatives = leaf(&NEGATIVES, &[3, 2]);

    let loss =
        functional::infonce_loss(&anchor, &positive, &negatives, 0.2, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "infonce_loss must stay attached to its anchor"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&anchor, "infonce_loss/anchor");

    let numeric = numeric_gradient(&ANCHOR, |x| {
        let probe = leaf(x, &[2, 2]);
        let positive = leaf(&POSITIVE, &[2, 2]);
        let negatives = leaf(&NEGATIVES, &[3, 2]);
        values(
            &functional::infonce_loss(&probe, &positive, &negatives, 0.2, Reduction::Mean).expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("infonce_loss/anchor", &analytic, &numeric);
}

#[test]
fn ls_infonce_loss_is_differentiable_in_positive_and_negatives() {
    let _guard = grad_mode_guard();
    let anchor = leaf(&ANCHOR, &[2, 2]);
    let positive = param(&POSITIVE, &[2, 2]);
    let negatives = param(&NEGATIVES, &[3, 2]);

    let loss =
        functional::infonce_loss(&anchor, &positive, &negatives, 0.2, Reduction::Sum).expect("forward");
    loss.backward().expect("backward");

    let positive_grad = gradient_of(&positive, "infonce_loss/positive");
    let numeric_positive = numeric_gradient(&POSITIVE, |x| {
        let anchor = leaf(&ANCHOR, &[2, 2]);
        let probe = leaf(x, &[2, 2]);
        let negatives = leaf(&NEGATIVES, &[3, 2]);
        values(&functional::infonce_loss(&anchor, &probe, &negatives, 0.2, Reduction::Sum).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences("infonce_loss/positive", &positive_grad, &numeric_positive);

    let negatives_grad = gradient_of(&negatives, "infonce_loss/negatives");
    let numeric_negatives = numeric_gradient(&NEGATIVES, |x| {
        let anchor = leaf(&ANCHOR, &[2, 2]);
        let positive = leaf(&POSITIVE, &[2, 2]);
        let probe = leaf(x, &[3, 2]);
        values(&functional::infonce_loss(&anchor, &positive, &probe, 0.2, Reduction::Sum).expect("probe"))
            .iter()
            .sum()
    });
    assert_matches_finite_differences(
        "infonce_loss/negatives",
        &negatives_grad,
        &numeric_negatives,
    );
}

#[test]
fn ls_infonce_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let anchor = leaf(&ANCHOR, &[2, 2]);
    let positive = leaf(&POSITIVE, &[2, 2]);
    let negatives = leaf(&NEGATIVES, &[3, 2]);

    let none = functional::infonce_loss(&anchor, &positive, &negatives, 0.2, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &[2]);
    assert_close(
        "infonce_loss/none",
        &values(&none),
        &[0.095543936, 0.49331966],
    );

    let mean = functional::infonce_loss(&anchor, &positive, &negatives, 0.2, Reduction::Mean).expect("mean");
    assert!(mean.shape().dims().is_empty());
    assert_close("infonce_loss/mean", &values(&mean), &[0.2944318]);

    let sum = functional::infonce_loss(&anchor, &positive, &negatives, 0.2, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("infonce_loss/sum", &values(&sum), &[0.5888636]);
}

/// The detached kernel special-cased a zero-norm embedding to a cosine
/// similarity of exactly `0.0`. The graph-preserving rewrite keeps that forward
/// value (a constant unit fallback replaces the zero divisor), so a degenerate
/// row still scores `ln(1 + num_negatives)`.
#[test]
fn ls_infonce_loss_zero_norm_row_keeps_its_forward_value() {
    let _guard = grad_mode_guard();
    let anchor = leaf(&[0.0, 0.0, 1.1, 0.4], &[2, 2]);
    let positive = leaf(&POSITIVE, &[2, 2]);
    let negatives = leaf(&NEGATIVES, &[3, 2]);

    let loss = functional::infonce_loss(&anchor, &positive, &negatives, 0.2, Reduction::None).expect("none");
    let observed = values(&loss);
    assert_close(
        "infonce_loss/zero-norm row",
        &observed[0..1],
        &[4.0f32.ln()],
    );
    assert!(
        observed[1].is_finite(),
        "the non-degenerate row must stay finite, got {observed:?}"
    );
}

// ---------------------------------------------------------------------------
// cross_entropy — the one-hot selector must live on the input's device
// ---------------------------------------------------------------------------

#[test]
fn ls_cross_entropy_output_stays_on_the_input_device() -> Result<()> {
    let _guard = grad_mode_guard();
    let logits = param(&LOGITS, &DIMS);
    let classes = focal_classes();

    let loss = functional::cross_entropy(&logits, &classes, None, Reduction::Mean, None)?;
    assert_eq!(
        loss.device(),
        logits.device(),
        "cross_entropy built its one-hot selector on a hardcoded DeviceType::Cpu; \
         it must follow the input's device"
    );

    let weight = leaf(&[0.5, 2.0, 1.0], &[3]);
    let weighted = functional::cross_entropy(&logits, &classes, Some(&weight), Reduction::Mean, None)?;
    assert_eq!(
        weighted.device(),
        logits.device(),
        "the class-weight tensor was hardcoded to DeviceType::Cpu as well"
    );
    Ok(())
}

#[test]
fn ls_cross_entropy_rejects_a_target_batch_mismatch() {
    let _guard = grad_mode_guard();
    let logits = leaf(&LOGITS, &DIMS);
    let classes = Tensor::<i64>::from_vec(vec![2i64, 0, 1], &[3]).expect("classes");
    assert!(
        functional::cross_entropy(&logits, &classes, None, Reduction::Mean, None).is_err(),
        "three targets for a two-row input must be an error, not an out-of-bounds write"
    );
}

// ---------------------------------------------------------------------------
// Reduction-mode coverage: every Reduction must compile
// ---------------------------------------------------------------------------

#[test]
fn ls_all_reduction_variants_are_accepted() {
    let _guard = grad_mode_guard();
    let input = leaf(&INPUT, &DIMS);
    let target = leaf(&TARGET, &DIMS);

    // All four variants must compile and succeed
    assert!(functional::mse_loss(&input, &target, Reduction::None).is_ok());
    assert!(functional::mse_loss(&input, &target, Reduction::Mean).is_ok());
    assert!(functional::mse_loss(&input, &target, Reduction::Sum).is_ok());
    assert!(functional::mse_loss(&input, &target, Reduction::BatchMean).is_ok());
}

// ---------------------------------------------------------------------------
// functional::loss_advanced — the `CustomLoss` framework
//
// `Reduction::apply` read the total out with `to_vec()` and rebuilt it with
// `Tensor::from_data` for *both* reduced arms, so every loss in that module was
// detached the moment it was reduced, however carefully its `forward` had been
// written. `SmoothL1Loss`, `HuberLoss`, `DiceLoss`, `IoULoss` and `WeightedLoss`
// were additionally severed inside their own `forward`. Measured before the fix:
// every one of the fourteen rows below reported `requires_grad == false`.
// ---------------------------------------------------------------------------

use torsh_nn::functional::{
    CustomLoss, DiceLoss, HuberLoss, IoULoss, LossFactory, Reduction, SmoothL1Loss, WeightedLoss,
};

fn advanced_operands() -> (Tensor, Tensor, Tensor) {
    (
        param(&INPUT, &DIMS),
        leaf(&TARGET, &DIMS),
        leaf(&LABELS, &DIMS),
    )
}

#[test]
fn ls_advanced_smooth_l1_is_differentiable_and_pinned() {
    let _guard = grad_mode_guard();
    let (predictions, targets, _) = advanced_operands();

    let none = SmoothL1Loss::new(1.0, Reduction::None)
        .compute_loss(&predictions, &targets)
        .expect("none");
    assert!(none.requires_grad(), "SmoothL1Loss must stay on the graph");
    assert_eq!(none.shape().dims(), &DIMS);
    assert_close(
        "SmoothL1Loss/None",
        &values(&none),
        &[0.125, 0.24500003, 0.6999999, 0.21125002, 1.0, 0.18],
    );

    let mean = SmoothL1Loss::new(1.0, Reduction::Mean)
        .compute_loss(&predictions, &targets)
        .expect("mean");
    assert!(
        mean.requires_grad(),
        "Reduction::Mean rebuilt the total through Tensor::from_data and detached it"
    );
    assert_eq!(mean.shape().dims(), &[1]);
    assert_close("SmoothL1Loss/Mean", &values(&mean), &[1.230625]);

    let sum = SmoothL1Loss::new(1.0, Reduction::Sum)
        .compute_loss(&predictions, &targets)
        .expect("sum");
    assert!(sum.requires_grad(), "Reduction::Sum detached the total too");
    assert_eq!(sum.shape().dims(), &[1]);
    assert_close("SmoothL1Loss/Sum", &values(&sum), &[2.46125]);

    sum.backward().expect("backward");
    let analytic = gradient_of(&predictions, "SmoothL1Loss");
    let numeric = numeric_gradient(&INPUT, |x| {
        let probe = leaf(x, &DIMS);
        let targets = leaf(&TARGET, &DIMS);
        values(
            &SmoothL1Loss::new(1.0, Reduction::Sum)
                .compute_loss(&probe, &targets)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("SmoothL1Loss", &analytic, &numeric);
}

#[test]
fn ls_advanced_huber_is_differentiable_and_pinned() {
    let _guard = grad_mode_guard();
    let (predictions, targets, _) = advanced_operands();

    let none = HuberLoss::new(0.5, Reduction::None)
        .compute_loss(&predictions, &targets)
        .expect("none");
    assert!(none.requires_grad(), "HuberLoss must stay on the graph");
    assert_close(
        "HuberLoss/None",
        &values(&none),
        &[0.125, 0.22500002, 0.47499996, 0.20000002, 0.625, 0.17500001],
    );

    let mean = HuberLoss::new(0.5, Reduction::Mean)
        .compute_loss(&predictions, &targets)
        .expect("mean");
    assert_eq!(mean.shape().dims(), &[1]);
    assert_close("HuberLoss/Mean", &values(&mean), &[0.9125]);

    let sum = HuberLoss::new(0.5, Reduction::Sum)
        .compute_loss(&predictions, &targets)
        .expect("sum");
    assert_close("HuberLoss/Sum", &values(&sum), &[1.825]);

    sum.backward().expect("backward");
    let analytic = gradient_of(&predictions, "HuberLoss");
    let numeric = numeric_gradient(&INPUT, |x| {
        let probe = leaf(x, &DIMS);
        let targets = leaf(&TARGET, &DIMS);
        values(
            &HuberLoss::new(0.5, Reduction::Sum)
                .compute_loss(&probe, &targets)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("HuberLoss", &analytic, &numeric);
}

#[test]
fn ls_advanced_dice_is_differentiable_and_pinned() {
    let _guard = grad_mode_guard();
    let (predictions, _, binary) = advanced_operands();

    let none = DiceLoss::new(1.0, Reduction::None)
        .compute_loss(&predictions, &binary)
        .expect("none");
    assert!(none.requires_grad(), "DiceLoss must stay on the graph");
    assert_eq!(none.shape().dims(), &[1]);
    assert_close("DiceLoss/None", &values(&none), &[0.38860774]);

    let mean = DiceLoss::new(1.0, Reduction::Mean)
        .compute_loss(&predictions, &binary)
        .expect("mean");
    assert_close("DiceLoss/Mean", &values(&mean), &[0.19430387]);

    let sum = DiceLoss::new(1.0, Reduction::Sum)
        .compute_loss(&predictions, &binary)
        .expect("sum");
    assert_close("DiceLoss/Sum", &values(&sum), &[0.38860774]);

    sum.backward().expect("backward");
    let analytic = gradient_of(&predictions, "DiceLoss");
    let numeric = numeric_gradient(&INPUT, |x| {
        let probe = leaf(x, &DIMS);
        let binary = leaf(&LABELS, &DIMS);
        values(
            &DiceLoss::new(1.0, Reduction::Sum)
                .compute_loss(&probe, &binary)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("DiceLoss", &analytic, &numeric);
}

#[test]
fn ls_advanced_iou_is_differentiable_and_pinned() {
    let _guard = grad_mode_guard();
    let (predictions, _, binary) = advanced_operands();

    let none = IoULoss::new(1.0, Reduction::None)
        .compute_loss(&predictions, &binary)
        .expect("none");
    assert!(none.requires_grad(), "IoULoss must stay on the graph");
    assert_eq!(none.shape().dims(), &[1]);
    assert_close("IoULoss/None", &values(&none), &[0.50962794]);

    let mean = IoULoss::new(1.0, Reduction::Mean)
        .compute_loss(&predictions, &binary)
        .expect("mean");
    assert_close("IoULoss/Mean", &values(&mean), &[0.25481397]);

    let sum = IoULoss::new(1.0, Reduction::Sum)
        .compute_loss(&predictions, &binary)
        .expect("sum");
    assert_close("IoULoss/Sum", &values(&sum), &[0.50962794]);

    sum.backward().expect("backward");
    let analytic = gradient_of(&predictions, "IoULoss");
    let numeric = numeric_gradient(&INPUT, |x| {
        let probe = leaf(x, &DIMS);
        let binary = leaf(&LABELS, &DIMS);
        values(
            &IoULoss::new(1.0, Reduction::Sum)
                .compute_loss(&probe, &binary)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("IoULoss", &analytic, &numeric);
}

#[test]
fn ls_advanced_weighted_loss_is_differentiable_and_pinned() {
    let _guard = grad_mode_guard();
    let (predictions, targets, _) = advanced_operands();

    let none = WeightedLoss::new(
        SmoothL1Loss::new(1.0, Reduction::None),
        vec![0.5, 2.0],
        Reduction::None,
    )
    .compute_loss(&predictions, &targets)
    .expect("none");
    assert!(none.requires_grad(), "WeightedLoss must stay on the graph");
    assert_close(
        "WeightedLoss/None",
        &values(&none),
        &[0.0625, 0.49000007, 0.34999996, 0.42250004, 0.5, 0.36],
    );

    let mean = WeightedLoss::new(
        SmoothL1Loss::new(1.0, Reduction::None),
        vec![0.5, 2.0],
        Reduction::Mean,
    )
    .compute_loss(&predictions, &targets)
    .expect("mean");
    assert_close("WeightedLoss/Mean", &values(&mean), &[1.0925]);

    mean.backward().expect("backward");
    let analytic = gradient_of(&predictions, "WeightedLoss");
    let numeric = numeric_gradient(&INPUT, |x| {
        let probe = leaf(x, &DIMS);
        let targets = leaf(&TARGET, &DIMS);
        values(
            &WeightedLoss::new(
                SmoothL1Loss::new(1.0, Reduction::None),
                vec![0.5, 2.0],
                Reduction::Mean,
            )
            .compute_loss(&probe, &targets)
            .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("WeightedLoss", &analytic, &numeric);
}

// ---------------------------------------------------------------------------
// functional::loss_advanced::LossFactory — the two factory losses that
// hand-rolled their own kernel instead of delegating. Both measured
// `requires_grad == false` in all three reduction modes before the fix.
// ---------------------------------------------------------------------------

#[test]
fn ls_factory_label_smoothing_ce_is_differentiable_and_pinned() {
    let _guard = grad_mode_guard();
    let logits = param(&LOGITS, &DIMS);
    let classes = leaf(&[2.0, 0.0], &[2]);

    let none = LossFactory::label_smoothing_ce(0.1, Reduction::None)
        .compute_loss(&logits, &classes)
        .expect("none");
    assert!(
        none.requires_grad(),
        "LabelSmoothingCE called the recording log_softmax and then rebuilt the \
         per-sample losses through Tensor::from_vec"
    );
    assert_eq!(none.shape().dims(), &[2]);
    assert_close(
        "LabelSmoothingCE/None",
        &values(&none),
        &[0.39131135, 1.5852852],
    );

    let mean = LossFactory::label_smoothing_ce(0.1, Reduction::Mean)
        .compute_loss(&logits, &classes)
        .expect("mean");
    assert_eq!(mean.shape().dims(), &[1]);
    assert_close("LabelSmoothingCE/Mean", &values(&mean), &[0.9882983]);

    let sum = LossFactory::label_smoothing_ce(0.1, Reduction::Sum)
        .compute_loss(&logits, &classes)
        .expect("sum");
    assert_close("LabelSmoothingCE/Sum", &values(&sum), &[1.9765966]);

    sum.backward().expect("backward");
    let analytic = gradient_of(&logits, "LabelSmoothingCE");
    let numeric = numeric_gradient(&LOGITS, |x| {
        let probe = leaf(x, &DIMS);
        let classes = leaf(&[2.0, 0.0], &[2]);
        values(
            &LossFactory::label_smoothing_ce(0.1, Reduction::Sum)
                .compute_loss(&probe, &classes)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("LabelSmoothingCE", &analytic, &numeric);
}

#[test]
fn ls_factory_center_loss_is_differentiable_and_pinned() {
    let _guard = grad_mode_guard();
    let features = param(&FEATURES, &[2, 2]);
    let labels = leaf(&[1.0, 2.0], &[2]);

    let none = LossFactory::center_loss(3, 2, 0.5, Reduction::None)
        .compute_loss(&features, &labels)
        .expect("none");
    assert!(
        none.requires_grad(),
        "the factory's CenterLoss must stay attached to its features"
    );
    assert_eq!(none.shape().dims(), &[2]);
    assert_close("FactoryCenterLoss/None", &values(&none), &[0.29, 0.685]);

    let mean = LossFactory::center_loss(3, 2, 0.5, Reduction::Mean)
        .compute_loss(&features, &labels)
        .expect("mean");
    assert_eq!(mean.shape().dims(), &[1]);
    assert_close("FactoryCenterLoss/Mean", &values(&mean), &[0.4875]);

    let sum = LossFactory::center_loss(3, 2, 0.5, Reduction::Sum)
        .compute_loss(&features, &labels)
        .expect("sum");
    assert_close("FactoryCenterLoss/Sum", &values(&sum), &[0.975]);

    sum.backward().expect("backward");
    let analytic = gradient_of(&features, "FactoryCenterLoss");
    let numeric = numeric_gradient(&FEATURES, |x| {
        let probe = leaf(x, &[2, 2]);
        let labels = leaf(&[1.0, 2.0], &[2]);
        values(
            &LossFactory::center_loss(3, 2, 0.5, Reduction::Sum)
                .compute_loss(&probe, &labels)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("FactoryCenterLoss", &analytic, &numeric);
}

#[test]
fn ls_factory_center_loss_rejects_an_out_of_range_label() {
    let _guard = grad_mode_guard();
    let features = leaf(&FEATURES, &[2, 2]);
    let labels = leaf(&[1.0, 9.0], &[2]);
    assert!(
        LossFactory::center_loss(3, 2, 0.5, Reduction::None)
            .compute_loss(&features, &labels)
            .is_err(),
        "a label outside [0, num_classes) must be an error, not a short loss vector"
    );
}

// ---------------------------------------------------------------------------
// W5-N: the four losses that were still severed after W4-L
// ---------------------------------------------------------------------------
//
// Measured on the pre-W5-N tree with `tests/zz_probe_w5n.rs` (deleted after the
// values below were captured):
//
// | loss                    | `requires_grad` | mechanism                        |
// |-------------------------|-----------------|----------------------------------|
// | `triplet_margin_loss`   | false           | `to_vec` + `Tensor::from_vec`    |
// | `contrastive_loss`      | false           | `to_vec` + `Tensor::from_vec`    |
// | `multi_margin_loss`     | false           | `to_vec` + `Tensor::from_vec`    |
// | `cosine_embedding_loss` | n/a — errored   | see below                        |
//
// `cosine_embedding_loss` did not merely detach: it computed a *single global*
// cosine over the flattened operands (`input1.mul(input2)?.sum()`,
// `input1.norm()`) and then compared a rank-1 target against that 0-D scalar,
// so every batched call died with
// `ShapeMismatch { expected: [2], got: [] }`. The only shape combination it
// ever served was a 1-D pair with a 0-D target — pinned below, because for that
// case the global cosine *is* the per-sample cosine.

/// Triplet fixture. Both samples sit strictly inside the hinge
/// (`d(a,p) - d(a,n) + margin` is `0.851` and `1.144`), and every pairwise
/// distance is bounded away from zero, so neither the hinge kink nor the
/// `p`-norm's singularity at the origin is within `FD_STEP` of a probe.
const TRI_ANCHOR: [f32; 4] = [0.5, -0.3, 1.2, 0.7];
const TRI_POSITIVE: [f32; 4] = [0.2, 0.4, 0.9, 0.1];
const TRI_NEGATIVE: [f32; 4] = [-0.8, 1.1, 2.0, -0.6];
const TRI_DIMS: [usize; 2] = [2, 2];
const TRI_MARGIN: f32 = 2.0;

#[test]
fn ls_triplet_margin_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let anchor = leaf(&TRI_ANCHOR, &TRI_DIMS);
    let positive = leaf(&TRI_POSITIVE, &TRI_DIMS);
    let negative = leaf(&TRI_NEGATIVE, &TRI_DIMS);

    let none =
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 2.0, Reduction::None)
            .expect("none");
    assert_eq!(none.shape().dims(), &[2]);
    assert_close("triplet/p2/none", &values(&none), &[0.85108006, 1.1443866]);

    let mean =
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 2.0, Reduction::Mean)
            .expect("mean");
    assert_eq!(
        mean.shape().dims(),
        &[1],
        "the reduced forms keep their historical `[1]` shape"
    );
    assert_close("triplet/p2/mean", &values(&mean), &[0.99773335]);

    let sum =
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 2.0, Reduction::Sum)
            .expect("sum");
    assert_eq!(sum.shape().dims(), &[1]);
    assert_close("triplet/p2/sum", &values(&sum), &[1.9954667]);

    // `p` is a free parameter, not a hardcoded 2.
    let p1 =
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 1.0, Reduction::None)
            .expect("p1");
    assert_close("triplet/p1/none", &values(&p1), &[0.29999995, 0.8000002]);
    let p3 =
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 3.0, Reduction::None)
            .expect("p3");
    assert_close("triplet/p3/none", &values(&p3), &[1.014682, 1.230003]);
}

#[test]
fn ls_triplet_margin_loss_keeps_its_rank_conventions() {
    let _guard = grad_mode_guard();

    // Rank 1: every element is its own sample with a one-element feature vector.
    let a1 = leaf(&TRI_ANCHOR, &[4]);
    let p1 = leaf(&TRI_POSITIVE, &[4]);
    let n1 = leaf(&TRI_NEGATIVE, &[4]);
    let rank1 = functional::triplet_margin_loss(&a1, &p1, &n1, TRI_MARGIN, 2.0, Reduction::None)
        .expect("rank-1 operands");
    assert_eq!(rank1.shape().dims(), &[4]);
    assert_close(
        "triplet/rank1/none",
        &values(&rank1),
        &[1.0, 1.3, 1.5000001, 1.3],
    );

    // Rank 3: everything after dim 0 is one flat feature vector, so the answer
    // must equal the rank-2 answer element for element.
    let a3 = leaf(&TRI_ANCHOR, &[2, 1, 2]);
    let p3 = leaf(&TRI_POSITIVE, &[2, 1, 2]);
    let n3 = leaf(&TRI_NEGATIVE, &[2, 1, 2]);
    let rank3 = functional::triplet_margin_loss(&a3, &p3, &n3, TRI_MARGIN, 2.0, Reduction::None)
        .expect("rank-3 operands");
    assert_eq!(rank3.shape().dims(), &[2]);
    assert_close(
        "triplet/rank3/none",
        &values(&rank3),
        &[0.85108006, 1.1443866],
    );
}

#[test]
fn ls_triplet_margin_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let anchor = param(&TRI_ANCHOR, &TRI_DIMS);
    let positive = leaf(&TRI_POSITIVE, &TRI_DIMS);
    let negative = leaf(&TRI_NEGATIVE, &TRI_DIMS);

    let loss =
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 2.0, Reduction::Sum)
            .expect("forward");
    assert!(
        loss.requires_grad(),
        "triplet_margin_loss must stay attached to its anchor"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&anchor, "triplet_margin_loss");

    let numeric = numeric_gradient(&TRI_ANCHOR, |x| {
        let probe = leaf(x, &TRI_DIMS);
        let positive = leaf(&TRI_POSITIVE, &TRI_DIMS);
        let negative = leaf(&TRI_NEGATIVE, &TRI_DIMS);
        values(
            &functional::triplet_margin_loss(&probe, &positive, &negative, TRI_MARGIN, 2.0, Reduction::Sum)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("triplet_margin_loss", &analytic, &numeric);
}

#[test]
fn ls_triplet_margin_loss_gradient_reaches_the_positive_and_negative_legs() {
    let _guard = grad_mode_guard();
    let anchor = leaf(&TRI_ANCHOR, &TRI_DIMS);
    let positive = param(&TRI_POSITIVE, &TRI_DIMS);
    let negative = param(&TRI_NEGATIVE, &TRI_DIMS);

    let loss =
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 2.0, Reduction::Sum)
            .expect("forward");
    loss.backward().expect("backward");
    let analytic_positive = gradient_of(&positive, "triplet/positive");
    let analytic_negative = gradient_of(&negative, "triplet/negative");

    let numeric_positive = numeric_gradient(&TRI_POSITIVE, |x| {
        let anchor = leaf(&TRI_ANCHOR, &TRI_DIMS);
        let probe = leaf(x, &TRI_DIMS);
        let negative = leaf(&TRI_NEGATIVE, &TRI_DIMS);
        values(
            &functional::triplet_margin_loss(&anchor, &probe, &negative, TRI_MARGIN, 2.0, Reduction::Sum)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("triplet/positive", &analytic_positive, &numeric_positive);

    let numeric_negative = numeric_gradient(&TRI_NEGATIVE, |x| {
        let anchor = leaf(&TRI_ANCHOR, &TRI_DIMS);
        let positive = leaf(&TRI_POSITIVE, &TRI_DIMS);
        let probe = leaf(x, &TRI_DIMS);
        values(
            &functional::triplet_margin_loss(&anchor, &positive, &probe, TRI_MARGIN, 2.0, Reduction::Sum)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("triplet/negative", &analytic_negative, &numeric_negative);
}

#[test]
fn ls_triplet_margin_loss_rejects_an_unknown_reduction() {
    let _guard = grad_mode_guard();
    let anchor = leaf(&TRI_ANCHOR, &TRI_DIMS);
    let positive = leaf(&TRI_POSITIVE, &TRI_DIMS);
    let negative = leaf(&TRI_NEGATIVE, &TRI_DIMS);
    assert!(
        functional::triplet_margin_loss(&anchor, &positive, &negative, TRI_MARGIN, 2.0, Reduction::BatchMean)
            .is_err(),
        "an unknown reduction must stay an error"
    );
}

#[test]
fn ls_triplet_margin_loss_rejects_a_rank0_operand() {
    let _guard = grad_mode_guard();
    let scalar = leaf(&[1.0], &[]);
    assert!(
        functional::triplet_margin_loss(&scalar, &scalar, &scalar, TRI_MARGIN, 2.0, Reduction::None)
            .is_err(),
        "a 0-D operand has no batch axis; that must be an error, not an index panic"
    );
}

/// Contrastive fixture: one similar pair (target `1`) and one dissimilar pair
/// (target `0`) whose distance `0.671` is well inside the `2.0` margin, so both
/// arms of the loss are active and neither sits on the hinge.
const CON_OUT1: [f32; 4] = [0.5, -0.3, 1.2, 0.7];
const CON_OUT2: [f32; 4] = [0.2, 0.4, 0.9, 0.1];
const CON_TARGET: [f32; 2] = [1.0, 0.0];
const CON_DIMS: [usize; 2] = [2, 2];
const CON_MARGIN: f32 = 2.0;

#[test]
fn ls_contrastive_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let out1 = leaf(&CON_OUT1, &CON_DIMS);
    let out2 = leaf(&CON_OUT2, &CON_DIMS);
    let target = leaf(&CON_TARGET, &[2]);

    let none =
        functional::contrastive_loss(&out1, &out2, &target, CON_MARGIN, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &[2]);
    assert_close("contrastive/none", &values(&none), &[0.58000004, 1.7667185]);

    let mean =
        functional::contrastive_loss(&out1, &out2, &target, CON_MARGIN, Reduction::Mean).expect("mean");
    assert_eq!(mean.shape().dims(), &[1]);
    assert_close("contrastive/mean", &values(&mean), &[1.1733593]);

    let sum = functional::contrastive_loss(&out1, &out2, &target, CON_MARGIN, Reduction::Sum).expect("sum");
    assert_eq!(sum.shape().dims(), &[1]);
    assert_close("contrastive/sum", &values(&sum), &[2.3467185]);

    // A dissimilar pair already beyond the margin contributes nothing, and a
    // similar pair's arm is unaffected by the margin.
    let far = leaf(&[0.5, -0.3, 9.0, 9.0], &CON_DIMS);
    let beyond =
        functional::contrastive_loss(&out1, &far, &target, CON_MARGIN, Reduction::None).expect("beyond");
    assert_close("contrastive/beyond-margin", &values(&beyond), &[0.0, 0.0]);
}

#[test]
fn ls_contrastive_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let out1 = param(&CON_OUT1, &CON_DIMS);
    let out2 = leaf(&CON_OUT2, &CON_DIMS);
    let target = leaf(&CON_TARGET, &[2]);

    let loss =
        functional::contrastive_loss(&out1, &out2, &target, CON_MARGIN, Reduction::Sum).expect("forward");
    assert!(
        loss.requires_grad(),
        "contrastive_loss must stay attached to its embeddings"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&out1, "contrastive_loss");

    let numeric = numeric_gradient(&CON_OUT1, |x| {
        let probe = leaf(x, &CON_DIMS);
        let out2 = leaf(&CON_OUT2, &CON_DIMS);
        let target = leaf(&CON_TARGET, &[2]);
        values(
            &functional::contrastive_loss(&probe, &out2, &target, CON_MARGIN, Reduction::Sum)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("contrastive_loss", &analytic, &numeric);
}

#[test]
fn ls_contrastive_loss_rejects_an_unknown_reduction() {
    let _guard = grad_mode_guard();
    let out1 = leaf(&CON_OUT1, &CON_DIMS);
    let out2 = leaf(&CON_OUT2, &CON_DIMS);
    let target = leaf(&CON_TARGET, &[2]);
    assert!(
        functional::contrastive_loss(&out1, &out2, &target, CON_MARGIN, Reduction::BatchMean).is_err(),
        "an unknown reduction must stay an error"
    );
}

/// Multi-margin fixture: three classes, both samples with every off-class term
/// at least `0.3` inside the hinge.
const MM_INPUT: [f32; 6] = [0.4, 1.5, -0.2, 2.0, 0.3, 0.9];
const MM_DIMS: [usize; 2] = [2, 3];
const MM_MARGIN: f32 = 2.0;

fn mm_target() -> torsh_tensor::Tensor<i64> {
    torsh_tensor::Tensor::<i64>::from_vec(vec![1i64, 0], &[2]).expect("target")
}

#[test]
fn ls_multi_margin_loss_forward_values_are_pinned() {
    let _guard = grad_mode_guard();
    let input = leaf(&MM_INPUT, &MM_DIMS);
    let target = mm_target();

    let none =
        functional::multi_margin_loss(&input, &target, 1, MM_MARGIN, None, Reduction::None).expect("none");
    assert_eq!(none.shape().dims(), &[2]);
    assert_close("multi_margin/p1/none", &values(&none), &[0.6, 0.6]);

    let mean =
        functional::multi_margin_loss(&input, &target, 1, MM_MARGIN, None, Reduction::Mean).expect("mean");
    assert!(
        mean.shape().dims().is_empty(),
        "this loss reduces through `apply_reduction`, which yields a 0-D scalar; \
         got {:?}",
        mean.shape().dims()
    );
    assert_close("multi_margin/p1/mean", &values(&mean), &[0.6]);

    let sum =
        functional::multi_margin_loss(&input, &target, 1, MM_MARGIN, None, Reduction::Sum).expect("sum");
    assert!(sum.shape().dims().is_empty());
    assert_close("multi_margin/p1/sum", &values(&sum), &[1.2]);

    let p2 =
        functional::multi_margin_loss(&input, &target, 2, MM_MARGIN, None, Reduction::None).expect("p2");
    assert_close("multi_margin/p2/none", &values(&p2), &[0.45, 0.45]);

    let weight = leaf(&[2.0, 3.0, 4.0], &[3]);
    let weighted =
        functional::multi_margin_loss(&input, &target, 1, MM_MARGIN, Some(&weight), Reduction::None)
            .expect("weighted");
    assert_close(
        "multi_margin/weighted/none",
        &values(&weighted),
        &[1.8000001, 1.2],
    );
}

#[test]
fn ls_multi_margin_loss_scores_an_out_of_range_target_as_zero() {
    let _guard = grad_mode_guard();
    let input = leaf(&MM_INPUT, &MM_DIMS);
    let target = torsh_tensor::Tensor::<i64>::from_vec(vec![1i64, 7], &[2]).expect("target");

    let none =
        functional::multi_margin_loss(&input, &target, 1, MM_MARGIN, None, Reduction::None).expect("none");
    assert_close("multi_margin/out-of-range", &values(&none), &[0.6, 0.0]);
}

#[test]
fn ls_multi_margin_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let input = param(&MM_INPUT, &MM_DIMS);
    let target = mm_target();

    let loss =
        functional::multi_margin_loss(&input, &target, 1, MM_MARGIN, None, Reduction::Sum).expect("forward");
    assert!(
        loss.requires_grad(),
        "multi_margin_loss must stay attached to its scores"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&input, "multi_margin_loss");

    let numeric = numeric_gradient(&MM_INPUT, |x| {
        let probe = leaf(x, &MM_DIMS);
        let target = mm_target();
        values(
            &functional::multi_margin_loss(&probe, &target, 1, MM_MARGIN, None, Reduction::Sum)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("multi_margin_loss", &analytic, &numeric);
}

#[test]
fn ls_multi_margin_loss_is_differentiable_with_squared_hinge_and_weights() {
    let _guard = grad_mode_guard();
    let input = param(&MM_INPUT, &MM_DIMS);
    let target = mm_target();
    let weight = leaf(&[2.0, 3.0, 4.0], &[3]);

    let loss = functional::multi_margin_loss(&input, &target, 2, MM_MARGIN, Some(&weight), Reduction::Sum)
        .expect("forward");
    assert!(loss.requires_grad());
    loss.backward().expect("backward");
    let analytic = gradient_of(&input, "multi_margin_loss/p2");

    let numeric = numeric_gradient(&MM_INPUT, |x| {
        let probe = leaf(x, &MM_DIMS);
        let target = mm_target();
        let weight = leaf(&[2.0, 3.0, 4.0], &[3]);
        values(
            &functional::multi_margin_loss(&probe, &target, 2, MM_MARGIN, Some(&weight), Reduction::Sum)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("multi_margin_loss/p2", &analytic, &numeric);
}

#[test]
fn ls_multi_margin_loss_rejects_a_non_2d_input() {
    let _guard = grad_mode_guard();
    let flat = leaf(&MM_INPUT, &[6]);
    let target = mm_target();
    assert!(
        functional::multi_margin_loss(&flat, &target, 1, MM_MARGIN, None, Reduction::None).is_err(),
        "a 1-D score tensor has no class axis; that must be an error, not an index panic"
    );
}

#[test]
fn ls_multi_margin_loss_rejects_a_degenerate_exponent() {
    let _guard = grad_mode_guard();
    let input = leaf(&MM_INPUT, &MM_DIMS);
    let target = mm_target();
    for p in [0, -1] {
        assert!(
            functional::multi_margin_loss(&input, &target, p, MM_MARGIN, None, Reduction::None).is_err(),
            "p = {p} cannot be expressed on the clamped hinge (`0^0 == 1`, `0^-1 == inf`) \
             and is outside PyTorch's documented domain, so it must be rejected rather than \
             silently scored differently from the predecessor"
        );
    }
    // Everything at or above 1 keeps working.
    for p in [1, 2, 3] {
        assert!(
            functional::multi_margin_loss(&input, &target, p, MM_MARGIN, None, Reduction::None).is_ok(),
            "p = {p} must stay available"
        );
    }
}

/// Cosine-embedding fixture: one `+1` pair and one `-1` pair whose cosine
/// `0.844` clears the `0.25` margin, so both arms are active.
const COS_IN1: [f32; 4] = [1.0, 2.0, 3.0, -1.0];
const COS_IN2: [f32; 4] = [2.0, 1.0, 4.0, 1.0];
const COS_TARGET: [f32; 2] = [1.0, -1.0];
const COS_DIMS: [usize; 2] = [2, 2];
const COS_MARGIN: f32 = 0.25;

#[test]
fn ls_cosine_embedding_loss_accepts_a_batched_target() {
    let _guard = grad_mode_guard();
    let input1 = leaf(&COS_IN1, &COS_DIMS);
    let input2 = leaf(&COS_IN2, &COS_DIMS);
    let target = leaf(&COS_TARGET, &[2]);

    let none = functional::cosine_embedding_loss(&input1, &input2, &target, COS_MARGIN, Reduction::None)
        .expect("a rank-1 target over a [N, D] pair is the documented PyTorch shape");
    assert_eq!(
        none.shape().dims(),
        &[2],
        "one loss per sample, cosine taken along the feature axis"
    );
    assert_close("cosine/none", &values(&none), &[0.2, 0.5936615]);

    let mean = functional::cosine_embedding_loss(&input1, &input2, &target, COS_MARGIN, Reduction::Mean)
        .expect("mean");
    assert_close("cosine/mean", &values(&mean), &[0.39683074]);

    let sum = functional::cosine_embedding_loss(&input1, &input2, &target, COS_MARGIN, Reduction::Sum)
        .expect("sum");
    assert_close("cosine/sum", &values(&sum), &[0.7936615]);
}

#[test]
fn ls_cosine_embedding_loss_preserves_the_unbatched_case() {
    let _guard = grad_mode_guard();
    // The one shape combination the predecessor served: 1-D operands with a 0-D
    // target. Its global cosine coincides with the per-sample cosine here, so
    // the pinned numbers come straight from the old implementation.
    let v1 = leaf(&[1.0, 2.0], &[2]);
    let v2 = leaf(&[2.0, 1.0], &[2]);
    let positive = leaf(&[1.0], &[]);
    let negative = leaf(&[-1.0], &[]);

    for reduction in [Reduction::None, Reduction::Mean, Reduction::Sum] {
        let loss = functional::cosine_embedding_loss(&v1, &v2, &positive, COS_MARGIN, reduction)
            .expect("unbatched");
        assert!(
            loss.shape().dims().is_empty(),
            "a 1-D pair produces a 0-D loss, as before; got {:?}",
            loss.shape().dims()
        );
        assert_close("cosine/unbatched/pos", &values(&loss), &[0.19999999]);
    }

    let neg = functional::cosine_embedding_loss(&v1, &v2, &negative, COS_MARGIN, Reduction::None)
        .expect("unbatched negative");
    assert_close("cosine/unbatched/neg", &values(&neg), &[0.55]);
}

#[test]
fn ls_cosine_embedding_loss_is_differentiable() {
    let _guard = grad_mode_guard();
    let input1 = param(&COS_IN1, &COS_DIMS);
    let input2 = leaf(&COS_IN2, &COS_DIMS);
    let target = leaf(&COS_TARGET, &[2]);

    let loss = functional::cosine_embedding_loss(&input1, &input2, &target, COS_MARGIN, Reduction::Sum)
        .expect("forward");
    assert!(
        loss.requires_grad(),
        "cosine_embedding_loss must stay attached to its embeddings"
    );
    loss.backward().expect("backward");
    let analytic = gradient_of(&input1, "cosine_embedding_loss");

    let numeric = numeric_gradient(&COS_IN1, |x| {
        let probe = leaf(x, &COS_DIMS);
        let input2 = leaf(&COS_IN2, &COS_DIMS);
        let target = leaf(&COS_TARGET, &[2]);
        values(
            &functional::cosine_embedding_loss(&probe, &input2, &target, COS_MARGIN, Reduction::Sum)
                .expect("probe"),
        )
        .iter()
        .sum()
    });
    assert_matches_finite_differences("cosine_embedding_loss", &analytic, &numeric);
}

#[test]
fn ls_cosine_embedding_loss_rejects_a_target_of_the_wrong_length() {
    let _guard = grad_mode_guard();
    let input1 = leaf(&COS_IN1, &COS_DIMS);
    let input2 = leaf(&COS_IN2, &COS_DIMS);
    let target = leaf(&[1.0, -1.0, 1.0], &[3]);
    assert!(
        functional::cosine_embedding_loss(&input1, &input2, &target, COS_MARGIN, Reduction::None).is_err(),
        "three labels for two samples must be an error"
    );
}
