//! Production-hardening regression tests for the torsh-nn softmax family and
//! the losses built on top of it.
//!
//! `functional::softmax` / `functional::log_softmax` used to hand-roll their
//! slice-wise kernel and hand the result to `Tensor::from_data`, which returns a
//! fresh detached leaf. Every consumer therefore fell off the autograd graph:
//! `cross_entropy` composed the textbook `-sum(one_hot * log_softmax(x))`
//! expression but `backward()` on it failed outright with "Called backward on
//! tensor that doesn't require grad".
//!
//! These tests pin the graph connectivity down with finite differences, and pin
//! the *values* down against an independent reference kernel so the delegation
//! cannot silently change what the functions compute.

use torsh_core::{Reduction, error::Result};
use torsh_nn::functional;
use torsh_tensor::Tensor;

/// Finite-difference step and tolerance mandated for these gradient checks.
const FD_STEP: f32 = 1e-2;

/// Tolerance for a single finite-difference comparison.
///
/// Relative for large gradients, absolute for small ones, so a component whose
/// true gradient is ~0 is not held to an impossible relative bound.
fn fd_tolerance(numeric: f32) -> f32 {
    2e-2 * numeric.abs().max(1.0)
}

/// Evaluate `f` at `data` and collapse the result to a single scalar.
///
/// Every function under test here already returns a scalar (or a one-element
/// tensor); summing is the identity in that case and keeps the helper usable for
/// the `"none"` reductions too.
fn scalar_value<F>(data: &[f32], dims: &[usize], f: &F) -> f32
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let input = Tensor::from_vec(data.to_vec(), dims).expect("finite-difference input");
    let output = f(&input).expect("finite-difference forward pass");
    output
        .to_vec()
        .expect("finite-difference output")
        .iter()
        .sum()
}

/// Central finite differences of `f` with respect to every element of `data`.
fn numeric_gradient<F>(data: &[f32], dims: &[usize], f: F) -> Vec<f32>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    (0..data.len())
        .map(|i| {
            let mut plus = data.to_vec();
            let mut minus = data.to_vec();
            plus[i] += FD_STEP;
            minus[i] -= FD_STEP;
            (scalar_value(&plus, dims, &f) - scalar_value(&minus, dims, &f)) / (2.0 * FD_STEP)
        })
        .collect()
}

/// Assert that an analytic gradient matches central finite differences.
fn assert_matches_finite_differences(label: &str, analytic: &[f32], numeric: &[f32]) {
    assert_eq!(
        analytic.len(),
        numeric.len(),
        "{label}: gradient length {} does not match the {} inputs perturbed",
        analytic.len(),
        numeric.len()
    );
    for (i, (&got, &want)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let tol = fd_tolerance(want);
        assert!(
            (got - want).abs() <= tol,
            "{label}: gradient[{i}] = {got}, finite differences gave {want} (tolerance {tol})\n\
             analytic = {analytic:?}\nnumeric  = {numeric:?}"
        );
    }
}

/// Independent slice-wise (log-)softmax reference, written straight from the
/// definition. This is deliberately *not* the implementation under test: it is
/// the value oracle that proves the delegation preserved both the numbers and
/// the max-subtraction stability property.
fn reference_softmax(data: &[f32], dims: &[usize], dim: usize, logarithmic: bool) -> Vec<f32> {
    let dim_size = dims[dim];
    let outer: usize = dims[..dim].iter().product();
    let inner: usize = dims[dim + 1..].iter().product();

    let mut out = vec![0.0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let base = o * dim_size * inner + i;
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim_size {
                let value = data[base + d * inner];
                if value > max_val {
                    max_val = value;
                }
            }
            let mut sum_exp = 0.0f32;
            for d in 0..dim_size {
                sum_exp += (data[base + d * inner] - max_val).exp();
            }
            for d in 0..dim_size {
                let idx = base + d * inner;
                out[idx] = if logarithmic {
                    data[idx] - max_val - sum_exp.ln()
                } else {
                    (data[idx] - max_val).exp() / sum_exp
                };
            }
        }
    }
    out
}

/// A fixed, non-uniform, strictly-positive covector used to turn a tensor-valued
/// activation into a scalar objective. Non-uniform on purpose: contracting a
/// softmax with a *constant* vector gives an identically zero gradient (softmax
/// rows sum to 1), which would make the finite-difference check vacuous.
fn probe_covector(numel: usize, dims: &[usize]) -> Tensor {
    let data: Vec<f32> = (0..numel).map(|i| 0.3 + 0.17 * (i as f32)).collect();
    Tensor::from_vec(data, dims).expect("probe covector")
}

// ---------------------------------------------------------------------------
// NN-LS (a) - cross_entropy must be differentiable end to end
// ---------------------------------------------------------------------------

#[test]
fn ls_cross_entropy_backward_matches_finite_differences() {
    let dims = [2usize, 3];
    let logits_data = vec![0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5];
    let target = Tensor::<i64>::from_vec(vec![2i64, 0], &[2]).expect("target");

    let logits = Tensor::from_vec(logits_data.clone(), &dims)
        .expect("logits")
        .requires_grad_(true);
    let loss = functional::cross_entropy(&logits, &target, None, Reduction::Mean, None).expect("forward");
    assert!(
        loss.requires_grad(),
        "cross_entropy must stay attached to its logits; a detached log_softmax \
         makes the whole loss non-differentiable"
    );

    loss.backward()
        .expect("backward on a cross_entropy loss must succeed");
    let analytic = logits
        .grad()
        .expect("cross_entropy must populate the logits gradient")
        .to_vec()
        .expect("gradient values");

    let target_for_fd = target.clone();
    let numeric = numeric_gradient(&logits_data, &dims, move |x| {
        functional::cross_entropy(x, &target_for_fd, None, Reduction::Mean, None)
    });
    assert_matches_finite_differences("cross_entropy", &analytic, &numeric);
}

#[test]
fn ls_cross_entropy_backward_matches_finite_differences_with_weights() {
    let dims = [3usize, 4];
    let logits_data: Vec<f32> = vec![
        0.4, -0.7, 1.3, 0.2, -1.1, 0.6, 0.05, 2.0, 0.9, 0.3, -0.4, -1.6,
    ];
    let target = Tensor::<i64>::from_vec(vec![1i64, 3, 0], &[3]).expect("target");
    let weight = Tensor::from_vec(vec![0.5f32, 2.0, 1.0, 0.25], &[4]).expect("class weights");

    let logits = Tensor::from_vec(logits_data.clone(), &dims)
        .expect("logits")
        .requires_grad_(true);
    let loss =
        functional::cross_entropy(&logits, &target, Some(&weight), Reduction::Sum, None).expect("forward");
    assert!(loss.requires_grad(), "weighted cross_entropy must record");

    loss.backward().expect("backward");
    let analytic = logits
        .grad()
        .expect("logits gradient")
        .to_vec()
        .expect("gradient values");

    let target_for_fd = target.clone();
    let weight_for_fd = weight.clone();
    let numeric = numeric_gradient(&logits_data, &dims, move |x| {
        functional::cross_entropy(x, &target_for_fd, Some(&weight_for_fd), Reduction::Sum, None)
    });
    assert_matches_finite_differences("weighted cross_entropy", &analytic, &numeric);
}

// ---------------------------------------------------------------------------
// NN-LS (b) - the softmax family itself must stay on the graph
// ---------------------------------------------------------------------------

#[test]
fn ls_softmax_output_requires_grad() {
    let input = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5], &[2, 3])
        .expect("input")
        .requires_grad_(true);
    assert!(
        functional::softmax(&input, Some(-1))
            .expect("softmax")
            .requires_grad(),
        "functional::softmax must propagate requires_grad"
    );
    assert!(
        functional::log_softmax(&input, Some(-1))
            .expect("log_softmax")
            .requires_grad(),
        "functional::log_softmax must propagate requires_grad"
    );
}

#[test]
fn ls_softmax_leaves_detached_inputs_detached() {
    let input = Tensor::from_vec(vec![0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5], &[2, 3]).expect("input");
    assert!(
        !functional::softmax(&input, Some(-1))
            .expect("softmax")
            .requires_grad(),
        "softmax of a plain tensor must stay a detached leaf"
    );
    assert!(
        !functional::log_softmax(&input, Some(-1))
            .expect("log_softmax")
            .requires_grad(),
        "log_softmax of a plain tensor must stay a detached leaf"
    );
}

#[test]
fn ls_softmax_backward_matches_finite_differences_on_a_middle_dim() {
    // A middle axis is the discriminating case: it exercises keepdim reductions
    // and the broadcast back out to the full shape, not just a trailing axis.
    let dims = [2usize, 3, 2];
    let data: Vec<f32> = vec![
        0.3, -1.2, 0.8, 1.7, -0.4, 0.05, 1.1, -0.9, 0.2, 0.6, -1.5, 0.45,
    ];
    let covector = probe_covector(data.len(), &dims);

    let input = Tensor::from_vec(data.clone(), &dims)
        .expect("input")
        .requires_grad_(true);
    let output = functional::softmax(&input, Some(1)).expect("softmax");
    assert!(
        output.requires_grad(),
        "softmax must record on a middle dim"
    );
    let objective = output
        .mul_op(&covector)
        .expect("contract")
        .sum()
        .expect("scalar objective");
    objective.backward().expect("backward");
    let analytic = input
        .grad()
        .expect("softmax must populate the input gradient")
        .to_vec()
        .expect("gradient values");

    let covector_for_fd = covector.clone();
    let numeric = numeric_gradient(&data, &dims, move |x| {
        functional::softmax(x, Some(1))?
            .mul_op(&covector_for_fd)?
            .sum()
    });
    assert_matches_finite_differences("softmax(dim=1)", &analytic, &numeric);
}

#[test]
fn ls_log_softmax_backward_matches_finite_differences_on_a_middle_dim() {
    let dims = [2usize, 3, 2];
    let data: Vec<f32> = vec![
        0.3, -1.2, 0.8, 1.7, -0.4, 0.05, 1.1, -0.9, 0.2, 0.6, -1.5, 0.45,
    ];
    let covector = probe_covector(data.len(), &dims);

    let input = Tensor::from_vec(data.clone(), &dims)
        .expect("input")
        .requires_grad_(true);
    let output = functional::log_softmax(&input, Some(1)).expect("log_softmax");
    assert!(
        output.requires_grad(),
        "log_softmax must record on a middle dim"
    );
    let objective = output
        .mul_op(&covector)
        .expect("contract")
        .sum()
        .expect("scalar objective");
    objective.backward().expect("backward");
    let analytic = input
        .grad()
        .expect("log_softmax must populate the input gradient")
        .to_vec()
        .expect("gradient values");

    let covector_for_fd = covector.clone();
    let numeric = numeric_gradient(&data, &dims, move |x| {
        functional::log_softmax(x, Some(1))?
            .mul_op(&covector_for_fd)?
            .sum()
    });
    assert_matches_finite_differences("log_softmax(dim=1)", &analytic, &numeric);
}

// ---------------------------------------------------------------------------
// NN-LS (c) - nll_loss must be differentiable end to end
// ---------------------------------------------------------------------------

#[test]
fn ls_nll_loss_backward_matches_finite_differences() {
    let dims = [3usize, 4];
    let log_prob_data: Vec<f32> = vec![
        -1.4, -0.9, -2.1, -1.1, -0.6, -1.9, -1.3, -1.7, -2.4, -0.8, -1.05, -1.6,
    ];
    let target = Tensor::<i64>::from_vec(vec![2i64, 0, 3], &[3]).expect("target");

    let log_probs = Tensor::from_vec(log_prob_data.clone(), &dims)
        .expect("log probs")
        .requires_grad_(true);
    let loss = functional::nll_loss(&log_probs, &target, None, None, Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "nll_loss must stay attached to its log-probabilities"
    );

    loss.backward().expect("backward on nll_loss must succeed");
    let analytic = log_probs
        .grad()
        .expect("nll_loss must populate the input gradient")
        .to_vec()
        .expect("gradient values");

    let target_for_fd = target.clone();
    let numeric = numeric_gradient(&log_prob_data, &dims, move |x| {
        functional::nll_loss(x, &target_for_fd, None, None, Reduction::Mean)
    });
    assert_matches_finite_differences("nll_loss", &analytic, &numeric);
}

#[test]
fn ls_nll_loss_backward_matches_finite_differences_with_weight_and_ignore_index() {
    let dims = [4usize, 3];
    let log_prob_data: Vec<f32> = vec![
        -1.2, -0.7, -1.9, -0.5, -2.2, -1.4, -1.8, -1.1, -0.6, -0.95, -1.35, -1.5,
    ];
    // Sample 2 is ignored; sample 3 has an out-of-range class index and must
    // contribute exactly zero, as it did before the rewrite.
    let target = Tensor::<i64>::from_vec(vec![1i64, 2, 0, 7], &[4]).expect("target");
    let weight = Tensor::from_vec(vec![0.25f32, 1.5, 0.75], &[3]).expect("class weights");

    let log_probs = Tensor::from_vec(log_prob_data.clone(), &dims)
        .expect("log probs")
        .requires_grad_(true);
    let loss =
        functional::nll_loss(&log_probs, &target, Some(&weight), Some(0), Reduction::Sum).expect("forward");
    assert!(loss.requires_grad(), "weighted nll_loss must record");

    loss.backward().expect("backward");
    let analytic = log_probs
        .grad()
        .expect("input gradient")
        .to_vec()
        .expect("gradient values");

    let target_for_fd = target.clone();
    let weight_for_fd = weight.clone();
    let numeric = numeric_gradient(&log_prob_data, &dims, move |x| {
        functional::nll_loss(x, &target_for_fd, Some(&weight_for_fd), Some(0), Reduction::Sum)
    });
    assert_matches_finite_differences("weighted nll_loss", &analytic, &numeric);
}

/// `reduction="mean"` combined with an `ignore_index` is the PyTorch default and
/// the only reduction branch that does not go through `Tensor::mean`: it divides
/// by the *valid* sample count with a freshly built `[1]`-shaped constant. Its
/// gradient route is therefore distinct from every other case covered above.
#[test]
fn ls_cross_entropy_mean_with_ignore_index_matches_finite_differences() {
    let dims = [3usize, 3];
    let logits_data = vec![0.5f32, -1.0, 2.0, 0.25, 1.5, -0.5, -0.75, 0.4, 1.1];
    // Sample 1 carries the ignored class and must contribute no loss, no
    // gradient, and no entry in the mean's denominator.
    let target = Tensor::<i64>::from_vec(vec![2i64, 1, 0], &[3]).expect("target");

    let logits = Tensor::from_vec(logits_data.clone(), &dims)
        .expect("logits")
        .requires_grad_(true);
    let loss = functional::cross_entropy(&logits, &target, None, Reduction::Mean, Some(1)).expect("forward");
    assert!(
        loss.requires_grad(),
        "cross_entropy must record through the valid-count mean branch"
    );

    loss.backward().expect("backward");
    let analytic = logits
        .grad()
        .expect("logits gradient")
        .to_vec()
        .expect("gradient values");

    let target_for_fd = target.clone();
    let numeric = numeric_gradient(&logits_data, &dims, move |x| {
        functional::cross_entropy(x, &target_for_fd, None, Reduction::Mean, Some(1))
    });
    assert_matches_finite_differences("cross_entropy(mean, ignore_index)", &analytic, &numeric);

    for (i, &value) in analytic.iter().enumerate().skip(3).take(3) {
        assert_eq!(
            value, 0.0,
            "the ignored sample must receive an exactly zero gradient, got {value} at {i}"
        );
    }
}

#[test]
fn ls_nll_loss_mean_with_ignore_index_matches_finite_differences() {
    let dims = [3usize, 3];
    let log_prob_data = vec![-1.4f32, -0.9, -2.1, -0.6, -1.9, -1.3, -2.4, -0.8, -1.05];
    let target = Tensor::<i64>::from_vec(vec![2i64, 1, 0], &[3]).expect("target");

    let log_probs = Tensor::from_vec(log_prob_data.clone(), &dims)
        .expect("log probs")
        .requires_grad_(true);
    let loss = functional::nll_loss(&log_probs, &target, None, Some(1), Reduction::Mean).expect("forward");
    assert!(
        loss.requires_grad(),
        "nll_loss must record through the valid-count mean branch"
    );

    loss.backward().expect("backward");
    let analytic = log_probs
        .grad()
        .expect("input gradient")
        .to_vec()
        .expect("gradient values");

    let target_for_fd = target.clone();
    let numeric = numeric_gradient(&log_prob_data, &dims, move |x| {
        functional::nll_loss(x, &target_for_fd, None, Some(1), Reduction::Mean)
    });
    assert_matches_finite_differences("nll_loss(mean, ignore_index)", &analytic, &numeric);
}

/// Pins the documented cost of making `nll_loss` differentiable.
///
/// The selector gather is a masked sum over the whole row, so a non-finite entry
/// anywhere in a row poisons that row even when the *selected* class is finite.
/// The element-wise predecessor read only the selected entry and was immune.
/// This test exists so the trade-off is a deliberate, visible contract rather
/// than a surprise; if a future change restores immunity, update it.
#[test]
fn ls_nll_loss_non_finite_entry_poisons_only_its_own_row() {
    let log_probs = Tensor::from_vec(
        vec![f32::NEG_INFINITY, -0.7, -1.9, -0.5, -2.2, -1.4],
        &[2, 3],
    )
    .expect("log probs");
    let target = Tensor::<i64>::from_vec(vec![1i64, 0], &[2]).expect("target");

    let per_sample = functional::nll_loss(&log_probs, &target, None, None, Reduction::None)
        .expect("forward")
        .to_vec()
        .expect("values");

    assert!(
        per_sample[0].is_nan(),
        "a -inf anywhere in row 0 poisons that row's masked sum, got {}",
        per_sample[0]
    );
    assert!(
        (per_sample[1] - 0.5).abs() < 1e-6,
        "a clean row must be unaffected, got {}",
        per_sample[1]
    );
}

// ---------------------------------------------------------------------------
// Value and error-contract invariance - the delegation must change nothing
// but the graph
// ---------------------------------------------------------------------------

#[test]
fn ls_softmax_values_match_the_reference_kernel() {
    let dims = [2usize, 3, 4];
    let data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.25 - 2.0).collect();

    for dim in 0..3usize {
        let input = Tensor::from_vec(data.clone(), &dims).expect("input");

        let soft = functional::softmax(&input, Some(dim as i32))
            .expect("softmax")
            .to_vec()
            .expect("values");
        let want_soft = reference_softmax(&data, &dims, dim, false);
        for (i, (&got, &want)) in soft.iter().zip(want_soft.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "softmax(dim={dim})[{i}] = {got}, reference = {want}"
            );
        }

        let logged = functional::log_softmax(&input, Some(dim as i32))
            .expect("log_softmax")
            .to_vec()
            .expect("values");
        let want_log = reference_softmax(&data, &dims, dim, true);
        for (i, (&got, &want)) in logged.iter().zip(want_log.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "log_softmax(dim={dim})[{i}] = {got}, reference = {want}"
            );
        }
    }
}

#[test]
fn ls_softmax_stays_finite_for_huge_logits() {
    // Without the max subtraction exp() overflows to +inf and the ratio becomes
    // NaN. This guards the stability property through the delegation.
    let input = Tensor::from_vec(vec![1000.0f32, 1000.5, 999.0, -1000.0], &[2, 2]).expect("input");

    let soft = functional::softmax(&input, Some(1))
        .expect("softmax")
        .to_vec()
        .expect("values");
    assert!(
        soft.iter().all(|v| v.is_finite()),
        "softmax must stay finite for huge logits, got {soft:?}"
    );
    assert!((soft[0] + soft[1] - 1.0).abs() < 1e-5);
    assert!((soft[2] + soft[3] - 1.0).abs() < 1e-5);

    let logged = functional::log_softmax(&input, Some(1))
        .expect("log_softmax")
        .to_vec()
        .expect("values");
    assert!(
        logged.iter().all(|v| v.is_finite()),
        "log_softmax must stay finite for huge logits, got {logged:?}"
    );
    assert!((logged[0].exp() + logged[1].exp() - 1.0).abs() < 1e-5);
}

#[test]
fn ls_softmax_rejects_out_of_range_dims() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).expect("input");

    for dim in [2i32, 7, -3, -9] {
        assert!(
            functional::softmax(&input, Some(dim)).is_err(),
            "softmax must reject dim {dim} on a 2-dimensional tensor"
        );
        assert!(
            functional::log_softmax(&input, Some(dim)).is_err(),
            "log_softmax must reject dim {dim} on a 2-dimensional tensor"
        );
    }
}

#[test]
fn ls_softmax_defaults_to_the_last_dim() {
    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("input");
    let defaulted = functional::softmax(&input, None)
        .expect("softmax")
        .to_vec()
        .expect("values");
    let explicit = functional::softmax(&input, Some(-1))
        .expect("softmax")
        .to_vec()
        .expect("values");
    assert_eq!(defaulted, explicit);

    let defaulted_log = functional::log_softmax(&input, None)
        .expect("log_softmax")
        .to_vec()
        .expect("values");
    let explicit_log = functional::log_softmax(&input, Some(-1))
        .expect("log_softmax")
        .to_vec()
        .expect("values");
    assert_eq!(defaulted_log, explicit_log);
}
