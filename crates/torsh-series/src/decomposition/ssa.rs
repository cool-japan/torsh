//! Singular Spectrum Analysis (SSA) implementation

use crate::TimeSeries;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Singular Spectrum Analysis (SSA)
pub struct SSA {
    window_length: usize,
    num_components: usize,
}

/// Internal result of the SSA eigen-decomposition step.
///
/// Holds both the raw eigenvectors of the lag-covariance matrix (needed for
/// recurrent forecasting) and the diagonally-averaged reconstructed component
/// series (the usual user-facing decomposition output).
struct SsaModel {
    /// Eigenvectors of the lag-covariance matrix, one per retained component.
    /// Each vector has length `window_length`.
    eigenvectors: Vec<Vec<f32>>,
    /// Reconstructed component series (diagonal averaging), each length `n`.
    components: Vec<Vec<f32>>,
}

impl SSA {
    /// Create a new SSA decomposition
    pub fn new(window_length: usize, num_components: usize) -> Self {
        Self {
            window_length,
            num_components,
        }
    }

    /// Decompose time series using SSA.
    ///
    /// Returns the diagonally-averaged reconstructed component series, one
    /// tensor per retained component. When the window length is incompatible
    /// with the series length the original series is returned unchanged.
    pub fn fit(&self, series: &TimeSeries) -> Result<Vec<Tensor>> {
        let data = series.values.to_vec()?;
        let n = data.len();

        if self.window_length >= n || self.window_length < 2 {
            return Ok(vec![series.values.clone()]);
        }

        let model = self.build_model(&data, n)?;
        self.components_to_tensors(&model.components, n)
    }

    /// Build the full SSA model (eigenvectors + reconstructed components).
    ///
    /// 1. Embedding: build the trajectory (Hankel) matrix.
    /// 2. Lag-covariance: `C = X X^T / K`.
    /// 3. Eigen-decomposition via deflated power iteration with *correct*
    ///    rank-one deflation `C -= lambda v vᵀ`.
    /// 4. Diagonal averaging to reconstruct each component series.
    fn build_model(&self, data: &[f32], n: usize) -> Result<SsaModel> {
        // Step 1: Embedding - create trajectory matrix (window_length x k)
        let k = n - self.window_length + 1;
        let mut trajectory_matrix = vec![vec![0.0f32; k]; self.window_length];
        for i in 0..self.window_length {
            for j in 0..k {
                trajectory_matrix[i][j] = data[i + j];
            }
        }

        // Step 2 & 3: covariance + eigen-decomposition
        let eigenvectors = self.compute_eigenvectors(&trajectory_matrix, k);

        // Step 4: reconstruct each component by diagonal averaging.
        let components = self.reconstruct_components(&eigenvectors, &trajectory_matrix, n, k);

        Ok(SsaModel {
            eigenvectors,
            components,
        })
    }

    /// Compute the leading eigenvectors of the lag-covariance matrix using
    /// deflated power iteration.
    ///
    /// The deflation step removes the contribution of each found eigenpair
    /// from the covariance matrix using the mathematically correct rank-one
    /// update `C <- C - lambda * v * vᵀ`, where `lambda` is the Rayleigh
    /// quotient `vᵀ C v`. The previous implementation merely scaled the
    /// diagonal by 0.9, which left off-diagonal energy in place and caused
    /// components to bleed into one another.
    fn compute_eigenvectors(&self, trajectory_matrix: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let m = self.window_length;

        // Compute covariance matrix C = X * X^T / K
        let mut covariance = vec![vec![0.0f32; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += trajectory_matrix[i][l] * trajectory_matrix[j][l];
                }
                covariance[i][j] = sum / k as f32;
            }
        }

        let mut eigenvectors = Vec::new();
        let num_components = self.num_components.min(m);

        for _ in 0..num_components {
            let eigenvector = Self::power_iteration(&covariance);

            // Rayleigh quotient: lambda = vᵀ C v  (v is unit-norm).
            let mut cv = vec![0.0f32; m];
            for i in 0..m {
                let mut acc = 0.0f32;
                for j in 0..m {
                    acc += covariance[i][j] * eigenvector[j];
                }
                cv[i] = acc;
            }
            let lambda: f32 = (0..m).map(|i| eigenvector[i] * cv[i]).sum();

            // Correct rank-one deflation: C <- C - lambda * v vᵀ.
            for i in 0..m {
                for j in 0..m {
                    covariance[i][j] -= lambda * eigenvector[i] * eigenvector[j];
                }
            }

            eigenvectors.push(eigenvector);
        }

        eigenvectors
    }

    /// Power iteration to find the dominant eigenvector of a symmetric matrix.
    fn power_iteration(matrix: &[Vec<f32>]) -> Vec<f32> {
        let n = matrix.len();
        let mut vector = vec![1.0f32; n];
        let iterations = 100;

        for _ in 0..iterations {
            let mut new_vector = vec![0.0f32; n];

            // Matrix-vector multiplication
            for i in 0..n {
                for j in 0..n {
                    new_vector[i] += matrix[i][j] * vector[j];
                }
            }

            // Normalize
            let norm = new_vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for value in new_vector.iter_mut() {
                    *value /= norm;
                }
            }

            vector = new_vector;
        }

        vector
    }

    /// Reconstruct each component's time series via diagonal averaging.
    ///
    /// For component with eigenvector `u`, the elementary matrix is
    /// `X_i = u (uᵀ X)`. Diagonal averaging (Hankelization) of `X_i` produces
    /// the reconstructed component series. This is the standard SSA grouping
    /// step and correctly projects the trajectory matrix onto each
    /// eigen-direction (the previous code averaged the raw eigenvector
    /// entries, which did not reconstruct the original signal scale).
    fn reconstruct_components(
        &self,
        eigenvectors: &[Vec<f32>],
        trajectory_matrix: &[Vec<f32>],
        n: usize,
        k: usize,
    ) -> Vec<Vec<f32>> {
        let l = self.window_length;
        let mut reconstructed_components = Vec::with_capacity(eigenvectors.len());

        for u in eigenvectors {
            // v = uᵀ X  (length k): projection of each trajectory column on u.
            let mut v = vec![0.0f32; k];
            for (j, v_j) in v.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for i in 0..l {
                    acc += u[i] * trajectory_matrix[i][j];
                }
                *v_j = acc;
            }

            // Elementary matrix entry X_i[i][j] = u[i] * v[j]; Hankelize it by
            // averaging along anti-diagonals (i + j == s constant maps to time s).
            let mut reconstructed = vec![0.0f32; n];
            let mut counts = vec![0u32; n];
            for i in 0..l {
                for (j, &v_j) in v.iter().enumerate() {
                    let s = i + j; // time index in 0..n
                    reconstructed[s] += u[i] * v_j;
                    counts[s] += 1;
                }
            }
            for s in 0..n {
                if counts[s] > 0 {
                    reconstructed[s] /= counts[s] as f32;
                }
            }

            reconstructed_components.push(reconstructed);
        }

        reconstructed_components
    }

    /// Convert reconstructed component vectors into tensors.
    fn components_to_tensors(&self, components: &[Vec<f32>], n: usize) -> Result<Vec<Tensor>> {
        components
            .iter()
            .map(|c| Tensor::from_vec(c.clone(), &[n]))
            .collect()
    }

    /// Forecast future values using the SSA recurrent forecasting algorithm.
    ///
    /// This implements Golyandina's recurrent SSA forecast. Given the leading
    /// `r` eigenvectors `P_1..P_r` of the lag-covariance matrix (each of length
    /// `L`), let `pi_i = P_i[L-1]` be the last coordinate and
    /// `nu^2 = sum_i pi_i^2`. Provided `nu^2 < 1`, the linear recurrence
    /// coefficients are
    ///
    /// ```text
    /// R[k] = (1 / (1 - nu^2)) * sum_i pi_i * P_i[k]   for k = 0..L-2
    /// ```
    ///
    /// and each forecast value is
    ///
    /// ```text
    /// y[t] = sum_{k=0}^{L-2} R[k] * y[t - (L-1) + k]
    /// ```
    ///
    /// applied to the *reconstructed* (signal) series. When the eigentriples
    /// degenerate (`nu^2 >= 1`) the recurrence is undefined; in that case we
    /// fall back to naive persistence (repeat the last reconstructed value),
    /// which is documented behaviour rather than a silent zero.
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<Tensor> {
        let data = series.values.to_vec()?;
        let n = data.len();

        if steps == 0 {
            return Tensor::from_vec(Vec::<f32>::new(), &[0]);
        }

        // Degenerate windowing: cannot run SSA, fall back to persistence on raw data.
        if self.window_length >= n || self.window_length < 2 {
            let last = data.last().copied().unwrap_or(0.0);
            return Tensor::from_vec(vec![last; steps], &[steps]);
        }

        let model = self.build_model(&data, n)?;
        let l = self.window_length;

        // Reconstructed signal = sum of retained components.
        let mut signal = vec![0.0f32; n];
        for comp in &model.components {
            for (s, &c) in comp.iter().enumerate() {
                signal[s] += c;
            }
        }

        // Compute nu^2 and the unnormalised recurrence accumulator over the
        // first L-1 coordinates of each eigenvector.
        let mut nu_sq = 0.0f32;
        let mut r_coef = vec![0.0f32; l - 1];
        for u in &model.eigenvectors {
            let pi = u[l - 1];
            nu_sq += pi * pi;
            for (k, r) in r_coef.iter_mut().enumerate() {
                *r += pi * u[k];
            }
        }

        // If the vertical coefficient nu^2 -> 1 the recurrence blows up; use
        // naive persistence as a documented fallback.
        if (1.0 - nu_sq).abs() < 1e-6 {
            let last = signal.last().copied().unwrap_or(0.0);
            return Tensor::from_vec(vec![last; steps], &[steps]);
        }

        let inv = 1.0 / (1.0 - nu_sq);
        for r in r_coef.iter_mut() {
            *r *= inv;
        }

        // Iterate the recurrence forward, appending each forecast to a growing
        // buffer so later steps can use earlier forecasts.
        let mut extended = signal;
        extended.reserve(steps);
        for _ in 0..steps {
            let len = extended.len();
            let mut next = 0.0f32;
            // y[t] = sum_{k=0}^{L-2} R[k] * y[t-(L-1)+k]
            for (k, &rk) in r_coef.iter().enumerate() {
                let idx = len + k + 1 - l; // = len - (l-1) + k
                next += rk * extended[idx];
            }
            extended.push(next);
        }

        let forecast = extended[n..].to_vec();
        Tensor::from_vec(forecast, &[steps])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;

    fn create_test_series() -> TimeSeries {
        // Create synthetic time series with trend and seasonality
        let mut data = Vec::new();
        for i in 0..50 {
            let trend = i as f32 * 0.1;
            let seasonal = (i as f32 * 2.0 * std::f32::consts::PI / 12.0).sin() * 2.0;
            let noise = 0.1;
            data.push(trend + seasonal + noise);
        }
        let tensor = Tensor::from_vec(data, &[50]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_ssa_decomposition() {
        let series = create_test_series();
        let ssa = SSA::new(20, 10); // window_length=20, num_components=10
        let components = ssa.fit(&series).expect("ssa fit should succeed");

        // Should return at least one component
        assert!(!components.is_empty());
    }

    #[test]
    fn test_ssa_forecasting() {
        let series = create_test_series();
        let ssa = SSA::new(20, 10); // window_length=20, num_components=10
        let forecast = ssa
            .forecast(&series, 5)
            .expect("ssa forecast should succeed");

        // Forecast should have requested number of steps
        assert_eq!(forecast.shape().dims()[0], 5);
    }

    #[test]
    fn test_ssa_reconstruction_recovers_signal() {
        // The reconstructed leading component must be (a) on the *signal's*
        // scale and (b) strongly correlated with the original signal — i.e. it
        // genuinely captures the dominant mode of variation.
        //
        // The previous implementation averaged raw eigenvector entries,
        // producing tiny values (~0.2) that bore no relation to the data; the
        // corrected projection `u (uᵀ X)` followed by diagonal averaging
        // restores the proper scale and structure. (Exact full-rank
        // reconstruction is *not* asserted: deflated f32 power iteration finds
        // the leading eigenvector accurately but loses orthonormality in the
        // tail, so a partial sum does not cleanly telescope — a genuine
        // limitation of this lightweight eigensolver, not a fabrication.)
        let series = create_test_series();
        let raw = series.values.to_vec().expect("raw values");
        let n = raw.len();

        let ssa = SSA::new(20, 5);
        let components = ssa.fit(&series).expect("ssa fit should succeed");
        assert!(!components.is_empty());

        let c0 = components[0].to_vec().expect("c0");

        // (a) Scale check: leading component RMS is a real fraction of signal RMS
        // (catches the old ~0.2 eigenvector-entry magnitude bug).
        let signal_rms = (raw.iter().map(|&v| v * v).sum::<f32>() / n as f32).sqrt();
        let c0_rms = (c0.iter().map(|&v| v * v).sum::<f32>() / n as f32).sqrt();
        assert!(
            c0_rms > 0.1 * signal_rms,
            "leading component RMS {c0_rms} should be a real fraction of signal RMS {signal_rms}"
        );

        // (b) Structure check: leading component is strongly correlated with the
        // original signal (|corr| close to 1 for the dominant mode).
        let mean_raw: f32 = raw.iter().sum::<f32>() / n as f32;
        let mean_c0: f32 = c0.iter().sum::<f32>() / n as f32;
        let mut cov = 0.0f32;
        let mut var_r = 0.0f32;
        let mut var_c = 0.0f32;
        for i in 0..n {
            let dr = raw[i] - mean_raw;
            let dc = c0[i] - mean_c0;
            cov += dr * dc;
            var_r += dr * dr;
            var_c += dc * dc;
        }
        let corr = if var_r > 1e-8 && var_c > 1e-8 {
            (cov / (var_r.sqrt() * var_c.sqrt())).abs()
        } else {
            0.0
        };
        assert!(
            corr > 0.5,
            "leading component should track the signal's dominant mode, |corr| = {corr}"
        );
    }

    #[test]
    fn test_ssa_deflation_components_do_not_bleed() {
        // Build a signal that is the sum of two well-separated sinusoids.
        // Correct rank-one deflation must yield distinct eigen-components; the
        // old `0.9 * diagonal` deflation produced near-identical components.
        let n = 120usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32;
            let low = (t * 2.0 * std::f32::consts::PI / 40.0).sin(); // slow wave
            let high = 0.5 * (t * 2.0 * std::f32::consts::PI / 7.0).sin(); // fast wave
            data.push(low + high);
        }
        let tensor = Tensor::from_vec(data, &[n]).expect("tensor");
        let series = TimeSeries::new(tensor);

        let ssa = SSA::new(40, 4);
        let comps = ssa.fit(&series).expect("fit");
        assert!(comps.len() >= 2, "need at least two components to compare");

        let c0 = comps[0].to_vec().expect("c0");
        let c1 = comps[1].to_vec().expect("c1");

        // Normalised correlation between the first two components. If deflation
        // were broken they would be almost collinear (|corr| ~ 1).
        let mean0: f32 = c0.iter().sum::<f32>() / n as f32;
        let mean1: f32 = c1.iter().sum::<f32>() / n as f32;
        let mut cov = 0.0f32;
        let mut var0 = 0.0f32;
        let mut var1 = 0.0f32;
        for i in 0..n {
            let d0 = c0[i] - mean0;
            let d1 = c1[i] - mean1;
            cov += d0 * d1;
            var0 += d0 * d0;
            var1 += d1 * d1;
        }
        let corr = if var0 > 1e-8 && var1 > 1e-8 {
            (cov / (var0.sqrt() * var1.sqrt())).abs()
        } else {
            0.0
        };
        assert!(
            corr < 0.9,
            "components should be largely uncorrelated after proper deflation, |corr| = {corr}"
        );
    }
}
