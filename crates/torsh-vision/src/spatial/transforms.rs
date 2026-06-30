//! Geometric transformations for computer vision using scirs2-spatial

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{Result, VisionError};
use scirs2_core::ndarray::{arr2, s, Array1, Array2, ArrayView2};
use scirs2_linalg::svd;
use scirs2_spatial::procrustes::{procrustes, procrustes_extended};
use scirs2_spatial::transform::{RigidTransform, Rotation};
use torsh_tensor::Tensor;

/// Image registration and alignment using spatial transformations
pub struct ImageRegistrar {
    tolerance: f64,
    max_iterations: usize,
}

impl ImageRegistrar {
    /// Create a new image registrar
    pub fn new(tolerance: f64, max_iterations: usize) -> Self {
        Self {
            tolerance,
            max_iterations,
        }
    }

    /// Register two images using feature point correspondences
    pub fn register_images(
        &self,
        source_points: &Array2<f64>,
        target_points: &Array2<f64>,
    ) -> Result<RegistrationResult> {
        if source_points.nrows() != target_points.nrows() {
            return Err(VisionError::InvalidArgument(
                "Source and target point sets must have same number of points".to_string(),
            ));
        }

        if source_points.nrows() < 3 {
            return Err(VisionError::InvalidArgument(
                "At least 3 point correspondences required for registration".to_string(),
            ));
        }

        // Use Procrustes analysis for rigid registration
        let (rotation, translation, scale) = procrustes_extended(
            &source_points.view(),
            &target_points.view(),
            true,
            true,
            true,
        )
        .map_err(|e| VisionError::Other(anyhow::anyhow!("Procrustes analysis failed: {}", e)))?;

        // Convert rotation matrix to Rotation type
        let rotation_transform = Rotation::from_matrix(&rotation.view()).map_err(|e| {
            VisionError::Other(anyhow::anyhow!("Rotation conversion failed: {}", e))
        })?;

        // Compute registration error
        let transformed_points = self.apply_transformation(
            source_points,
            &rotation_transform,
            &translation.translation,
            scale,
        )?;
        let error = self.compute_registration_error(&transformed_points, target_points)?;

        Ok(RegistrationResult {
            rotation: rotation_transform,
            translation: translation.translation,
            scale,
            error,
            converged: error < self.tolerance,
        })
    }

    /// Apply rigid transformation to a set of points
    pub fn apply_transformation(
        &self,
        points: &Array2<f64>,
        rotation: &Rotation,
        translation: &Array1<f64>,
        scale: f64,
    ) -> Result<Array2<f64>> {
        let n_dims = points.ncols();

        // Scale first
        let scaled = points.mapv(|v| v * scale);

        // Apply rotation: use as_matrix() for arbitrary dimensionality
        // apply_multiple requires exactly 3 columns; fall back to matrix dot for 2D
        let rotated = if n_dims == 3 {
            rotation
                .apply_multiple(&scaled.view())
                .map_err(|e| VisionError::Other(anyhow::anyhow!("Rotation apply failed: {}", e)))?
        } else {
            // For 2D (or other dims), extract the upper-left sub-matrix of the rotation matrix
            let rot_mat = rotation.as_matrix(); // 3×3
            let sub_mat = rot_mat
                .slice(scirs2_core::ndarray::s![..n_dims, ..n_dims])
                .to_owned();
            scaled.dot(&sub_mat.t())
        };

        // Apply translation
        let mut result = rotated;
        for mut row in result.outer_iter_mut() {
            for i in 0..n_dims.min(translation.len()) {
                row[i] += translation[i];
            }
        }

        Ok(result)
    }

    /// Compute registration error between two point sets
    fn compute_registration_error(
        &self,
        points1: &Array2<f64>,
        points2: &Array2<f64>,
    ) -> Result<f64> {
        if points1.shape() != points2.shape() {
            return Err(VisionError::InvalidArgument(
                "Point sets must have same shape".to_string(),
            ));
        }

        let mut total_error = 0.0;
        let n_points = points1.nrows();

        for i in 0..n_points {
            let row1 = points1.row(i);
            let row2 = points2.row(i);
            let diff = &row1 - &row2;
            total_error += diff.mapv(|x| x * x).sum();
        }

        Ok((total_error / n_points as f64).sqrt())
    }
}

/// Result of image registration
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    pub rotation: Rotation,
    pub translation: Array1<f64>,
    pub scale: f64,
    pub error: f64,
    pub converged: bool,
}

/// 3D pose estimation for computer vision
pub struct PoseEstimator {
    config: PoseConfig,
}

#[derive(Debug, Clone)]
pub struct PoseConfig {
    pub method: PoseMethod,
    pub ransac_threshold: f64,
    pub max_iterations: usize,
}

#[derive(Debug, Clone)]
pub enum PoseMethod {
    PnP,        // Perspective-n-Point
    Essential,  // Essential matrix estimation
    Homography, // Homography estimation
}

impl Default for PoseConfig {
    fn default() -> Self {
        Self {
            method: PoseMethod::PnP,
            ransac_threshold: 1.0,
            max_iterations: 1000,
        }
    }
}

impl PoseEstimator {
    /// Create a new pose estimator
    pub fn new(config: PoseConfig) -> Self {
        Self { config }
    }

    /// Estimate 3D pose from 2D-3D point correspondences
    pub fn estimate_pose(
        &self,
        points_2d: &Array2<f64>,
        points_3d: &Array2<f64>,
    ) -> Result<PoseEstimate> {
        if points_2d.nrows() != points_3d.nrows() {
            return Err(VisionError::InvalidArgument(
                "2D and 3D point sets must have same number of points".to_string(),
            ));
        }

        match self.config.method {
            PoseMethod::PnP => self.solve_pnp(points_2d, points_3d),
            PoseMethod::Essential => self.estimate_essential_matrix(points_2d, points_3d),
            PoseMethod::Homography => self.estimate_homography(points_2d, points_3d),
        }
    }

    /// Solve the Perspective-n-Point problem via the Direct Linear Transform.
    ///
    /// Given `points_2d` (image projections, one `(u, v)` per row) and the
    /// corresponding `points_3d` (world coordinates, one `(X, Y, Z)` per row),
    /// this recovers the camera rotation `R` and translation `t` such that
    /// `s * [u, v, 1]ᵀ = K * [R | t] * [X, Y, Z, 1]ᵀ`.
    ///
    /// # Assumptions
    ///
    /// The image points are assumed to be in **normalized** camera coordinates
    /// (i.e. the intrinsic matrix `K` is the identity, or the points have
    /// already been pre-multiplied by `K⁻¹`). With a known non-identity `K`,
    /// normalize the 2D points first.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 6 correspondences are supplied (the DLT
    /// needs at least 6 points to constrain the 11 free parameters of the
    /// projection matrix), if the inputs are mis-shaped, or if the SVD fails.
    fn solve_pnp(&self, points_2d: &Array2<f64>, points_3d: &Array2<f64>) -> Result<PoseEstimate> {
        let n = points_2d.nrows();
        if n < 6 {
            return Err(VisionError::InvalidArgument(format!(
                "DLT PnP requires at least 6 point correspondences, got {n}"
            )));
        }
        if points_2d.ncols() < 2 {
            return Err(VisionError::InvalidArgument(
                "2D points must have at least 2 columns (u, v)".to_string(),
            ));
        }
        if points_3d.ncols() < 3 {
            return Err(VisionError::InvalidArgument(
                "3D points must have at least 3 columns (X, Y, Z)".to_string(),
            ));
        }

        // Build the 2N x 12 DLT constraint matrix from the projection equations.
        let mut a = Array2::<f64>::zeros((2 * n, 12));
        for i in 0..n {
            let u = points_2d[[i, 0]];
            let v = points_2d[[i, 1]];
            let x = points_3d[[i, 0]];
            let y = points_3d[[i, 1]];
            let z = points_3d[[i, 2]];

            let r0 = 2 * i;
            // s*u row: [X Y Z 1 0 0 0 0 -uX -uY -uZ -u]
            a[[r0, 0]] = x;
            a[[r0, 1]] = y;
            a[[r0, 2]] = z;
            a[[r0, 3]] = 1.0;
            a[[r0, 8]] = -u * x;
            a[[r0, 9]] = -u * y;
            a[[r0, 10]] = -u * z;
            a[[r0, 11]] = -u;

            let r1 = 2 * i + 1;
            // s*v row: [0 0 0 0 X Y Z 1 -vX -vY -vZ -v]
            a[[r1, 4]] = x;
            a[[r1, 5]] = y;
            a[[r1, 6]] = z;
            a[[r1, 7]] = 1.0;
            a[[r1, 8]] = -v * x;
            a[[r1, 9]] = -v * y;
            a[[r1, 10]] = -v * z;
            a[[r1, 11]] = -v;
        }

        // The projection vector is the right singular vector of the smallest
        // singular value of A (the last row of Vᵀ).
        let h = smallest_singular_vector(&a)?;
        let p = Array2::from_shape_vec((3, 4), h.to_vec()).map_err(|e| {
            VisionError::Other(anyhow::anyhow!("Failed to reshape PnP solution: {e}"))
        })?;

        // Split into the left 3x3 block M (= scale * R) and the last column.
        let m = p.slice(s![.., 0..3]).to_owned();
        let p_col = p.slice(s![.., 3]).to_owned();

        // Recover a proper rotation by projecting M onto SO(3) via SVD, and the
        // scale from the singular values of M (which are all `scale` for a true
        // scaled rotation).
        let (u_mat, sigma, vt_mat) = svd(&m.view(), false, None).map_err(|e| {
            VisionError::Other(anyhow::anyhow!("SVD of PnP projection block failed: {e}"))
        })?;
        let scale_mag = sigma.mean().unwrap_or(0.0);
        if scale_mag < 1e-12 {
            return Err(VisionError::InvalidInput(
                "Degenerate PnP configuration: projection block is rank-deficient".to_string(),
            ));
        }
        let mut rot = u_mat.dot(&vt_mat);
        // Ensure a right-handed (det = +1) rotation rather than a reflection.
        if matrix3_determinant(&rot) < 0.0 {
            rot.mapv_inplace(|x| -x);
        }

        // Translation shares the projection scale; resolve the global sign so
        // that the reconstructed depth of the points is positive (cheirality).
        let mut translation = &p_col / scale_mag;
        let depth_sign = pnp_mean_depth_sign(&rot, &translation, points_3d);
        if depth_sign < 0.0 {
            rot.mapv_inplace(|x| -x);
            translation.mapv_inplace(|x| -x);
            // Negating R can flip the determinant; restore a proper rotation.
            if matrix3_determinant(&rot) < 0.0 {
                // Flip a single column to keep det = +1 while preserving the
                // recovered viewing direction as closely as possible.
                for r in 0..3 {
                    rot[[r, 2]] = -rot[[r, 2]];
                }
            }
        }

        let rotation = Rotation::from_matrix(&rot.view()).map_err(|e| {
            VisionError::Other(anyhow::anyhow!(
                "Recovered PnP rotation is not a valid SO(3) matrix: {e}"
            ))
        })?;

        let error =
            self.compute_reprojection_error(points_2d, points_3d, &rotation, &translation)?;

        Ok(PoseEstimate {
            rotation,
            translation,
            confidence: 1.0 / (1.0 + error),
            method: self.config.method.clone(),
            inlier_count: n,
            essential_matrix: None,
            homography: None,
        })
    }

    /// Estimate the essential matrix with the normalized 8-point algorithm and
    /// decompose it into a relative rotation and (unit) translation direction.
    ///
    /// `points_2d` and `points_3d` are interpreted as the two sets of matched,
    /// **normalized** image points (one `(x, y)` per row) from the first and
    /// second views respectively. (The `points_3d` name is inherited from the
    /// shared [`PoseEstimator::estimate_pose`] entry point; only its first two
    /// columns are used here.)
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 8 correspondences are supplied, if the
    /// inputs are mis-shaped, or if any SVD fails.
    fn estimate_essential_matrix(
        &self,
        points_2d: &Array2<f64>,
        points_3d: &Array2<f64>,
    ) -> Result<PoseEstimate> {
        let n = points_2d.nrows();
        if n < 8 {
            return Err(VisionError::InvalidArgument(format!(
                "The 8-point algorithm requires at least 8 correspondences, got {n}"
            )));
        }
        if points_2d.ncols() < 2 || points_3d.ncols() < 2 {
            return Err(VisionError::InvalidArgument(
                "Essential matrix estimation needs at least 2 columns (x, y) per point set"
                    .to_string(),
            ));
        }

        // Hartley normalization of both point sets improves conditioning.
        let (norm1, t1) = normalize_points_2d(points_2d)?;
        let (norm2, t2) = normalize_points_2d(points_3d)?;

        // Build the N x 9 epipolar constraint matrix from x2ᵀ E x1 = 0.
        let mut a = Array2::<f64>::zeros((n, 9));
        for i in 0..n {
            let x1 = norm1[[i, 0]];
            let y1 = norm1[[i, 1]];
            let x2 = norm2[[i, 0]];
            let y2 = norm2[[i, 1]];
            a[[i, 0]] = x2 * x1;
            a[[i, 1]] = x2 * y1;
            a[[i, 2]] = x2;
            a[[i, 3]] = y2 * x1;
            a[[i, 4]] = y2 * y1;
            a[[i, 5]] = y2;
            a[[i, 6]] = x1;
            a[[i, 7]] = y1;
            a[[i, 8]] = 1.0;
        }

        let e_vec = smallest_singular_vector(&a)?;
        let e_norm = Array2::from_shape_vec((3, 3), e_vec.to_vec()).map_err(|e| {
            VisionError::Other(anyhow::anyhow!("Failed to reshape essential matrix: {e}"))
        })?;

        // Enforce the rank-2 / equal-singular-value constraint of a true
        // essential matrix: E = U diag(s, s, 0) Vᵀ.
        let (u_mat, sigma, vt_mat) = svd(&e_norm.view(), true, None).map_err(|e| {
            VisionError::Other(anyhow::anyhow!(
                "SVD of normalized essential matrix failed: {e}"
            ))
        })?;
        let s_avg = 0.5 * (sigma[0] + sigma[1]);
        let s_constrained = Array2::from_diag(&Array1::from(vec![s_avg, s_avg, 0.0]));
        let e_rank2 = u_mat.dot(&s_constrained).dot(&vt_mat);

        // Denormalize: E = T2ᵀ * E_norm * T1.
        let essential = t2.t().dot(&e_rank2).dot(&t1);

        // Decompose the rank-2 essential matrix into the four (R, t) candidates.
        let (u_raw, _s_e, vt_raw) = svd(&e_rank2.view(), true, None).map_err(|e| {
            VisionError::Other(anyhow::anyhow!(
                "SVD of constrained essential matrix failed: {e}"
            ))
        })?;
        // A true essential matrix is rank-2, so its SVD has a zero singular
        // value. Some SVD backends leave the corresponding singular-vector
        // column/row as zeros instead of completing the orthonormal basis,
        // which would make `U` / `Vᵀ` (and the recovered translation
        // `t = U[:, 2]`) degenerate. Complete both to proper orthonormal 3x3
        // matrices so the rotation/translation extraction is well-defined.
        let u_e = complete_orthonormal_basis_3x3(&u_raw)?;
        let vt_e = complete_orthonormal_basis_3x3(&vt_raw.t().to_owned())?
            .t()
            .to_owned();
        let w = arr2(&[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);

        let mut r_a = u_e.dot(&w).dot(&vt_e);
        if matrix3_determinant(&r_a) < 0.0 {
            r_a.mapv_inplace(|x| -x);
        }
        let mut r_b = u_e.dot(&w.t()).dot(&vt_e);
        if matrix3_determinant(&r_b) < 0.0 {
            r_b.mapv_inplace(|x| -x);
        }
        let t_dir = u_e.slice(s![.., 2]).to_owned();

        // Select the physically valid (R, t) by cheirality: the configuration
        // that reconstructs the most points in front of both cameras.
        let candidates: [(Array2<f64>, Array1<f64>); 4] = [
            (r_a.clone(), t_dir.clone()),
            (r_a, -&t_dir),
            (r_b.clone(), t_dir.clone()),
            (r_b, -&t_dir),
        ];
        let mut best_idx = 0usize;
        let mut best_in_front = -1isize;
        for (idx, (r_c, t_c)) in candidates.iter().enumerate() {
            let in_front = count_points_in_front(r_c, t_c, &norm1, &norm2);
            if in_front > best_in_front {
                best_in_front = in_front;
                best_idx = idx;
            }
        }
        let (rot, translation) = candidates[best_idx].clone();

        let rotation = Rotation::from_matrix(&rot.view()).map_err(|e| {
            VisionError::Other(anyhow::anyhow!(
                "Recovered relative rotation is not a valid SO(3) matrix: {e}"
            ))
        })?;

        // Algebraic residual of the epipolar constraint as a confidence proxy.
        let residual = epipolar_residual(&essential, points_2d, points_3d);

        Ok(PoseEstimate {
            rotation,
            translation,
            confidence: 1.0 / (1.0 + residual),
            method: self.config.method.clone(),
            inlier_count: n,
            essential_matrix: Some(essential),
            homography: None,
        })
    }

    /// Estimate a 3x3 homography with the Direct Linear Transform.
    ///
    /// `points_2d` and `points_3d` are the source and destination planar point
    /// sets (one `(x, y)` per row); only the first two columns of each are used.
    /// The recovered `H` maps source to destination in homogeneous coordinates,
    /// `[x', y', w]ᵀ ~ H * [x, y, 1]ᵀ`, and is normalized so that `H[2][2] = 1`.
    ///
    /// A homography is a planar projective transform rather than a rigid pose,
    /// so the returned [`PoseEstimate::rotation`] / `translation` are left as the
    /// identity / zero; the meaningful result is in [`PoseEstimate::homography`].
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 4 correspondences are supplied, if the
    /// inputs are mis-shaped, if the SVD fails, or if the recovered homography is
    /// degenerate (`H[2][2] ≈ 0`).
    fn estimate_homography(
        &self,
        points_2d: &Array2<f64>,
        points_3d: &Array2<f64>,
    ) -> Result<PoseEstimate> {
        let homography = estimate_homography_dlt(points_2d, points_3d)?;

        // Real geometric residual: mean reprojection error of the source points
        // mapped through H against the destination points.
        let residual = homography_reprojection_error(&homography, points_2d, points_3d)?;

        Ok(PoseEstimate {
            rotation: Rotation::identity(),
            translation: Array1::zeros(3),
            confidence: 1.0 / (1.0 + residual),
            method: self.config.method.clone(),
            inlier_count: points_2d.nrows(),
            essential_matrix: None,
            homography: Some(homography),
        })
    }

    /// Mean reprojection error of the world points after projecting them with
    /// the recovered camera pose (`K = I`).
    fn compute_reprojection_error(
        &self,
        points_2d: &Array2<f64>,
        points_3d: &Array2<f64>,
        rotation: &Rotation,
        translation: &Array1<f64>,
    ) -> Result<f64> {
        let n = points_2d.nrows();
        if n == 0 {
            return Ok(0.0);
        }
        let r = rotation.as_matrix();
        let mut total = 0.0;
        let mut counted = 0usize;
        for i in 0..n {
            let x = points_3d[[i, 0]];
            let y = points_3d[[i, 1]];
            let z = points_3d[[i, 2]];
            // Camera-frame point: X_c = R * X_w + t.
            let xc = r[[0, 0]] * x + r[[0, 1]] * y + r[[0, 2]] * z + translation[0];
            let yc = r[[1, 0]] * x + r[[1, 1]] * y + r[[1, 2]] * z + translation[1];
            let zc = r[[2, 0]] * x + r[[2, 1]] * y + r[[2, 2]] * z + translation[2];
            if zc.abs() < 1e-12 {
                continue;
            }
            let proj_u = xc / zc;
            let proj_v = yc / zc;
            let du = proj_u - points_2d[[i, 0]];
            let dv = proj_v - points_2d[[i, 1]];
            total += (du * du + dv * dv).sqrt();
            counted += 1;
        }
        if counted == 0 {
            return Err(VisionError::InvalidInput(
                "All reconstructed points lie on the camera plane (zero depth)".to_string(),
            ));
        }
        Ok(total / counted as f64)
    }
}

/// Compute the right singular vector associated with the smallest singular
/// value of `a` (i.e. the last row of `Vᵀ`), used to solve homogeneous systems
/// `A x = 0` in a total-least-squares sense.
fn smallest_singular_vector(a: &Array2<f64>) -> Result<Array1<f64>> {
    let (_u, _s, vt) = svd(&a.view(), true, None)
        .map_err(|e| VisionError::Other(anyhow::anyhow!("SVD failed: {e}")))?;
    // Singular values are returned in descending order, so the smallest
    // corresponds to the final row of Vᵀ.
    let last = vt.nrows() - 1;
    Ok(vt.row(last).to_owned())
}

/// Determinant of a 3x3 matrix.
fn matrix3_determinant(m: &Array2<f64>) -> f64 {
    m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
        - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
        + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
}

/// Repair a 3x3 matrix whose columns should be an orthonormal basis but whose
/// last column may be zero or non-orthonormal (as produced by some SVD backends
/// for rank-deficient inputs).
///
/// The first two columns are re-orthonormalized with Gram-Schmidt, and the
/// third is set to their cross product, yielding a right-handed orthonormal
/// frame (`det = +1`). Returns an error if the first two columns are
/// themselves degenerate (collinear or zero), which would make the basis
/// unrecoverable.
fn complete_orthonormal_basis_3x3(m: &Array2<f64>) -> Result<Array2<f64>> {
    if m.shape() != [3, 3] {
        return Err(VisionError::InvalidArgument(format!(
            "complete_orthonormal_basis_3x3 expects a 3x3 matrix, got {:?}",
            m.shape()
        )));
    }
    let col = |j: usize| [m[[0, j]], m[[1, j]], m[[2, j]]];
    let norm = |v: &[f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: &[f64; 3], b: &[f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };

    // Normalize the first column.
    let c0 = col(0);
    let n0 = norm(&c0);
    if n0 < 1e-12 {
        return Err(VisionError::InvalidInput(
            "Cannot complete orthonormal basis: first column is degenerate".to_string(),
        ));
    }
    let e0 = [c0[0] / n0, c0[1] / n0, c0[2] / n0];

    // Gram-Schmidt the second column against the first.
    let c1 = col(1);
    let proj = dot(&c1, &e0);
    let mut e1 = [
        c1[0] - proj * e0[0],
        c1[1] - proj * e0[1],
        c1[2] - proj * e0[2],
    ];
    let n1 = norm(&e1);
    if n1 < 1e-12 {
        return Err(VisionError::InvalidInput(
            "Cannot complete orthonormal basis: first two columns are collinear".to_string(),
        ));
    }
    e1 = [e1[0] / n1, e1[1] / n1, e1[2] / n1];

    // Third column is the cross product, giving a right-handed orthonormal frame.
    let e2 = cross(&e0, &e1);

    let mut out = Array2::<f64>::zeros((3, 3));
    for r in 0..3 {
        out[[r, 0]] = e0[r];
        out[[r, 1]] = e1[r];
        out[[r, 2]] = e2[r];
    }
    Ok(out)
}

/// Mean sign of the reconstructed depth `z_c` for the supplied world points
/// under camera pose `(R, t)`. A positive value means most points are in front
/// of the camera.
fn pnp_mean_depth_sign(rot: &Array2<f64>, t: &Array1<f64>, points_3d: &Array2<f64>) -> f64 {
    let mut acc = 0.0;
    let n = points_3d.nrows();
    if n == 0 {
        return 1.0;
    }
    for i in 0..n {
        let x = points_3d[[i, 0]];
        let y = points_3d[[i, 1]];
        let z = points_3d[[i, 2]];
        let zc = rot[[2, 0]] * x + rot[[2, 1]] * y + rot[[2, 2]] * z + t[2];
        acc += zc;
    }
    if acc >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// Hartley normalization: translate points to have zero centroid and scale so
/// that the mean distance to the origin is sqrt(2). Returns the normalized
/// points (with their first two columns) and the 3x3 similarity transform `T`
/// such that `x_norm = T * x`.
fn normalize_points_2d(points: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    let n = points.nrows();
    if n == 0 {
        return Err(VisionError::InvalidArgument(
            "Cannot normalize an empty point set".to_string(),
        ));
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        cx += points[[i, 0]];
        cy += points[[i, 1]];
    }
    cx /= n as f64;
    cy /= n as f64;

    let mut mean_dist = 0.0;
    for i in 0..n {
        let dx = points[[i, 0]] - cx;
        let dy = points[[i, 1]] - cy;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n as f64;

    let scale = if mean_dist > 1e-12 {
        (2.0_f64).sqrt() / mean_dist
    } else {
        1.0
    };

    let t = arr2(&[
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0],
    ]);

    let mut normalized = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        normalized[[i, 0]] = scale * (points[[i, 0]] - cx);
        normalized[[i, 1]] = scale * (points[[i, 1]] - cy);
    }
    Ok((normalized, t))
}

/// Count how many correspondences triangulate to a positive depth in both the
/// reference camera (identity pose) and the second camera `(R, t)` — the
/// cheirality test used to disambiguate the four essential-matrix solutions.
fn count_points_in_front(
    r: &Array2<f64>,
    t: &Array1<f64>,
    pts1: &Array2<f64>,
    pts2: &Array2<f64>,
) -> isize {
    let n = pts1.nrows();
    let mut count = 0isize;
    for i in 0..n {
        let x1 = arr_point_homog(pts1, i);
        let x2 = arr_point_homog(pts2, i);
        if let Some(point3d) = triangulate_midpoint(r, t, &x1, &x2) {
            // Depth in camera 1 (identity pose).
            let z1 = point3d[2];
            // Depth in camera 2.
            let z2 =
                r[[2, 0]] * point3d[0] + r[[2, 1]] * point3d[1] + r[[2, 2]] * point3d[2] + t[2];
            if z1 > 0.0 && z2 > 0.0 {
                count += 1;
            }
        }
    }
    count
}

/// Homogeneous normalized image point `(x, y, 1)` for row `i`.
fn arr_point_homog(points: &Array2<f64>, i: usize) -> [f64; 3] {
    [points[[i, 0]], points[[i, 1]], 1.0]
}

/// Linear triangulation (DLT) of a single correspondence between camera 1 with
/// pose `[I | 0]` and camera 2 with pose `[R | t]`. Returns the 3D point, or
/// `None` if the configuration is degenerate.
fn triangulate_midpoint(
    r: &Array2<f64>,
    t: &Array1<f64>,
    x1: &[f64; 3],
    x2: &[f64; 3],
) -> Option<[f64; 3]> {
    // Projection matrices: P1 = [I | 0], P2 = [R | t].
    let mut p1 = Array2::<f64>::zeros((3, 4));
    p1[[0, 0]] = 1.0;
    p1[[1, 1]] = 1.0;
    p1[[2, 2]] = 1.0;
    let mut p2 = Array2::<f64>::zeros((3, 4));
    for row in 0..3 {
        for col in 0..3 {
            p2[[row, col]] = r[[row, col]];
        }
        p2[[row, 3]] = t[row];
    }

    // Build the 4x4 homogeneous system from the cross-product constraints
    // x × (P X) = 0 (two independent rows per view).
    let mut a = Array2::<f64>::zeros((4, 4));
    for col in 0..4 {
        a[[0, col]] = x1[0] * p1[[2, col]] - p1[[0, col]];
        a[[1, col]] = x1[1] * p1[[2, col]] - p1[[1, col]];
        a[[2, col]] = x2[0] * p2[[2, col]] - p2[[0, col]];
        a[[3, col]] = x2[1] * p2[[2, col]] - p2[[1, col]];
    }

    let (_u, _s, vt) = svd(&a.view(), true, None).ok()?;
    let last = vt.nrows() - 1;
    let h = vt.row(last);
    let w = h[3];
    if w.abs() < 1e-12 {
        return None;
    }
    Some([h[0] / w, h[1] / w, h[2] / w])
}

/// Mean first-order (Sampson) epipolar distance over all correspondences,
/// used as a scale-invariant confidence proxy for the essential matrix.
///
/// The Sampson distance approximates the geometric reprojection error and is
/// invariant to the arbitrary scale of `E`:
/// `d = (x2ᵀ E x1)² / (‖(E x1)_{xy}‖² + ‖(Eᵀ x2)_{xy}‖²)`.
fn epipolar_residual(e: &Array2<f64>, pts1: &Array2<f64>, pts2: &Array2<f64>) -> f64 {
    let n = pts1.nrows();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..n {
        let x1 = arr_point_homog(pts1, i);
        let x2 = arr_point_homog(pts2, i);
        // E * x1
        let ex1 = [
            e[[0, 0]] * x1[0] + e[[0, 1]] * x1[1] + e[[0, 2]] * x1[2],
            e[[1, 0]] * x1[0] + e[[1, 1]] * x1[1] + e[[1, 2]] * x1[2],
            e[[2, 0]] * x1[0] + e[[2, 1]] * x1[1] + e[[2, 2]] * x1[2],
        ];
        // Eᵀ * x2
        let etx2 = [
            e[[0, 0]] * x2[0] + e[[1, 0]] * x2[1] + e[[2, 0]] * x2[2],
            e[[0, 1]] * x2[0] + e[[1, 1]] * x2[1] + e[[2, 1]] * x2[2],
            e[[0, 2]] * x2[0] + e[[1, 2]] * x2[1] + e[[2, 2]] * x2[2],
        ];
        let num = x2[0] * ex1[0] + x2[1] * ex1[1] + x2[2] * ex1[2];
        let denom = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
        if denom > 1e-18 {
            total += (num * num) / denom;
        }
    }
    (total / n as f64).sqrt()
}

/// Direct Linear Transform homography estimation from >= 4 planar
/// correspondences. Returns the 3x3 homography normalized so that `H[2][2] = 1`.
fn estimate_homography_dlt(src: &Array2<f64>, dst: &Array2<f64>) -> Result<Array2<f64>> {
    let n = src.nrows();
    if n < 4 {
        return Err(VisionError::InvalidArgument(format!(
            "Homography estimation (DLT) requires at least 4 correspondences, got {n}"
        )));
    }
    if dst.nrows() != n {
        return Err(VisionError::InvalidArgument(
            "Source and destination point sets must have the same number of points".to_string(),
        ));
    }
    if src.ncols() < 2 || dst.ncols() < 2 {
        return Err(VisionError::InvalidArgument(
            "Homography points must have at least 2 columns (x, y)".to_string(),
        ));
    }

    // Build the 2N x 9 DLT matrix; each correspondence contributes two rows.
    let mut a = Array2::<f64>::zeros((2 * n, 9));
    for i in 0..n {
        let x = src[[i, 0]];
        let y = src[[i, 1]];
        let xp = dst[[i, 0]];
        let yp = dst[[i, 1]];

        let r0 = 2 * i;
        a[[r0, 0]] = -x;
        a[[r0, 1]] = -y;
        a[[r0, 2]] = -1.0;
        a[[r0, 6]] = xp * x;
        a[[r0, 7]] = xp * y;
        a[[r0, 8]] = xp;

        let r1 = 2 * i + 1;
        a[[r1, 3]] = -x;
        a[[r1, 4]] = -y;
        a[[r1, 5]] = -1.0;
        a[[r1, 6]] = yp * x;
        a[[r1, 7]] = yp * y;
        a[[r1, 8]] = yp;
    }

    let h = smallest_singular_vector(&a)?;
    let mut homography = Array2::from_shape_vec((3, 3), h.to_vec())
        .map_err(|e| VisionError::Other(anyhow::anyhow!("Failed to reshape homography: {e}")))?;

    // Normalize so that H[2][2] = 1.
    let scale = homography[[2, 2]];
    if scale.abs() < 1e-12 {
        return Err(VisionError::InvalidInput(
            "Degenerate homography: H[2][2] is approximately zero".to_string(),
        ));
    }
    homography.mapv_inplace(|v| v / scale);
    Ok(homography)
}

/// Mean Euclidean reprojection error of the source points mapped through `H`
/// against the destination points.
fn homography_reprojection_error(
    h: &Array2<f64>,
    src: &Array2<f64>,
    dst: &Array2<f64>,
) -> Result<f64> {
    let n = src.nrows();
    if n == 0 {
        return Ok(0.0);
    }
    let mut total = 0.0;
    let mut counted = 0usize;
    for i in 0..n {
        let x = src[[i, 0]];
        let y = src[[i, 1]];
        let wx = h[[0, 0]] * x + h[[0, 1]] * y + h[[0, 2]];
        let wy = h[[1, 0]] * x + h[[1, 1]] * y + h[[1, 2]];
        let ww = h[[2, 0]] * x + h[[2, 1]] * y + h[[2, 2]];
        if ww.abs() < 1e-12 {
            continue;
        }
        let px = wx / ww;
        let py = wy / ww;
        let dx = px - dst[[i, 0]];
        let dy = py - dst[[i, 1]];
        total += (dx * dx + dy * dy).sqrt();
        counted += 1;
    }
    if counted == 0 {
        return Err(VisionError::InvalidInput(
            "All homography-mapped points are at infinity (zero homogeneous weight)".to_string(),
        ));
    }
    Ok(total / counted as f64)
}

/// Result of pose estimation
#[derive(Debug, Clone)]
pub struct PoseEstimate {
    /// Estimated camera rotation.
    ///
    /// - For [`PoseMethod::PnP`] this is the rotation of the camera relative to
    ///   the world frame, recovered from the DLT projection matrix.
    /// - For [`PoseMethod::Essential`] this is the relative rotation between the
    ///   two views, recovered by decomposing the essential matrix.
    /// - For [`PoseMethod::Homography`] a homography is a planar projective
    ///   transform, not a rigid pose, so the rotation is left as the identity
    ///   and the meaningful result is exposed through [`PoseEstimate::homography`].
    pub rotation: Rotation,
    /// Estimated camera translation (up to scale for the essential-matrix case).
    pub translation: Array1<f64>,
    /// Confidence in `[0, 1]` derived from the real geometric/algebraic residual.
    pub confidence: f64,
    /// Estimation method that produced this result.
    pub method: PoseMethod,
    /// Number of correspondences used (this implementation uses all of them; it
    /// does not yet perform RANSAC inlier selection).
    pub inlier_count: usize,
    /// The estimated 3x3 essential matrix, when [`PoseMethod::Essential`] is used.
    pub essential_matrix: Option<Array2<f64>>,
    /// The estimated 3x3 homography matrix, when [`PoseMethod::Homography`] is used.
    pub homography: Option<Array2<f64>>,
}

/// Geometric transformation utilities
pub struct GeometricProcessor {
    default_interpolation: InterpolationMethod,
}

#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    Nearest,
    Bilinear,
    Bicubic,
}

impl GeometricProcessor {
    /// Create a new geometric processor
    pub fn new(interpolation: InterpolationMethod) -> Self {
        Self {
            default_interpolation: interpolation,
        }
    }

    /// Apply affine transformation to an image
    pub fn apply_affine_transform(
        &self,
        image: &Tensor,
        _transform_matrix: &Array2<f64>,
    ) -> Result<Tensor> {
        // Placeholder for affine transformation
        // Real implementation would apply the transformation matrix to image coordinates
        Ok(image.clone())
    }

    /// Rectify image using homography
    pub fn rectify_image(&self, image: &Tensor, _homography: &Array2<f64>) -> Result<Tensor> {
        // Placeholder for image rectification
        Ok(image.clone())
    }

    /// Correct perspective distortion
    pub fn correct_perspective(
        &self,
        image: &Tensor,
        corner_points: &Array2<f64>,
        target_points: &Array2<f64>,
    ) -> Result<Tensor> {
        if corner_points.nrows() != 4 || target_points.nrows() != 4 {
            return Err(VisionError::InvalidArgument(
                "Perspective correction requires exactly 4 corner points".to_string(),
            ));
        }

        // Compute homography matrix
        let homography = self.compute_homography(corner_points, target_points)?;

        // Apply rectification
        self.rectify_image(image, &homography)
    }

    fn compute_homography(
        &self,
        _source: &Array2<f64>,
        _target: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        // Placeholder for homography computation
        Ok(Array2::eye(3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // arr2 imported above

    #[test]
    fn test_image_registrar_creation() {
        let registrar = ImageRegistrar::new(1e-6, 100);
        assert_eq!(registrar.tolerance, 1e-6);
        assert_eq!(registrar.max_iterations, 100);
    }

    #[test]
    fn test_pose_estimator_creation() {
        let config = PoseConfig::default();
        let estimator = PoseEstimator::new(config);
        assert!(matches!(estimator.config.method, PoseMethod::PnP));
    }

    #[test]
    fn test_geometric_processor_creation() {
        let processor = GeometricProcessor::new(InterpolationMethod::Bilinear);
        assert!(matches!(
            processor.default_interpolation,
            InterpolationMethod::Bilinear
        ));
    }

    #[test]
    fn test_registration_with_invalid_points() {
        let registrar = ImageRegistrar::new(1e-6, 100);
        let source = arr2(&[[1.0, 2.0]]);
        let target = arr2(&[[2.0, 3.0], [4.0, 5.0]]);

        let result = registrar.register_images(&source, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_transformation_identity_rotation() {
        use scirs2_spatial::transform::Rotation;

        let registrar = ImageRegistrar::new(1e-6, 100);
        let rotation = Rotation::identity();
        let translation = scirs2_core::ndarray::Array1::zeros(3);
        let scale = 1.0_f64;

        let points = arr2(&[[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        let result = registrar
            .apply_transformation(&points, &rotation, &translation, scale)
            .expect("apply_transformation should succeed with identity");

        // Identity rotation + zero translation + scale 1.0 → same points
        for i in 0..3 {
            for j in 0..3 {
                let diff = (result[[i, j]] - points[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Expected identity transform, got diff {diff} at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_apply_transformation_scales_points() {
        use scirs2_spatial::transform::Rotation;

        let registrar = ImageRegistrar::new(1e-6, 100);
        let rotation = Rotation::identity();
        let translation = scirs2_core::ndarray::Array1::zeros(3);
        let scale = 2.0_f64;

        let points = arr2(&[[1.0_f64, 2.0, 3.0]]);
        let result = registrar
            .apply_transformation(&points, &rotation, &translation, scale)
            .expect("apply_transformation should succeed");

        assert!((result[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 4.0).abs() < 1e-10);
        assert!((result[[0, 2]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply_transformation_translation() {
        use scirs2_core::ndarray::Array1;
        use scirs2_spatial::transform::Rotation;

        let registrar = ImageRegistrar::new(1e-6, 100);
        let rotation = Rotation::identity();
        let translation = Array1::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let scale = 1.0_f64;

        let points = arr2(&[[0.0_f64, 0.0, 0.0]]);
        let result = registrar
            .apply_transformation(&points, &rotation, &translation, scale)
            .expect("apply_transformation with translation should succeed");

        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((result[[0, 2]] - 3.0).abs() < 1e-10);
    }

    // --- Homography (DLT) -------------------------------------------------

    /// Apply a known homography to a point to generate a synthetic destination.
    fn apply_homography(h: &Array2<f64>, x: f64, y: f64) -> (f64, f64) {
        let wx = h[[0, 0]] * x + h[[0, 1]] * y + h[[0, 2]];
        let wy = h[[1, 0]] * x + h[[1, 1]] * y + h[[1, 2]];
        let ww = h[[2, 0]] * x + h[[2, 1]] * y + h[[2, 2]];
        (wx / ww, wy / ww)
    }

    #[test]
    fn test_estimate_homography_recovers_known_transform() {
        // A non-trivial homography: anisotropic scale + translation + a small
        // projective term so it is not a pure affine map.
        let h_true = arr2(&[[1.5_f64, 0.0, 3.0], [0.0, 2.0, -1.0], [0.001, 0.002, 1.0]]);

        let src = arr2(&[
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [2.0, 0.5],
        ]);
        let mut dst = Array2::<f64>::zeros((src.nrows(), 2));
        for i in 0..src.nrows() {
            let (xp, yp) = apply_homography(&h_true, src[[i, 0]], src[[i, 1]]);
            dst[[i, 0]] = xp;
            dst[[i, 1]] = yp;
        }

        let estimator = PoseEstimator::new(PoseConfig {
            method: PoseMethod::Homography,
            ..Default::default()
        });
        let result = estimator
            .estimate_pose(&src, &dst)
            .expect("homography estimation should succeed");

        let h_est = result
            .homography
            .expect("homography result must carry the estimated matrix");

        // Compare element-wise (both are normalized so H[2][2] = 1).
        for r in 0..3 {
            for c in 0..3 {
                let diff = (h_est[[r, c]] - h_true[[r, c]]).abs();
                assert!(
                    diff < 1e-6,
                    "Homography mismatch at [{r},{c}]: got {}, want {} (diff {diff})",
                    h_est[[r, c]],
                    h_true[[r, c]]
                );
            }
        }
        // Real residual must be (near) zero for an exact synthetic transform.
        assert!(result.confidence > 0.999);
    }

    #[test]
    fn test_estimate_homography_pure_translation() {
        // Translation-only planar transform of 4 points.
        let src = arr2(&[[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let dst = arr2(&[[2.0_f64, 3.0], [3.0, 3.0], [3.0, 4.0], [2.0, 4.0]]);

        let h = estimate_homography_dlt(&src, &dst).expect("DLT should succeed");
        // Recovered H should map src -> dst exactly.
        for i in 0..src.nrows() {
            let (xp, yp) = apply_homography(&h, src[[i, 0]], src[[i, 1]]);
            assert!((xp - dst[[i, 0]]).abs() < 1e-9);
            assert!((yp - dst[[i, 1]]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_estimate_homography_too_few_points_errors() {
        // 3 points < required 4 -> honest error, never identity.
        let src = arr2(&[[0.0_f64, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let dst = arr2(&[[1.0_f64, 1.0], [2.0, 1.0], [1.0, 2.0]]);
        let result = estimate_homography_dlt(&src, &dst);
        assert!(result.is_err());

        let estimator = PoseEstimator::new(PoseConfig {
            method: PoseMethod::Homography,
            ..Default::default()
        });
        assert!(estimator.estimate_pose(&src, &dst).is_err());
    }

    // --- Essential matrix (8-point) --------------------------------------

    #[test]
    fn test_estimate_essential_too_few_points_errors() {
        let pts1 = arr2(&[
            [0.1_f64, 0.1],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.1, 0.3],
            [0.2, 0.3],
            [0.4, 0.1],
            [0.1, 0.4],
        ]); // 7 points < 8
        let pts2 = pts1.clone();

        let estimator = PoseEstimator::new(PoseConfig {
            method: PoseMethod::Essential,
            ..Default::default()
        });
        assert!(estimator.estimate_pose(&pts1, &pts2).is_err());
    }

    #[test]
    fn test_estimate_essential_pure_translation_recovers_rotation() {
        // Generate a synthetic two-view setup: camera 2 is the identity rotation
        // translated along +x. Project a cloud of 3D points into both normalized
        // cameras and run the 8-point algorithm.
        let r_true = Rotation::identity();
        let r_mat = r_true.as_matrix();
        let t_true = [0.5_f64, 0.0, 0.0];

        // A spread of 3D points in front of both cameras.
        let world = [
            [-0.3_f64, -0.2, 3.0],
            [0.4, -0.1, 4.0],
            [0.1, 0.3, 3.5],
            [-0.2, 0.25, 5.0],
            [0.35, 0.15, 4.5],
            [-0.4, -0.3, 3.2],
            [0.2, -0.35, 6.0],
            [0.0, 0.0, 3.8],
            [0.45, 0.4, 5.5],
            [-0.45, 0.1, 4.2],
        ];

        let mut pts1 = Array2::<f64>::zeros((world.len(), 2));
        let mut pts2 = Array2::<f64>::zeros((world.len(), 2));
        for (i, p) in world.iter().enumerate() {
            // Camera 1: [I | 0].
            pts1[[i, 0]] = p[0] / p[2];
            pts1[[i, 1]] = p[1] / p[2];
            // Camera 2: [R | t].
            let xc = r_mat[[0, 0]] * p[0] + r_mat[[0, 1]] * p[1] + r_mat[[0, 2]] * p[2] + t_true[0];
            let yc = r_mat[[1, 0]] * p[0] + r_mat[[1, 1]] * p[1] + r_mat[[1, 2]] * p[2] + t_true[1];
            let zc = r_mat[[2, 0]] * p[0] + r_mat[[2, 1]] * p[1] + r_mat[[2, 2]] * p[2] + t_true[2];
            pts2[[i, 0]] = xc / zc;
            pts2[[i, 1]] = yc / zc;
        }

        let estimator = PoseEstimator::new(PoseConfig {
            method: PoseMethod::Essential,
            ..Default::default()
        });
        let result = estimator
            .estimate_pose(&pts1, &pts2)
            .expect("essential matrix estimation should succeed");

        assert!(
            result.essential_matrix.is_some(),
            "essential matrix must be returned"
        );

        // Recovered rotation should be (close to) the identity. Compare the
        // rotation angle of R_est * R_trueᵀ, which should be near 0.
        let r_est = result.rotation.as_matrix();
        let trace = r_est[[0, 0]] + r_est[[1, 1]] + r_est[[2, 2]];
        let angle = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
        assert!(
            angle < 1e-3,
            "Recovered rotation angle {angle} rad (expected ~0); should be < 1e-3"
        );

        // Translation direction should align with +x (up to sign/scale).
        let t_est = &result.translation;
        let norm = (t_est[0] * t_est[0] + t_est[1] * t_est[1] + t_est[2] * t_est[2]).sqrt();
        assert!(norm > 1e-9, "translation direction must be non-degenerate");
        let cos_align = (t_est[0] / norm).abs();
        assert!(
            cos_align > 0.99,
            "Translation should align with x-axis, |cos| = {cos_align}"
        );

        // Algebraic epipolar residual should be tiny for exact data.
        assert!(result.confidence > 0.99);
    }

    // --- PnP (DLT) --------------------------------------------------------

    #[test]
    fn test_solve_pnp_too_few_points_errors() {
        // 5 points < required 6.
        let pts_3d = arr2(&[
            [0.0_f64, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 5.0],
            [1.0, 1.0, 5.0],
            [0.5, 0.5, 6.0],
        ]);
        let pts_2d = arr2(&[
            [0.0_f64, 0.0],
            [0.2, 0.0],
            [0.0, 0.2],
            [0.2, 0.2],
            [0.08, 0.08],
        ]);
        let estimator = PoseEstimator::new(PoseConfig {
            method: PoseMethod::PnP,
            ..Default::default()
        });
        assert!(estimator.estimate_pose(&pts_2d, &pts_3d).is_err());
    }

    #[test]
    fn test_solve_pnp_recovers_known_pose() {
        // Known camera pose: rotation about y by ~15 degrees, translation in z.
        let angles = scirs2_core::ndarray::array![0.0_f64, 0.26, 0.0];
        let r_true =
            Rotation::from_euler(&angles.view(), "xyz").expect("euler rotation should build");
        let r_mat = r_true.as_matrix();
        let t_true = [0.2_f64, -0.1, 6.0];

        // A non-coplanar 3D point cloud (DLT PnP needs general position).
        let world = [
            [-1.0_f64, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.5],
            [0.5, -0.5, 2.0],
            [-0.5, 0.5, 1.0],
            [0.8, 0.2, 0.7],
        ];

        let mut pts_3d = Array2::<f64>::zeros((world.len(), 3));
        let mut pts_2d = Array2::<f64>::zeros((world.len(), 2));
        for (i, p) in world.iter().enumerate() {
            pts_3d[[i, 0]] = p[0];
            pts_3d[[i, 1]] = p[1];
            pts_3d[[i, 2]] = p[2];
            let xc = r_mat[[0, 0]] * p[0] + r_mat[[0, 1]] * p[1] + r_mat[[0, 2]] * p[2] + t_true[0];
            let yc = r_mat[[1, 0]] * p[0] + r_mat[[1, 1]] * p[1] + r_mat[[1, 2]] * p[2] + t_true[1];
            let zc = r_mat[[2, 0]] * p[0] + r_mat[[2, 1]] * p[1] + r_mat[[2, 2]] * p[2] + t_true[2];
            pts_2d[[i, 0]] = xc / zc;
            pts_2d[[i, 1]] = yc / zc;
        }

        let estimator = PoseEstimator::new(PoseConfig {
            method: PoseMethod::PnP,
            ..Default::default()
        });
        let result = estimator
            .estimate_pose(&pts_2d, &pts_3d)
            .expect("PnP should succeed");

        // Reprojection error must be small for exact synthetic data.
        assert!(
            result.confidence > 0.99,
            "PnP confidence {} too low (large reprojection error)",
            result.confidence
        );

        // Recovered rotation should match the ground truth.
        let r_est = result.rotation.as_matrix();
        let mut frob = 0.0;
        for r in 0..3 {
            for c in 0..3 {
                let d = r_est[[r, c]] - r_mat[[r, c]];
                frob += d * d;
            }
        }
        assert!(
            frob.sqrt() < 1e-3,
            "Recovered rotation differs from truth (Frobenius {})",
            frob.sqrt()
        );

        // Translation should match (DLT recovers it up to the resolved scale).
        for k in 0..3 {
            let d = (result.translation[k] - t_true[k]).abs();
            assert!(
                d < 1e-2,
                "Translation component {k} mismatch: got {}, want {}",
                result.translation[k],
                t_true[k]
            );
        }
    }

    #[test]
    fn test_estimators_never_return_silent_identity() {
        // Regression guard: a genuinely non-identity homography must NOT come
        // back as identity rotation + the input pretending to be a transform.
        let src = arr2(&[[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let dst = arr2(&[[5.0_f64, 5.0], [7.0, 5.0], [7.0, 8.0], [5.0, 8.0]]);
        let h = estimate_homography_dlt(&src, &dst).expect("DLT should succeed");
        // H must be a real, non-identity transform.
        let is_identity = (h[[0, 0]] - 1.0).abs() < 1e-6
            && h[[0, 1]].abs() < 1e-6
            && h[[0, 2]].abs() < 1e-6
            && h[[1, 0]].abs() < 1e-6
            && (h[[1, 1]] - 1.0).abs() < 1e-6;
        assert!(
            !is_identity,
            "Estimated homography must not be the identity"
        );
    }
}
