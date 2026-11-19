#![forbid(unsafe_code)]

//! Pure-Rust symmetric Jacobi eigensolvers (3x3 PCA + 4x4 Horn unit-quaternion Kabsch alignment).
//!
//! Horn's (1987) unit-quaternion formulation finds the exact optimal proper rotation `R in SO(3)`
//! (`det(R) = +1`, never a mirror reflection even for 3 coplanar pharmacophore points) and
//! translation `t` minimizing `sum_i || R * mobile[i] + t - target[i] ||^2`.

/// Result of a rigid 3D superposition mapping `mobile -> target`.
#[derive(Debug, Clone, Copy)]
pub struct RigidAlignment {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub rmsd: f64,
}

impl RigidAlignment {
    #[inline]
    #[must_use]
    pub fn apply(&self, p: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        let t = &self.translation;
        [
            r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
            r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
            r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
        ]
    }

    #[inline]
    #[must_use]
    pub fn apply_dir(&self, v: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        [
            r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
        ]
    }
}

/// Jacobi eigendecomposition of a real symmetric 3x3 matrix.
/// Returns `(eigenvalues, eigenvectors_as_rows)` sorted in descending order of eigenvalue.
#[must_use]
pub fn symmetric_eigen_3x3(mat: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = mat;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..48 {
        // Pick largest off-diagonal entry
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = a[0][1].abs();
        if a[0][2].abs() > max_off {
            max_off = a[0][2].abs();
            p = 0;
            q = 2;
        }
        if a[1][2].abs() > max_off {
            max_off = a[1][2].abs();
            p = 1;
            q = 2;
        }
        if max_off < 1e-13 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..3 {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for row in &mut v {
            let vp = row[p];
            let vq = row[q];
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }

    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let vals = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let vecs = [
        [v[0][order[0]], v[1][order[0]], v[2][order[0]]],
        [v[0][order[1]], v[1][order[1]], v[2][order[1]]],
        [v[0][order[2]], v[1][order[2]], v[2][order[2]]],
    ];
    (vals, vecs)
}

/// Jacobi eigendecomposition of a real symmetric 4x4 matrix; returns the unit eigenvector for
/// the largest eigenvalue.
#[must_use]
fn max_eigenvector_4x4(mut a: [[f64; 4]; 4]) -> [f64; 4] {
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    for _ in 0..80 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = 0.0_f64;
        for (i, row) in a.iter().enumerate() {
            for (j, &cell) in row.iter().enumerate().skip(i + 1) {
                let val = cell.abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-14 {
            break;
        }
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..4 {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for row in &mut v {
            let vp = row[p];
            let vq = row[q];
            row[p] = c * vp - s * vq;
            row[q] = s * vp + c * vq;
        }
    }

    let mut best_col = 0usize;
    let mut best_val = a[0][0];
    for (i, row) in a.iter().enumerate().skip(1) {
        if row[i] > best_val {
            best_val = row[i];
            best_col = i;
        }
    }
    let mut q = [
        v[0][best_col],
        v[1][best_col],
        v[2][best_col],
        v[3][best_col],
    ];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n > 1e-12 {
        for c in &mut q {
            *c /= n;
        }
    } else {
        q = [1.0, 0.0, 0.0, 0.0];
    }
    q
}

/// Optimal proper 3D rigid superposition (`SO(3)`) mapping `mobile[i] -> target[i]`
/// via Horn's (1987) unit-quaternion algorithm.
#[must_use]
pub fn kabsch_align(mobile: &[[f64; 3]], target: &[[f64; 3]]) -> RigidAlignment {
    let n = mobile.len().min(target.len());
    if n == 0 {
        return RigidAlignment {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
            rmsd: 0.0,
        };
    }
    let inv_n = 1.0 / (n as f64);
    let mut cm = [0.0; 3];
    let mut ct = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            cm[k] += mobile[i][k] * inv_n;
            ct[k] += target[i][k] * inv_n;
        }
    }
    if n == 1 {
        return RigidAlignment {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [ct[0] - cm[0], ct[1] - cm[1], ct[2] - cm[2]],
            rmsd: 0.0,
        };
    }

    // Cross-covariance matrix S = sum_i (m_i - cm) (t_i - ct)^T
    let mut s = [[0.0_f64; 3]; 3];
    for i in 0..n {
        let pm = [
            mobile[i][0] - cm[0],
            mobile[i][1] - cm[1],
            mobile[i][2] - cm[2],
        ];
        let pt = [
            target[i][0] - ct[0],
            target[i][1] - ct[1],
            target[i][2] - ct[2],
        ];
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += pm[r] * pt[c];
            }
        }
    }

    let sxx = s[0][0];
    let sxy = s[0][1];
    let sxz = s[0][2];
    let syx = s[1][0];
    let syy = s[1][1];
    let syz = s[1][2];
    let szx = s[2][0];
    let szy = s[2][1];
    let szz = s[2][2];

    // Horn's 4x4 symmetric key matrix N
    let n_mat = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let [qw, qx, qy, qz] = max_eigenvector_4x4(n_mat);

    let rot = [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ];

    let r_cm = [
        rot[0][0] * cm[0] + rot[0][1] * cm[1] + rot[0][2] * cm[2],
        rot[1][0] * cm[0] + rot[1][1] * cm[1] + rot[1][2] * cm[2],
        rot[2][0] * cm[0] + rot[2][1] * cm[1] + rot[2][2] * cm[2],
    ];
    let translation = [ct[0] - r_cm[0], ct[1] - r_cm[1], ct[2] - r_cm[2]];

    let aln = RigidAlignment {
        rotation: rot,
        translation,
        rmsd: 0.0,
    };
    let mut sq_sum = 0.0;
    for i in 0..n {
        let p = aln.apply(mobile[i]);
        let dx = p[0] - target[i][0];
        let dy = p[1] - target[i][1];
        let dz = p[2] - target[i][2];
        sq_sum += dx * dx + dy * dy + dz * dz;
    }
    RigidAlignment {
        rotation: rot,
        translation,
        rmsd: (sq_sum * inv_n).sqrt(),
    }
}
