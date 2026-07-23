use ndarray::Array1;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 3D 粒子状态数据结构
#[derive(Debug, Clone)]
pub struct ParticleState {
    pub x: Array1<f64>,   // shape (N,) — 位置 x
    pub y: Array1<f64>,   // shape (N,) — 位置 y
    pub z: Array1<f64>,   // shape (N,) — 位置 z
    pub vx: Array1<f64>,  // shape (N,) — 速度 vx
    pub vy: Array1<f64>,  // shape (N,) — 速度 vy
    pub vz: Array1<f64>,  // shape (N,) — 速度 vz
    pub fx: Array1<f64>,  // shape (N,) — 受力 fx
    pub fy: Array1<f64>,  // shape (N,) — 受力 fy
    pub fz: Array1<f64>,  // shape (N,) — 受力 fz
    pub q: Array1<f64>,   // shape (N,) — 电荷量，默认 1.0
    pub m: Array1<f64>,   // shape (N,) — 质量，默认 1.0
}

impl ParticleState {
    /// 创建 n 个粒子：位置在 [0,1) 内随机初始化，速度/力为 0，电荷为 1.0，质量为 1.0
    pub fn zeros(n: usize, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or(0);
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Array1<f64> = (0..n).map(|_| rng.gen()).collect();
        let y: Array1<f64> = (0..n).map(|_| rng.gen()).collect();
        let z: Array1<f64> = (0..n).map(|_| rng.gen()).collect();
        Self {
            x,
            y,
            z,
            vx: Array1::zeros(n),
            vy: Array1::zeros(n),
            vz: Array1::zeros(n),
            fx: Array1::zeros(n),
            fy: Array1::zeros(n),
            fz: Array1::zeros(n),
            q: Array1::ones(n),
            m: Array1::ones(n),
        }
    }

    /// 创建 n 个粒子，并指定每个粒子的电荷量
    pub fn with_charges(n: usize, seed: Option<u64>, charges: &[f64]) -> Self {
        let mut s = Self::zeros(n, seed);
        let len = charges.len().min(n);
        for i in 0..len {
            s.q[i] = charges[i];
        }
        s
    }

    /// 创建 n 个粒子，指定电荷量和质量
    pub fn with_charges_and_masses(n: usize, seed: Option<u64>, charges: &[f64], masses: &[f64]) -> Self {
        let mut s = Self::zeros(n, seed);
        let len = charges.len().min(n);
        for i in 0..len {
            s.q[i] = charges[i];
        }
        let mlen = masses.len().min(n);
        for i in 0..mlen {
            s.m[i] = masses[i];
        }
        s
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// 添加一个粒子（带质量参数）
    pub fn add_particle(&mut self, x: f64, y: f64, z: f64, q: f64, mass: f64, vx: f64, vy: f64, vz: f64) {
        let mut new_x: Vec<f64> = self.x.iter().copied().collect();
        let mut new_y: Vec<f64> = self.y.iter().copied().collect();
        let mut new_z: Vec<f64> = self.z.iter().copied().collect();
        let mut new_vx: Vec<f64> = self.vx.iter().copied().collect();
        let mut new_vy: Vec<f64> = self.vy.iter().copied().collect();
        let mut new_vz: Vec<f64> = self.vz.iter().copied().collect();
        let mut new_fx: Vec<f64> = self.fx.iter().copied().collect();
        let mut new_fy: Vec<f64> = self.fy.iter().copied().collect();
        let mut new_fz: Vec<f64> = self.fz.iter().copied().collect();
        let mut new_q: Vec<f64> = self.q.iter().copied().collect();
        let mut new_m: Vec<f64> = self.m.iter().copied().collect();
        new_x.push(x);
        new_y.push(y);
        new_z.push(z);
        new_vx.push(vx);
        new_vy.push(vy);
        new_vz.push(vz);
        new_fx.push(0.0);
        new_fy.push(0.0);
        new_fz.push(0.0);
        new_q.push(q);
        new_m.push(mass);
        self.x = ndarray::Array1::from_vec(new_x);
        self.y = ndarray::Array1::from_vec(new_y);
        self.z = ndarray::Array1::from_vec(new_z);
        self.vx = ndarray::Array1::from_vec(new_vx);
        self.vy = ndarray::Array1::from_vec(new_vy);
        self.vz = ndarray::Array1::from_vec(new_vz);
        self.fx = ndarray::Array1::from_vec(new_fx);
        self.fy = ndarray::Array1::from_vec(new_fy);
        self.fz = ndarray::Array1::from_vec(new_fz);
        self.q = ndarray::Array1::from_vec(new_q);
        self.m = ndarray::Array1::from_vec(new_m);
    }

    /// 删除指定索引的粒子
    pub fn remove_particle(&mut self, index: usize) {
        if index >= self.len() {
            return;
        }
        let remove_at = |i: usize, arr: &ndarray::Array1<f64>| -> ndarray::Array1<f64> {
            let v: Vec<f64> = arr.iter().enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &v)| v)
                .collect();
            ndarray::Array1::from_vec(v)
        };
        self.x = remove_at(index, &self.x);
        self.y = remove_at(index, &self.y);
        self.z = remove_at(index, &self.z);
        self.vx = remove_at(index, &self.vx);
        self.vy = remove_at(index, &self.vy);
        self.vz = remove_at(index, &self.vz);
        self.fx = remove_at(index, &self.fx);
        self.fy = remove_at(index, &self.fy);
        self.fz = remove_at(index, &self.fz);
        self.q = remove_at(index, &self.q);
        self.m = remove_at(index, &self.m);
    }

    /// 深拷贝
    pub fn copy(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
            vx: self.vx.clone(),
            vy: self.vy.clone(),
            vz: self.vz.clone(),
            fx: self.fx.clone(),
            fy: self.fy.clone(),
            fz: self.fz.clone(),
            q: self.q.clone(),
            m: self.m.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_creates_correct_number_of_particles() {
        let n = 10;
        let particles = ParticleState::zeros(n, Some(42));
        assert_eq!(particles.len(), n);
        assert_eq!(particles.x.len(), n);
        assert_eq!(particles.y.len(), n);
        assert_eq!(particles.z.len(), n);
        assert_eq!(particles.vx.len(), n);
        assert_eq!(particles.vy.len(), n);
        assert_eq!(particles.vz.len(), n);
        assert_eq!(particles.fx.len(), n);
        assert_eq!(particles.fy.len(), n);
        assert_eq!(particles.fz.len(), n);
        assert_eq!(particles.q.len(), n);
        assert_eq!(particles.m.len(), n);
    }

    #[test]
    fn test_zeros_initializes_velocity_and_force_to_zero() {
        let particles = ParticleState::zeros(5, Some(0));
        for i in 0..5 {
            assert_eq!(particles.vx[i], 0.0);
            assert_eq!(particles.vy[i], 0.0);
            assert_eq!(particles.vz[i], 0.0);
            assert_eq!(particles.fx[i], 0.0);
            assert_eq!(particles.fy[i], 0.0);
            assert_eq!(particles.fz[i], 0.0);
        }
    }

    #[test]
    fn test_zeros_initializes_charge_to_one() {
        let particles = ParticleState::zeros(5, Some(0));
        for i in 0..5 {
            assert!((particles.q[i] - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_zeros_initializes_mass_to_one() {
        let particles = ParticleState::zeros(5, Some(0));
        for i in 0..5 {
            assert!((particles.m[i] - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_with_charges_sets_custom_charges() {
        let charges = vec![1.0, -1.0, 2.5, -0.5];
        let particles = ParticleState::with_charges(4, Some(0), &charges);
        assert!((particles.q[0] - 1.0).abs() < 1e-15);
        assert!((particles.q[1] + 1.0).abs() < 1e-15);
        assert!((particles.q[2] - 2.5).abs() < 1e-15);
        assert!((particles.q[3] + 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_with_charges_truncates_to_n() {
        let charges = vec![1.0, -1.0];
        let particles = ParticleState::with_charges(1, Some(0), &charges);
        assert_eq!(particles.len(), 1);
        assert!((particles.q[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_copy_produces_independent_clone() {
        let mut p1 = ParticleState::zeros(3, Some(42));
        p1.x[0] = 99.0;
        p1.z[1] = 77.0;
        p1.q[1] = -5.0;
        p1.m[2] = 2.5;
        let p2 = p1.copy();
        p1.x[0] = 0.0;
        p1.z[1] = 0.0;
        p1.q[1] = 0.0;
        p1.m[2] = 1.0;
        assert!((p2.x[0] - 99.0).abs() < 1e-15);
        assert!((p2.z[1] - 77.0).abs() < 1e-15);
        assert!((p2.q[1] + 5.0).abs() < 1e-15);
        assert!((p2.m[2] - 2.5).abs() < 1e-15);
    }

    #[test]
    fn test_is_empty() {
        let particles = ParticleState::zeros(0, Some(0));
        assert!(particles.is_empty());
        let particles = ParticleState::zeros(1, Some(0));
        assert!(!particles.is_empty());
    }

    #[test]
    fn test_seed_consistency() {
        let p1 = ParticleState::zeros(100, Some(42));
        let p2 = ParticleState::zeros(100, Some(42));
        for i in 0..100 {
            assert!((p1.x[i] - p2.x[i]).abs() < 1e-15);
            assert!((p1.y[i] - p2.y[i]).abs() < 1e-15);
            assert!((p1.z[i] - p2.z[i]).abs() < 1e-15);
        }
    }

    #[test]
    fn test_different_seed_produces_different_positions() {
        let p1 = ParticleState::zeros(100, Some(0));
        let p2 = ParticleState::zeros(100, Some(1));
        let all_same = p1.x.iter().zip(p2.x.iter()).all(|(a, b)| (a - b).abs() < 1e-15);
        assert!(!all_same);
    }

    #[test]
    fn test_add_particle_with_mass() {
        let mut p = ParticleState::zeros(2, Some(0));
        p.add_particle(0.5, 0.5, 0.5, -1.0, 2.5, 0.1, 0.2, 0.3);
        assert_eq!(p.len(), 3);
        assert!((p.q[2] + 1.0).abs() < 1e-15);
        assert!((p.m[2] - 2.5).abs() < 1e-15);
        assert!((p.vx[2] - 0.1).abs() < 1e-15);
        assert!((p.vy[2] - 0.2).abs() < 1e-15);
        assert!((p.vz[2] - 0.3).abs() < 1e-15);
    }

    #[test]
    fn test_remove_particle_with_mass() {
        let mut p = ParticleState::zeros(3, Some(0));
        p.m[0] = 0.5;
        p.m[1] = 1.0;
        p.m[2] = 2.0;
        p.remove_particle(1);
        assert_eq!(p.len(), 2);
        assert!((p.m[0] - 0.5).abs() < 1e-15);
        assert!((p.m[1] - 2.0).abs() < 1e-15);
    }
}