use ndarray::Array3;

/// 3D 计算网格（周期边界）
#[derive(Debug, Clone)]
pub struct Grid3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl Grid3D {
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> Self {
        Self { nx, ny, nz, dx, dy, dz }
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }

    pub fn lx(&self) -> f64 {
        self.nx as f64 * self.dx
    }

    pub fn ly(&self) -> f64 {
        self.ny as f64 * self.dy
    }

    pub fn lz(&self) -> f64 {
        self.nz as f64 * self.dz
    }

    /// 周期包裹：将连续坐标 (x, y, z) 包裹到 [0, Lx) x [0, Ly) x [0, Lz)
    pub fn periodic_wrap(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let lx = self.lx();
        let ly = self.ly();
        let lz = self.lz();
        let xw = ((x % lx) + lx) % lx;
        let yw = ((y % ly) + ly) % ly;
        let zw = ((z % lz) + lz) % lz;
        (xw, yw, zw)
    }

    /// 世界坐标 → 网格连续索引
    pub fn world_to_grid(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let (xw, yw, zw) = self.periodic_wrap(x, y, z);
        (xw / self.dx, yw / self.dy, zw / self.dz)
    }

    /// 整数网格索引 → 世界坐标
    pub fn grid_to_world(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        (i as f64 * self.dx, j as f64 * self.dy, k as f64 * self.dz)
    }

    /// 三线性插值权重（周期性包裹）
    /// 返回: (i0, i1, j0, j1, k0, k1, wx0, wx1, wy0, wy1, wz0, wz1)
    pub fn trilinear_weights(
        &self, gx: f64, gy: f64, gz: f64,
    ) -> (usize, usize, usize, usize, usize, usize, f64, f64, f64, f64, f64, f64) {
        let i0 = (gx.floor() as i64).rem_euclid(self.nx as i64) as usize;
        let j0 = (gy.floor() as i64).rem_euclid(self.ny as i64) as usize;
        let k0 = (gz.floor() as i64).rem_euclid(self.nz as i64) as usize;
        let i1 = (i0 + 1) % self.nx;
        let j1 = (j0 + 1) % self.ny;
        let k1 = (k0 + 1) % self.nz;

        let fx = gx - gx.floor();
        let fy = gy - gy.floor();
        let fz = gz - gz.floor();

        let wx0 = 1.0 - fx;
        let wx1 = fx;
        let wy0 = 1.0 - fy;
        let wy1 = fy;
        let wz0 = 1.0 - fz;
        let wz1 = fz;

        (i0, i1, j0, j1, k0, k1, wx0, wx1, wy0, wy1, wz0, wz1)
    }

    /// 创建零初始化的网格数组
    pub fn zeros(&self) -> Array3<f64> {
        Array3::zeros((self.nx, self.ny, self.nz))
    }
}