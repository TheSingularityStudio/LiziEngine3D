use ndarray::Array2;

/// 2D 计算网格（周期边界）
#[derive(Debug, Clone)]
pub struct Grid2D {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
}

impl Grid2D {
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Self {
        Self { nx, ny, dx, dy }
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    pub fn lx(&self) -> f64 {
        self.nx as f64 * self.dx
    }

    pub fn ly(&self) -> f64 {
        self.ny as f64 * self.dy
    }

    /// 周期包裹：将连续坐标 (x, y) 包裹到 [0, Lx) x [0, Ly)
    pub fn periodic_wrap(&self, x: f64, y: f64) -> (f64, f64) {
        let lx = self.lx();
        let ly = self.ly();
        let xw = ((x % lx) + lx) % lx;
        let yw = ((y % ly) + ly) % ly;
        (xw, yw)
    }

    /// 世界坐标 → 网格连续索引
    pub fn world_to_grid(&self, x: f64, y: f64) -> (f64, f64) {
        let (xw, yw) = self.periodic_wrap(x, y);
        (xw / self.dx, yw / self.dy)
    }

    /// 整数网格索引 → 世界坐标
    pub fn grid_to_world(&self, i: usize, j: usize) -> (f64, f64) {
        (i as f64 * self.dx, j as f64 * self.dy)
    }

    /// 双线性插值权重（周期性包裹）
    /// 返回: (i0, i1, j0, j1, wx0, wx1, wy0, wy1)
    pub fn bilinear_weights(&self, gx: f64, gy: f64) -> (usize, usize, usize, usize, f64, f64, f64, f64) {
        let i0 = (gx.floor() as i64).rem_euclid(self.nx as i64) as usize;
        let j0 = (gy.floor() as i64).rem_euclid(self.ny as i64) as usize;
        let i1 = (i0 + 1) % self.nx;
        let j1 = (j0 + 1) % self.ny;

        let fx = gx - gx.floor();
        let fy = gy - gy.floor();

        let wx0 = 1.0 - fx;
        let wx1 = fx;
        let wy0 = 1.0 - fy;
        let wy1 = fy;

        (i0, i1, j0, j1, wx0, wx1, wy0, wy1)
    }

    /// 创建零初始化的网格数组
    pub fn zeros(&self) -> Array2<f64> {
        Array2::zeros((self.nx, self.ny))
    }
}