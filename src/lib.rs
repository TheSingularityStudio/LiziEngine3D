pub mod core;
pub mod visual;
pub mod presets;
pub mod gui;

// 重新导出 core 模块，保持外部 API 兼容
pub use core::grid;
pub use core::integrator;
pub use core::interp;
pub use core::particles;
pub use core::poisson_fft;
pub use core::poisson_solver;
pub use core::scatter;
pub use core::sim;
