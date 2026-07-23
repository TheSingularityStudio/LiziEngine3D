use serde::{Deserialize, Serialize};
use bincode;
use crate::core::grid::Grid2D;
use crate::core::particles::ParticleState;
use crate::core::sim::ElectrostaticSim2D;
use crate::core::boundary::BoundaryType;

/// LZ2D 文件魔数（前 4 字节）
const MAGIC_HEADER: &[u8; 4] = b"LZ2D";

/// 当前文件格式版本（v4 新增质量参数）
const LZ2D_VERSION: u32 = 4;

/// 内部序列化结构（不直接暴露）
#[derive(Serialize, Deserialize)]
struct Lz2dFile {
    version: u32,
    grid_nx: usize,
    grid_ny: usize,
    grid_dx: f64,
    grid_dy: f64,
    eps_poisson: f64,
    boundary_type: u8,  // 0=Periodic, 1=Reflective, 2=Open
    max_speed: Option<f64>,
    step_count: usize,
    particles_x: Vec<f64>,
    particles_y: Vec<f64>,
    particles_vx: Vec<f64>,
    particles_vy: Vec<f64>,
    particles_q: Vec<f64>,
    /// 粒子质量（v4 新增）
    particles_m: Vec<f64>,
    /// 重力参数（v2 新增）
    gravity_enabled: bool,
    gravity_x: f64,
    gravity_y: f64,
    /// 摩擦力参数（v3 新增）
    friction_enabled: bool,
    friction_damping: f64,
}

impl Lz2dFile {
    fn from_sim(sim: &ElectrostaticSim2D, step_count: usize) -> Self {
        let boundary_code = match sim.boundary_type {
            BoundaryType::Periodic => 0u8,
            BoundaryType::Reflective => 1u8,
            BoundaryType::Open => 2u8,
        };

        Self {
            version: LZ2D_VERSION,
            grid_nx: sim.grid.nx,
            grid_ny: sim.grid.ny,
            grid_dx: sim.grid.dx,
            grid_dy: sim.grid.dy,
            eps_poisson: sim.eps_poisson,
            boundary_type: boundary_code,
            max_speed: sim.max_speed,
            step_count,
            particles_x: sim.particles.x.to_vec(),
            particles_y: sim.particles.y.to_vec(),
            particles_vx: sim.particles.vx.to_vec(),
            particles_vy: sim.particles.vy.to_vec(),
            particles_q: sim.particles.q.to_vec(),
            particles_m: sim.particles.m.to_vec(),
            gravity_enabled: sim.gravity_enabled,
            gravity_x: sim.gravity_x,
            gravity_y: sim.gravity_y,
            friction_enabled: sim.friction_enabled,
            friction_damping: sim.friction_damping,
        }
    }

    fn into_sim(self) -> Result<(ElectrostaticSim2D, usize), String> {
        let boundary_type = match self.boundary_type {
            0 => BoundaryType::Periodic,
            1 => BoundaryType::Reflective,
            2 => BoundaryType::Open,
            other => return Err(format!("未知边界类型代码: {}", other)),
        };

        let n = self.particles_x.len();

        // 处理质量：旧版本文件可能没有 particles_m，使用默认值 1.0
        let masses = if self.particles_m.len() == n {
            self.particles_m
        } else {
            vec![1.0; n]
        };

        let particles = ParticleState {
            x: ndarray::Array1::from_vec(self.particles_x),
            y: ndarray::Array1::from_vec(self.particles_y),
            vx: ndarray::Array1::from_vec(self.particles_vx),
            vy: ndarray::Array1::from_vec(self.particles_vy),
            fx: ndarray::Array1::zeros(n),
            fy: ndarray::Array1::zeros(n),
            q: ndarray::Array1::from_vec(self.particles_q),
            m: ndarray::Array1::from_vec(masses),
        };

        let grid = Grid2D::new(self.grid_nx, self.grid_ny, self.grid_dx, self.grid_dy);

        let mut sim = ElectrostaticSim2D::with_config(
            grid,
            particles,
            self.eps_poisson,
            boundary_type,
            self.max_speed,
        );
        sim.gravity_enabled = self.gravity_enabled;
        sim.gravity_x = self.gravity_x;
        sim.gravity_y = self.gravity_y;
        sim.friction_enabled = self.friction_enabled;
        sim.friction_damping = self.friction_damping;

        Ok((sim, self.step_count))
    }
}

/// 将模拟器状态保存为 .lz2d 二进制文件
///
/// 格式: [魔数 4B] [版本号 4B(u32)] [bincode 编码的 Lz2dFile]
pub fn save_to_file(sim: &ElectrostaticSim2D, step_count: usize, path: &str) -> Result<(), String> {
    let lz2d = Lz2dFile::from_sim(sim, step_count);

    let encoded = bincode::serialize(&lz2d)
        .map_err(|e| format!("序列化失败: {}", e))?;

    let mut file_content = Vec::with_capacity(8 + encoded.len());
    file_content.extend_from_slice(MAGIC_HEADER);
    file_content.extend_from_slice(&LZ2D_VERSION.to_le_bytes());
    file_content.extend_from_slice(&encoded);

    std::fs::write(path, &file_content)
        .map_err(|e| format!("写入文件失败 '{}': {}", path, e))?;

    Ok(())
}

/// 从 .lz2d 二进制文件加载模拟器状态
///
/// 返回: (ElectrostaticSim2D, step_count)
pub fn load_from_file(path: &str) -> Result<(ElectrostaticSim2D, usize), String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("读取文件失败 '{}': {}", path, e))?;

    if data.len() < 8 {
        return Err("文件过短，不是有效的 LZ2D 文件".to_string());
    }

    // 验证魔数
    if &data[0..4] != MAGIC_HEADER {
        return Err("无效的 LZ2D 文件魔数".to_string());
    }

    // 读取版本号
    let _version = u32::from_le_bytes([
        data[4], data[5], data[6], data[7],
    ]);

    // 解析 bincode 负载
    let lz2d: Lz2dFile = bincode::deserialize(&data[8..])
        .map_err(|e| format!("反序列化失败: {}", e))?;

    lz2d.into_sim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::grid::Grid2D;
    use crate::core::particles::ParticleState;
    use crate::core::sim::ElectrostaticSim2D;

    fn create_test_sim() -> ElectrostaticSim2D {
        let grid = Grid2D::new(16, 16, 1.0, 1.0);
        let particles = ParticleState::zeros(5, Some(42));
        ElectrostaticSim2D::with_config(
            grid,
            particles,
            1e-12,
            BoundaryType::Periodic,
            Some(10.0),
        )
    }

    #[test]
    fn test_save_load_roundtrip() {
        let sim = create_test_sim();
        let temp_path = std::env::temp_dir().join("test_roundtrip.lz2d");
        let path_str = temp_path.to_string_lossy().to_string();

        save_to_file(&sim, 42, &path_str).unwrap();
        let (loaded, step_count) = load_from_file(&path_str).unwrap();

        assert_eq!(step_count, 42);
        assert_eq!(loaded.grid.nx, 16);
        assert_eq!(loaded.grid.ny, 16);
        assert_eq!(loaded.grid.dx, 1.0);
        assert_eq!(loaded.grid.dy, 1.0);
        assert_eq!(loaded.eps_poisson, 1e-12);
        assert_eq!(loaded.max_speed, Some(10.0));
        assert_eq!(loaded.boundary_type, BoundaryType::Periodic);
        assert_eq!(loaded.particles.len(), 5);

        // 验证粒子位置还原
        for i in 0..5 {
            assert!((loaded.particles.x[i] - sim.particles.x[i]).abs() < 1e-15);
            assert!((loaded.particles.y[i] - sim.particles.y[i]).abs() < 1e-15);
        }

        // 验证质量还原
        for i in 0..5 {
            assert!((loaded.particles.m[i] - 1.0).abs() < 1e-15);
        }

        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_invalid_magic() {
        // 构造一个大于 8 字节但不含魔数的文件
        let data = vec![0x00u8; 16];
        let temp_path = std::env::temp_dir().join("test_invalid.lz2d");
        std::fs::write(&temp_path, &data).unwrap();

        let result = load_from_file(&temp_path.to_string_lossy());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("魔数"));

        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_save_load_with_custom_mass() {
        let mut sim = create_test_sim();
        // 设置自定义质量
        for i in 0..sim.particles.len() {
            sim.particles.m[i] = (i as f64 + 1.0) * 0.5;
        }
        let temp_path = std::env::temp_dir().join("test_mass.lz2d");
        let path_str = temp_path.to_string_lossy().to_string();

        save_to_file(&sim, 0, &path_str).unwrap();
        let (loaded, _) = load_from_file(&path_str).unwrap();

        for i in 0..sim.particles.len() {
            assert!((loaded.particles.m[i] - sim.particles.m[i]).abs() < 1e-15);
        }

        std::fs::remove_file(&temp_path).ok();
    }
}