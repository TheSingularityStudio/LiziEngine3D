use ndarray::Array1;
use lizi_engine_3d::core::grid::Grid3D;
use lizi_engine_3d::core::particles::ParticleState;
use lizi_engine_3d::core::sim::ElectrostaticSim3D;
use lizi_engine_3d::core::interp::gather_field_to_particles_trilinear;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[allow(dead_code)]
fn l2_norm(a: &Array1<f64>) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// 单电荷方向一致性验证（在 XY 平面）
#[test]
fn validate_single_charge_direction_consistency() {
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 1.0;
    let dy = 1.0;
    let dz = 1.0;
    let eps = 1e-12;

    let grid = Grid3D::new(nx, ny, nz, dx, dy, dz);

    let mut particles = ParticleState::zeros(1, Some(0));
    particles.x[0] = 0.5 * grid.lx();
    particles.y[0] = 0.5 * grid.ly();
    particles.z[0] = 0.5 * grid.lz();

    let mut sim = ElectrostaticSim3D::new(grid.clone(), particles, eps);
    sim.compute_fields();

    let ex = sim.ex.as_ref().expect("ex should be set");
    let ey = sim.ey.as_ref().expect("ey should be set");
    let ez = sim.ez.as_ref().expect("ez should be set");

    let cx = sim.particles.x[0] / grid.dx;
    let cy = sim.particles.y[0] / grid.dy;

    let sample_r = [4, 6, 8];
    let sample_angles: Vec<f64> = (0..12)
        .map(|i| 2.0 * std::f64::consts::PI * i as f64 / 12.0)
        .collect();

    let lx = grid.nx as f64;
    let ly = grid.ny as f64;

    let mut errors: Vec<f64> = Vec::new();

    for &r in &sample_r {
        for &th in &sample_angles {
            let gx = cx + r as f64 * th.cos();
            let gy = cy + r as f64 * th.sin();
            let _gz = 16.0; // mid-plane

            let gxw = ((gx % lx) + lx) % lx;
            let gyw = ((gy % ly) + ly) % ly;

            let qx = gxw * grid.dx;
            let qy = gyw * grid.dy;
            let qz = 0.5 * grid.lz();

            let mut qstate = ParticleState::zeros(1, Some(0));
            qstate.x[0] = qx;
            qstate.y[0] = qy;
            qstate.z[0] = qz;

            let (fx, fy, _fz) = gather_field_to_particles_trilinear(&grid, &qstate, ex, ey, ez);
            let exq = fx[0];
            let eyq = fy[0];

            let dxr = periodic_delta(gxw, cx, lx);
            let dyr = periodic_delta(gyw, cy, ly);
            let nr = (dxr * dxr + dyr * dyr).sqrt();
            if nr < 1e-9 { continue; }
            let rx = dxr / nr;
            let ry = dyr / nr;

            let en = (exq * exq + eyq * eyq).sqrt();
            if en < 1e-12 { continue; }

            let cosang = (exq * rx + eyq * ry) / en;
            errors.push(1.0 - cosang);
        }
    }

    let err = if errors.is_empty() { 0.0 } else { errors.iter().sum::<f64>() / errors.len() as f64 };
    assert!(err <= 6e-1, "平均方向误差(1-cos)={:.6e} 超出阈值", err);
}

fn periodic_delta(a: f64, b: f64, period: f64) -> f64 {
    let d = a - b;
    ((d + 0.5 * period).rem_euclid(period)) - 0.5 * period
}

/// 随机初始条件数值稳定性验证
#[test]
fn validate_random_numerical_stability() {
    let nx = 32;
    let ny = 32;
    let nz = 32;
    let dx = 1.0;
    let dy = 1.0;
    let dz = 1.0;
    let n = 200;
    let steps = 20;
    let dt = 0.05;
    let eps = 1e-12;

    let grid = Grid3D::new(nx, ny, nz, dx, dy, dz);
    let lx = grid.lx();
    let ly = grid.ly();
    let lz = grid.lz();

    let mut particles = ParticleState::zeros(n, Some(0));
    for i in 0..n {
        particles.x[i] *= lx;
        particles.y[i] *= ly;
        particles.z[i] *= lz;
    }
    let mut rng = StdRng::seed_from_u64(123);
    for i in 0..n {
        particles.vx[i] = (rng.gen::<f64>() - 0.5) * 0.02;
        particles.vy[i] = (rng.gen::<f64>() - 0.5) * 0.02;
        particles.vz[i] = (rng.gen::<f64>() - 0.5) * 0.02;
    }

    let mut sim = ElectrostaticSim3D::new(grid, particles, eps);

    let mut max_speed = 0.0f64;
    for _ in 0..steps {
        sim.step(dt);
        let speed_max = sim.particles.vx.iter()
            .zip(sim.particles.vy.iter())
            .zip(sim.particles.vz.iter())
            .map(|((vx, vy), vz)| (vx * vx + vy * vy + vz * vz).sqrt())
            .fold(0.0f64, f64::max);
        if speed_max > max_speed { max_speed = speed_max; }
    }

    assert!(max_speed <= 50.0, "多步最大速度 = {:.6e} 超出安全阈值", max_speed);
}