use crate::core::grid::Grid3D;
use crate::core::particles::ParticleState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoundaryType {
    #[default]
    Periodic,
    Reflective,
    Open,
}

impl BoundaryType {
    pub fn all() -> [BoundaryType; 3] {
        [BoundaryType::Periodic, BoundaryType::Reflective, BoundaryType::Open]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            BoundaryType::Periodic => "周期边界",
            BoundaryType::Reflective => "反弹边界",
            BoundaryType::Open => "虚空边界",
        }
    }
}

pub fn apply_boundary_conditions(
    particles: &mut ParticleState,
    grid: &Grid3D,
    boundary_type: BoundaryType,
) {
    let lx = grid.lx();
    let ly = grid.ly();
    let lz = grid.lz();
    match boundary_type {
        BoundaryType::Periodic => {
            for p in 0..particles.len() {
                particles.x[p] = ((particles.x[p] % lx) + lx) % lx;
                particles.y[p] = ((particles.y[p] % ly) + ly) % ly;
                particles.z[p] = ((particles.z[p] % lz) + lz) % lz;
            }
        }
        BoundaryType::Reflective => {
            for p in 0..particles.len() {
                if particles.x[p] < 0.0 {
                    particles.x[p] = -particles.x[p];
                    particles.vx[p] = -particles.vx[p];
                } else if particles.x[p] >= lx {
                    particles.x[p] = 2.0 * lx - particles.x[p];
                    particles.vx[p] = -particles.vx[p];
                }
                if particles.y[p] < 0.0 {
                    particles.y[p] = -particles.y[p];
                    particles.vy[p] = -particles.vy[p];
                } else if particles.y[p] >= ly {
                    particles.y[p] = 2.0 * ly - particles.y[p];
                    particles.vy[p] = -particles.vy[p];
                }
                if particles.z[p] < 0.0 {
                    particles.z[p] = -particles.z[p];
                    particles.vz[p] = -particles.vz[p];
                } else if particles.z[p] >= lz {
                    particles.z[p] = 2.0 * lz - particles.z[p];
                    particles.vz[p] = -particles.vz[p];
                }
                particles.x[p] = particles.x[p].clamp(0.0, lx);
                particles.y[p] = particles.y[p].clamp(0.0, ly);
                particles.z[p] = particles.z[p].clamp(0.0, lz);
            }
        }
        BoundaryType::Open => {
            let mut to_remove = Vec::new();
            for p in 0..particles.len() {
                if particles.x[p] < 0.0 || particles.x[p] >= lx
                    || particles.y[p] < 0.0 || particles.y[p] >= ly
                    || particles.z[p] < 0.0 || particles.z[p] >= lz
                {
                    to_remove.push(p);
                }
            }
            for &idx in to_remove.iter().rev() {
                particles.remove_particle(idx);
            }
        }
    }
}

pub fn apply_speed_limit(particles: &mut ParticleState, max_speed: f64) {
    if max_speed <= 0.0 { return; }
    let max_speed_sq = max_speed * max_speed;
    for p in 0..particles.len() {
        let speed_sq = particles.vx[p] * particles.vx[p]
            + particles.vy[p] * particles.vy[p]
            + particles.vz[p] * particles.vz[p];
        if speed_sq > max_speed_sq {
            let factor = max_speed / speed_sq.sqrt();
            particles.vx[p] *= factor;
            particles.vy[p] *= factor;
            particles.vz[p] *= factor;
        }
    }
}