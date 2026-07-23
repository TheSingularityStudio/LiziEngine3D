/// 将浮点值映射到 RGB 热力图颜色（类似 viridis 风格）
/// value 应在 [min, max] 范围内
pub fn heatmap_rgb(value: f64, min: f64, max: f64) -> (u8, u8, u8) {
    let range = max - min;
    if range < 1e-12 {
        return (68, 1, 84); // 深紫，默认值
    }
    let t = ((value - min) / range).clamp(0.0, 1.0);

    let r = viridis_r(t);
    let g = viridis_g(t);
    let b = viridis_b(t);

    (r, g, b)
}

// 改进的 viridis colormap 近似（8 段分段线性，减少连接点颜色跳变）
fn viridis_r(t: f64) -> u8 {
    // 基于 viridis 关键节点插值，8 段均匀分段
    let nodes = [
        (0.000, 68.0),
        (0.125, 68.0),
        (0.250, 218.0),
        (0.375, 253.0),
        (0.500, 253.0),
        (0.625, 155.0),
        (0.750, 53.0),
        (0.875, 103.0),
        (1.000, 253.0),
    ];
    interpolate_color(t, &nodes)
}

fn viridis_g(t: f64) -> u8 {
    let nodes = [
        (0.000, 1.0),
        (0.125, 28.0),
        (0.250, 96.0),
        (0.375, 141.0),
        (0.500, 174.0),
        (0.625, 201.0),
        (0.750, 201.0),
        (0.875, 121.0),
        (1.000, 0.0),
    ];
    interpolate_color(t, &nodes)
}

fn viridis_b(t: f64) -> u8 {
    let nodes = [
        (0.000, 84.0),
        (0.125, 64.0),
        (0.250, 34.0),
        (0.375, 34.0),
        (0.500, 77.0),
        (0.625, 127.0),
        (0.750, 154.0),
        (0.875, 177.0),
        (1.000, 0.0),
    ];
    interpolate_color(t, &nodes)
}

/// 在分段节点之间进行线性插值
fn interpolate_color(t: f64, nodes: &[(f64, f64)]) -> u8 {
    if t <= nodes[0].0 {
        return nodes[0].1 as u8;
    }
    if t >= nodes.last().unwrap().0 {
        return nodes.last().unwrap().1 as u8;
    }
    for i in 0..nodes.len() - 1 {
        let (t0, v0) = nodes[i];
        let (t1, v1) = nodes[i + 1];
        if t >= t0 && t < t1 {
            let frac = (t - t0) / (t1 - t0);
            return (v0 + frac * (v1 - v0)) as u8;
        }
    }
    nodes.last().unwrap().1 as u8
}

/// 将 u32 RGB 打包为 0x00RRGGBB 格式
pub fn pack_rgb(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | (b as u32)
}