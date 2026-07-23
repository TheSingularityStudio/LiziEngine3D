use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolMode {
    DragParticle,
    PlaceParticle,
    DeleteParticle,
    Inspect,
}

impl ToolMode {
    pub fn all() -> [ToolMode; 4] {
        [ToolMode::DragParticle, ToolMode::PlaceParticle, ToolMode::DeleteParticle, ToolMode::Inspect]
    }
    pub fn display_name(&self) -> &'static str {
        match self {
            ToolMode::DragParticle => "拖动",
            ToolMode::PlaceParticle => "放置",
            ToolMode::DeleteParticle => "删除",
            ToolMode::Inspect => "查看",
        }
    }
    pub fn icon(&self) -> &'static str {
        match self {
            ToolMode::DragParticle => "\u{1F5B1}",
            ToolMode::PlaceParticle => "+",
            ToolMode::DeleteParticle => "-",
            ToolMode::Inspect => "\u{1F50D}",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlaceParticleParams {
    pub charge: f64,
    pub mass: f64,
    pub fixed: bool,
}

impl Default for PlaceParticleParams {
    fn default() -> Self { Self { charge: 1.0, mass: 1.0, fixed: false } }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementEntry {
    pub charge: f64,
    pub mass: f64,
    pub fixed: bool,
}

impl Default for PlacementEntry {
    fn default() -> Self { Self { charge: 1.0, mass: 1.0, fixed: false } }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ArrangeMode {
    Stack, Horizontal, Vertical, Grid,
}

impl ArrangeMode {
    pub fn all() -> [ArrangeMode; 4] {
        [ArrangeMode::Stack, ArrangeMode::Horizontal, ArrangeMode::Vertical, ArrangeMode::Grid]
    }
    pub fn display_name(&self) -> &'static str {
        match self {
            ArrangeMode::Stack => "堆叠",
            ArrangeMode::Horizontal => "水平排列",
            ArrangeMode::Vertical => "垂直排列",
            ArrangeMode::Grid => "网格排列",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementListData {
    pub entries: Vec<PlacementEntry>,
    pub spacing: f64,
    pub arrange_mode: ArrangeMode,
}

#[derive(Debug, Clone)]
pub struct PlacementList {
    pub enabled: bool,
    pub entries: Vec<PlacementEntry>,
    pub spacing: f64,
    pub arrange_mode: ArrangeMode,
}

impl Default for PlacementList {
    fn default() -> Self {
        Self { enabled: false, entries: vec![PlacementEntry::default()], spacing: 0.03, arrange_mode: ArrangeMode::Horizontal }
    }
}

impl PlacementList {
    pub fn export_json(&self) -> Result<String, String> {
        let data = PlacementListData { entries: self.entries.clone(), spacing: self.spacing, arrange_mode: self.arrange_mode };
        serde_json::to_string_pretty(&data).map_err(|e| format!("序列化失败: {}", e))
    }
    pub fn import_json(&mut self, json_str: &str) -> Result<(), String> {
        let data: PlacementListData = serde_json::from_str(json_str).map_err(|e| format!("反序列化失败: {}", e))?;
        self.entries = data.entries; self.spacing = data.spacing; self.arrange_mode = data.arrange_mode; Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct HoveredParticleInfo {
    pub index: usize,
    pub x: f64, pub y: f64, pub z: f64,
    pub vx: f64, pub vy: f64, pub vz: f64,
    pub q: f64, pub m: f64,
}

/// 3D 轨道相机状态
#[derive(Debug, Clone)]
pub struct OrbitCamera {
    pub azimuth: f32,   // 水平角度 (rad)
    pub elevation: f32, // 俯仰角度 (rad)
    pub distance: f32,  // 到原点的距离
    pub target_x: f32,
    pub target_y: f32,
    pub target_z: f32,
    pub last_mouse_pos: Option<(f32, f32)>,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            azimuth: std::f32::consts::FRAC_PI_4,
            elevation: std::f32::consts::FRAC_PI_6,
            distance: 3.0,
            target_x: 0.5, target_y: 0.5, target_z: 0.5,
            last_mouse_pos: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InteractionState {
    pub tool_mode: ToolMode,
    pub place_params: PlaceParticleParams,
    pub placement_list: PlacementList,
    pub dragging: bool,
    pub dragged_particle_index: Option<usize>,
    pub selection_radius: f64,
    pub camera: OrbitCamera,
    pub hovered_particle: Option<HoveredParticleInfo>,
    pub show_heatmap: bool,
    pub show_grid: bool,
    pub show_axes: bool,
}

impl Default for InteractionState {
    fn default() -> Self {
        Self {
            tool_mode: ToolMode::DragParticle,
            place_params: PlaceParticleParams::default(),
            placement_list: PlacementList::default(),
            dragging: false,
            dragged_particle_index: None,
            selection_radius: 0.05,
            camera: OrbitCamera::default(),
            hovered_particle: None,
            show_heatmap: true,
            show_grid: false,
            show_axes: true,
        }
    }
}

impl InteractionState {
    pub fn new() -> Self { Self::default() }
    pub fn reset_view(&mut self) { self.camera = OrbitCamera::default(); }

    pub fn compute_placement_offsets(&self) -> Vec<(f64, f64, f64)> {
        let count = self.placement_list.entries.len();
        if count == 0 { return Vec::new(); }
        let spacing = self.placement_list.spacing;
        match self.placement_list.arrange_mode {
            ArrangeMode::Stack => vec![(0.0, 0.0, 0.0); count],
            ArrangeMode::Horizontal => {
                let start = -(count as f64 - 1.0) / 2.0 * spacing;
                (0..count).map(|i| (start + i as f64 * spacing, 0.0, 0.0)).collect()
            }
            ArrangeMode::Vertical => {
                let start = -(count as f64 - 1.0) / 2.0 * spacing;
                (0..count).map(|i| (0.0, start + i as f64 * spacing, 0.0)).collect()
            }
            ArrangeMode::Grid => {
                let cols = (count as f64).sqrt().ceil() as usize;
                let rows = (count + cols - 1) / cols;
                let start_x = -(cols as f64 - 1.0) / 2.0 * spacing;
                let start_y = -(rows as f64 - 1.0) / 2.0 * spacing;
                (0..count).map(|i| {
                    (start_x + (i % cols) as f64 * spacing, start_y + (i / cols) as f64 * spacing, 0.0)
                }).collect()
            }
        }
    }
}