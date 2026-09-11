//pub mod raw_miniquad {
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;
    use ::log::{info,error};


    use las::{Point, Reader};
    use las::copc::{CopcEntryReader, VoxelKey};
    use miniquad::*;

    use crate::ColorMode;
    use crate::classify::{ClassEdits, ClassMask, effective_class};
    use crate::spatial::{RayHit, SpatialGrid};

    const CHUNK_SIZE: f32 = 50.0;

    /// Custom class definitions: id -> (label, [r,g,b])
    pub type ClassDefs = HashMap<u8, (String, [f32; 3])>;

    /// Height band to draw, in viewer space (metres above the shared viewer
    /// origin) — the same space as `Stage::positions[..][2]`. Points outside it
    /// are culled in the vertex shader and skipped by section extraction and
    /// picking, so a clipped point is not merely invisible: it cannot be
    /// measured, inspected or lassoed either.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct ZClip {
        pub min: f32,
        pub max: f32,
    }

    impl ZClip {
        /// Draws everything. Finite rather than infinite so it survives the
        /// trip through a `float` uniform on any GL ES implementation.
        pub const OPEN: ZClip = ZClip { min: -1.0e9, max: 1.0e9 };

        pub fn contains(&self, z: f32) -> bool {
            z >= self.min && z <= self.max
        }
    }

    impl Default for ZClip {
        fn default() -> Self { ZClip::OPEN }
    }

    /// Which classification ids are shown.
    ///
    /// Not a new type: an op's `from` restriction and the visible set are the
    /// same kind of thing and get intersected when an op is made, so one
    /// representation keeps that a single `&`. See [`ClassMask`].
    ///
    /// The [`ZClip`] doctrine applies verbatim, and this is the setting it was
    /// written for: a hidden class is not merely invisible — it cannot be
    /// measured, inspected, lassoed, sectioned, fitted to or snapped to
    /// either. Every site that consults `z_clip` consults this too. That is
    /// what makes it the primary control over *what you snap to*: hide
    /// everything but ground and "lowest Z in radius" means the ground.
    ///
    /// It is a **view setting**, like the height band. It is not written to
    /// `classes.txt`, and it does **not** affect export — *Omit classes* does
    /// that.
    pub type ClassVis = ClassMask;

    /// The colour given to a point its class filter hides.
    ///
    /// GLSL ES 100 has no bit operations and no vertex-stage discard, so
    /// visibility rides in on a channel that cannot hold a real value: a
    /// negative red. The vertex shader spots it and pushes the point past the
    /// far plane, the same trick `ZClip` already uses. The alternative was a
    /// `vec4` colour, which costs 4 bytes per point on the GPU — about 17% of
    /// vertex memory — and touches `Vertex`, `build_chunks`,
    /// `upload_split_verts` and section extraction. This costs nothing and
    /// makes `rebuild_colors` the single update path for both colour changes
    /// and visibility toggles.
    ///
    /// It overloads a channel, which is why it is named rather than written
    /// as a bare `-1.0` at either end.
    pub const HIDDEN_COLOR: [f32; 3] = [-1.0, 0.0, 0.0];

    /// Load class definitions from classes.txt in the current directory.
    /// Format: id;r;g;b;label  (one per line, # for comments)
    pub fn load_class_defs() -> Option<ClassDefs> {
        let path = Path::new("classes.txt");
        if !path.exists() { return None; }
        let content = std::fs::read_to_string(path).ok()?;
        let mut defs = HashMap::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let parts: Vec<&str> = line.splitn(5, ';').collect();
            if parts.len() < 5 { continue; }
            if let (Ok(id), Ok(r), Ok(g), Ok(b)) = (
                parts[0].trim().parse::<u8>(),
                parts[1].trim().parse::<u8>(),
                parts[2].trim().parse::<u8>(),
                parts[3].trim().parse::<u8>(),
            ) {
                let label = parts[4].trim().to_string();
                defs.insert(id, (label, [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0]));
            }
        }
        if defs.is_empty() { None } else {
            info!("Loaded {} class definitions from classes.txt", defs.len());
            Some(defs)
        }
    }

    /// Write class definitions to classes.txt in the current directory,
    /// in the same format `load_class_defs` reads: id;r;g;b;label.
    pub fn save_class_defs(defs: &ClassDefs) -> std::io::Result<()> {
        let mut ids: Vec<u8> = defs.keys().copied().collect();
        ids.sort();
        let mut out = String::from("# id;r;g;b;label\n");
        for id in ids {
            let (label, c) = &defs[&id];
            out.push_str(&format!(
                "{};{};{};{};{}\n",
                id,
                (c[0] * 255.0).round() as u8,
                (c[1] * 255.0).round() as u8,
                (c[2] * 255.0).round() as u8,
                label,
            ));
        }
        std::fs::write("classes.txt", out)
    }

    /// Default ASPRS label for a class id, used to seed the class editor for
    /// ids that have no classes.txt entry.
    pub fn default_class_label(id: u8) -> String {
        match id {
            0 => "Never Classified".to_string(),
            1 => "Unclassified".to_string(),
            2 => "Ground".to_string(),
            3 => "Low Vegetation".to_string(),
            4 => "Medium Vegetation".to_string(),
            5 => "High Vegetation".to_string(),
            6 => "Building".to_string(),
            7 => "Low Point".to_string(),
            8 => "Model Key Point".to_string(),
            9 => "Water".to_string(),
            10 => "Rail".to_string(),
            11 => "Road Surface".to_string(),
            12 => "Overlap Points".to_string(),
            13 => "Wire Guard".to_string(),
            14 => "Wire Conductor".to_string(),
            15 => "Transmission Tower".to_string(),
            16 => "Wire Structure Connector".to_string(),
            17 => "Bridge Deck".to_string(),
            18 => "High Noise".to_string(),
            _ => format!("Class {}", id),
        }
    }

    /// Point cloud backed by las::Point for full attribute access.
    pub struct VizPointcloud {
        pub points: Vec<Point>,
        pub min: [f64; 3],
        pub max: [f64; 3],
        pub intensity_min: u16,
        pub intensity_max: u16,
        pub time_min: f64,
        pub time_max: f64,
        /// Highest classification id present (class 12 recovered from `is_overlap`).
        pub max_class: u8,
    }

    impl VizPointcloud {
        /// Refresh `max_class` after points have been reclassified, so the
        /// Classes editor lists any id the operator has just introduced.
        pub fn recompute_max_class(&mut self) {
            self.max_class = self.points.iter()
                .map(|p| if p.is_overlap { 12 } else { p.classification.into() })
                .max()
                .unwrap_or(0);
        }

        pub fn load(filename: &str) -> VizPointcloud {
            let mut reader = Reader::from_path(filename).unwrap();
            let n = reader.header().number_of_points() as usize;

            let point_data = reader.read_all().unwrap();
            let mut points: Vec<Point> = Vec::with_capacity(n);
            for point in point_data.points() {
                points.push(point.unwrap());
            }

            let mut min = [f64::MAX; 3];
            let mut max = [f64::MIN; 3];
            let mut intensity_min = u16::MAX;
            let mut intensity_max = u16::MIN;
            let mut time_min = f64::MAX;
            let mut time_max = f64::MIN;
            let mut max_class: u8 = 0;

            for point in &points {
                min[0] = min[0].min(point.x);
                min[1] = min[1].min(point.y);
                min[2] = min[2].min(point.z);
                max[0] = max[0].max(point.x);
                max[1] = max[1].max(point.y);
                max[2] = max[2].max(point.z);
                intensity_min = intensity_min.min(point.intensity);
                intensity_max = intensity_max.max(point.intensity);

                if let Some(time) = point.gps_time {
                    time_min = time_min.min(time);
                    time_max = time_max.max(time);
                }

                let class_id: u8 = if point.is_overlap { 12 } else { point.classification.into() };
                max_class = max_class.max(class_id);
            }

            // Detect 8-bit colors stored in 16-bit fields (common in older LAS files)
            // and scale to the full 16-bit range so the 65535 divisor works correctly.
            let max_color = points.iter()
                .filter_map(|p| p.color.as_ref())
                .flat_map(|c| [c.red, c.green, c.blue])
                .max()
                .unwrap_or(0);
            if max_color > 0 && max_color <= 255 {
                for p in &mut points {
                    if let Some(ref mut c) = p.color {
                        c.red = c.red.saturating_mul(257);
                        c.green = c.green.saturating_mul(257);
                        c.blue = c.blue.saturating_mul(257);
                    }
                }
            }

            info!("{} points loaded from {}", points.len(), filename);
            info!("Intensity range: {}..{}", intensity_min, intensity_max);
            info!("Time range: {}..{}", time_min, time_max);
            VizPointcloud { points, min, max, intensity_min, intensity_max, time_min, time_max, max_class }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Vertex {
        pub pos: [f32; 3],
        pub color: [f32; 3],
    }

    pub struct Chunk {
        bindings: Bindings,
        num_points: i32,
        aabb_min: [f32; 3],
        aabb_max: [f32; 3],
        point_indices: Vec<usize>,
    }

    impl SectionData {
        /// Free the GPU buffers backing this section. A section is rebuilt on
        /// every colour, depth or endpoint change, so dropping one without
        /// this leaks a position, colour and index buffer each time.
        pub fn unload(self, ctx: &mut dyn RenderingBackend) {
            let Bindings { vertex_buffers, index_buffer, .. } = self.bindings;
            for vb in vertex_buffers {
                ctx.delete_buffer(vb);
            }
            ctx.delete_buffer(index_buffer);
        }
    }

    pub struct SectionData {
        bindings: Bindings,
        pub num_points: i32,
        pub length: f32,
        pub height_min: f32,
        pub height_max: f32,
        /// Full Z span of the source dataset — used by the shader for height-based colouring.
        pub height_range: f32,
        /// Section-space positions of the points drawn, as `[along, height, 0]`
        /// — the same filtered set that is on screen, so a snap in the section
        /// view can never land on a point the operator cannot see.
        points: Vec<[f32; 3]>,
        /// Index over `points` in (along, height), built on first use. A
        /// section is rebuilt on every colour or ramp change, and most of those
        /// sections are never digitised on, so paying for an index up front
        /// would put a build on every slider frame.
        index: std::cell::OnceCell<SpatialGrid>,
    }

    impl SectionData {
        /// The section's points and an index over them, for section-view
        /// snapping. Both are in (along, height) — the grid's X and Y.
        pub fn snap_index(&self) -> (&SpatialGrid, &[[f32; 3]]) {
            (self.index.get_or_init(|| SpatialGrid::build(&self.points)), &self.points)
        }
    }

    pub struct Stage {
        pipeline: Pipeline,
        /// Source file path — used as the UI label and to identify the stage.
        pub filepath: String,
        pub chunks: Vec<Chunk>,
        pub total_points: i32,
        pub cloud: VizPointcloud,
        pub positions: Vec<[f32; 3]>,
        /// Origin used to convert world coords → viewer-space `positions`.
        /// In single-file mode this equals `cloud.min`; in multi-stage mode all
        /// stages share one origin so they live in the same coordinate space.
        pub viewer_origin: [f64; 3],
        pub class_defs: Option<ClassDefs>,
        pub rgb_override: Option<[f32; 3]>,
        /// Whether this cloud is drawn. Hidden clouds stay resident on the GPU
        /// and keep their index in `stages`, they are only skipped when
        /// rendering, picking and cutting sections.
        pub visible: bool,
        /// Lower/upper bound of the height colour bar, relative to the dataset
        /// minimum (same space as `positions[..][2]`). Defaults to `0.0` and the
        /// full `height_range()`; editable from the UI.
        pub color_h_min: f32,
        pub color_h_max: f32,
        /// Height band to draw; see [`ZClip`].
        pub z_clip: ZClip,
        /// Which classes are shown; see [`ClassVis`].
        pub class_vis: ClassVis,
        /// Manual classification ops. Unlike the COPC case all points are
        /// resident, so ops are applied directly and only kept for persistence
        /// and undo.
        pub edits: ClassEdits,
        /// XY index over `positions`. Built once in `Stage::new` and never
        /// invalidated: all points are resident, and `positions` is written
        /// once and never touched again — reclassification and `reload_points`
        /// replace `cloud.points` and the colours, not the geometry.
        /// Unused until phase 4 wires up snapping — see spatial.rs.
        #[allow(dead_code)]
        pub index: SpatialGrid,
    }

    #[allow(clippy::too_many_arguments)]
    fn color_for_point(
        point: &Point, pos_z: f32, h_color_min: f32, h_color_max: f32,
        intensity_min: u16, intensity_range: f32, time: f64, time_min: f64, time_range: f64,
        mode: ColorMode, class_defs: &Option<ClassDefs>,
        override_rgb: Option<[f32; 3]>, class_vis: ClassVis,
    ) -> [f32; 3] {
        if !class_vis.contains(effective_class(point)) {
            return HIDDEN_COLOR;
        }
        match mode {
            ColorMode::Height => {
                // `pos_z` is relative to the dataset minimum; the colour bar maps
                // [h_color_min, h_color_max] (also relative to that minimum) onto [0, 1].
                let span = h_color_max - h_color_min;
                let h = if span > 0.0 { ((pos_z - h_color_min) / span).clamp(0.0, 1.0) } else { 0.5 };
                let r = (2.0 * h - 0.5).clamp(0.0, 1.0);
                let g = (1.0 - 2.0 * (h - 0.5).abs()).clamp(0.0, 1.0);
                let b = (1.0 - 2.0 * h).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMode::Time => {
                // GPS time is absolute (~1e8..1e9 s), so the offset must be taken
                // in f64: an f32 round-trip quantises it to ~32 s steps.
                let h = if time_range > 0.0 {
                    (((time - time_min) / time_range) as f32).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let r = (2.0 * h - 0.5).clamp(0.0, 1.0);
                let g = (1.0 - 2.0 * (h - 0.5).abs()).clamp(0.0, 1.0);
                let b = (1.0 - 2.0 * h).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMode::Intensity => {
                let h = if intensity_range > 0.0 {
                    let lin = (point.intensity - intensity_min) as f32 / intensity_range;
                    ((1.0 + lin * 99.0).log10() / 2.0).clamp(0.0, 1.0)
                } else {
                    0.5
                };
                let r = (2.0 * h - 0.5).clamp(0.0, 1.0);
                let g = (1.0 - 2.0 * (h - 0.5).abs()).clamp(0.0, 1.0);
                let b = (1.0 - 2.0 * h).clamp(0.0, 1.0);
                [r, g, b]
            }
            ColorMode::Classification => {
                let class_id: u8 = if point.is_overlap { 12 } else { point.classification.into() };
                if let Some(defs) = class_defs &&
                    let Some((_label, color)) = defs.get(&class_id) {
                        return *color;
                    
                }
                classification_color(class_id)
            }
            ColorMode::Rgb => {
                if let Some(c) = override_rgb {
                    return c;
                }
                if let Some(color) = &point.color {
                    let r = color.red as f32 / 65535.0;
                    let g = color.green as f32 / 65535.0;
                    let b = color.blue as f32 / 65535.0;
                    [r, g, b]
                } else {
                    [0.5, 0.5, 0.5]
                }
            }
            ColorMode::PointSourceId => {
                let psid = point.point_source_id;
                let h = (psid as f32 * 0.618_034) % 1.0;
                hsv_to_rgb(h, 0.8, 0.9)
            }
        }
    }

    pub fn classification_color(class: u8) -> [f32; 3] {
        match class {
            0 => [0.5, 0.5, 0.5],
            1 => [0.6, 0.6, 0.6],
            2 => [0.6, 0.4, 0.2],
            3 => [0.0, 0.8, 0.0],
            4 => [0.0, 0.6, 0.0],
            5 => [0.0, 0.4, 0.0],
            6 => [1.0, 0.0, 0.0],
            7 => [1.0, 0.5, 0.0],
            8 => [0.8, 0.8, 0.0],
            9 => [0.0, 0.0, 1.0],
            10 => [0.8, 0.8, 0.8],
            11 => [0.3, 0.3, 0.3],
            12 => [0.9, 0.9, 0.0],
            13 => [0.7, 0.7, 0.0],
            14 => [0.9, 0.9, 0.0],
            15 => [0.4, 0.0, 0.6],
            16 => [0.6, 0.0, 0.8],
            17 => [0.2, 0.2, 0.6],
            18 => [1.0, 0.0, 0.5],
            _ => {
                let h = (class as f32 * 0.618_034) % 1.0;
                hsv_to_rgb(h, 0.8, 0.9)
            }
        }
    }

    fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
        let i = (h * 6.0).floor();
        let f = h * 6.0 - i;
        let p = v * (1.0 - s);
        let q = v * (1.0 - f * s);
        let t = v * (1.0 - (1.0 - f) * s);
        match (i as i32) % 6 {
            0 => [v, t, p],
            1 => [q, v, p],
            2 => [p, v, t],
            3 => [p, q, v],
            4 => [t, p, v],
            _ => [v, p, q],
        }
    }

    fn extract_frustum_planes(vp: &glam::Mat4) -> [[f32; 4]; 6] {
        let c = vp.to_cols_array_2d();
        let mut planes = [[0.0f32; 4]; 6];
        for i in 0..4 {
            planes[0][i] = c[i][3] + c[i][0];
            planes[1][i] = c[i][3] - c[i][0];
            planes[2][i] = c[i][3] + c[i][1];
            planes[3][i] = c[i][3] - c[i][1];
            planes[4][i] = c[i][3] + c[i][2];
            planes[5][i] = c[i][3] - c[i][2];
        }
        for plane in &mut planes {
            let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
            if len > 0.0 {
                for v in plane.iter_mut() {
                    *v /= len;
                }
            }
        }
        planes
    }

    fn aabb_outside_frustum(planes: &[[f32; 4]; 6], amin: &[f32; 3], amax: &[f32; 3]) -> bool {
        for plane in planes {
            let px = if plane[0] >= 0.0 { amax[0] } else { amin[0] };
            let py = if plane[1] >= 0.0 { amax[1] } else { amin[1] };
            let pz = if plane[2] >= 0.0 { amax[2] } else { amin[2] };
            if plane[0] * px + plane[1] * py + plane[2] * pz + plane[3] < 0.0 {
                return true;
            }
        }
        false
    }

    fn create_pipeline(ctx: &mut dyn RenderingBackend) -> Pipeline {
        // On desktop OpenGL, gl_PointSize in the vertex shader is only respected
        // when GL_PROGRAM_POINT_SIZE (0x8642) is enabled. This is always active in
        // OpenGL ES / WebGL so no guard is needed, but on desktop it must be set.
        unsafe { miniquad::gl::glEnable(0x8642); }

        let shader = ctx
            .new_shader(
                miniquad::ShaderSource::Glsl {
                    vertex: shader::VERTEX,
                    fragment: shader::FRAGMENT,
                },
                shader::meta(),
            )
            .unwrap();

        // Positions and colours live in separate vertex buffers. Only the
        // colours ever change — a recolour after a ramp or class edit then
        // re-uploads half as many bytes, and never touches the position data.
        ctx.new_pipeline(
            &[BufferLayout::default(), BufferLayout::default()],
            &[
                VertexAttribute::with_buffer("pos", VertexFormat::Float3, 0),
                VertexAttribute::with_buffer("color", VertexFormat::Float3, 1),
            ],
            shader,
            PipelineParams {
                primitive_type: PrimitiveType::Points,
                depth_test: Comparison::LessOrEqual,
                depth_write: true,
                ..Default::default()
            },
        )
    }

    /// Free every GPU buffer a chunk owns. Positions and colours sit in
    /// separate buffers, so both have to be released.
    fn free_chunk(ctx: &mut dyn RenderingBackend, chunk: Chunk) {
        let Bindings { vertex_buffers, index_buffer, .. } = chunk.bindings;
        for vb in vertex_buffers {
            ctx.delete_buffer(vb);
        }
        ctx.delete_buffer(index_buffer);
    }

    /// Upload interleaved section vertices as the separate position and
    /// colour buffers the pipeline expects. Sections are static, so both are
    /// immutable.
    fn upload_split_verts(
        ctx: &mut dyn RenderingBackend,
        verts: &[Vertex],
    ) -> (BufferId, BufferId) {
        let positions: Vec<[f32; 3]> = verts.iter().map(|v| v.pos).collect();
        let colors: Vec<[f32; 3]> = verts.iter().map(|v| v.color).collect();
        (
            ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&positions),
            ),
            ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&colors),
            ),
        )
    }

    fn build_chunks(
        ctx: &mut dyn RenderingBackend,
        positions: &[[f32; 3]],
        colors: &[[f32; 3]],
    ) -> Vec<Chunk> {
        let mut chunk_map: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (i, pos) in positions.iter().enumerate() {
            let cx = (pos[0] / CHUNK_SIZE).floor() as i32;
            let cy = (pos[1] / CHUNK_SIZE).floor() as i32;
            chunk_map.entry((cx, cy)).or_default().push(i);
        }

        chunk_map.into_values()
            .map(|point_indices| {
                let chunk_pos: Vec<[f32; 3]> =
                    point_indices.iter().map(|&i| positions[i]).collect();
                let chunk_colors: Vec<[f32; 3]> =
                    point_indices.iter().map(|&i| colors[i]).collect();
                let mut aabb_min = [f32::MAX; 3];
                let mut aabb_max = [f32::MIN; 3];
                for pos in &chunk_pos {
                    for j in 0..3 {
                        aabb_min[j] = aabb_min[j].min(pos[j]);
                        aabb_max[j] = aabb_max[j].max(pos[j]);
                    }
                }

                let n = chunk_pos.len() as i32;
                let pos_buffer = ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&chunk_pos),
                );
                let color_buffer = ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Dynamic,
                    BufferSource::slice(&chunk_colors),
                );
                let indices: Vec<u32> = (0..n as u32).collect();
                let index_buffer = ctx.new_buffer(
                    BufferType::IndexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&indices),
                );

                Chunk {
                    bindings: Bindings {
                        vertex_buffers: vec![pos_buffer, color_buffer],
                        index_buffer,
                        images: vec![],
                    },
                    num_points: n,
                    aabb_min,
                    aabb_max,
                    point_indices,
                }
            })
            .collect()
    }

    impl Stage {
        pub fn new(
            ctx: &mut dyn RenderingBackend,
            filename: &str,
            viewer_origin: Option<[f64; 3]>,
            color_mode: ColorMode,
            class_defs: Option<ClassDefs>,
        ) -> Stage {
            let cloud = VizPointcloud::load(filename);

            let height_range = (cloud.max[2] - cloud.min[2]) as f32;
            let intensity_range = (cloud.intensity_max - cloud.intensity_min) as f32;
            let time_range = cloud.time_max - cloud.time_min;

            let viewer_origin = viewer_origin.unwrap_or(cloud.min);

            let positions: Vec<[f32; 3]> = cloud
                .points
                .iter()
                .map(|p| [
                    (p.x - viewer_origin[0]) as f32,
                    (p.y - viewer_origin[1]) as f32,
                    (p.z - viewer_origin[2]) as f32,
                ])
                .collect();

            let colors: Vec<[f32; 3]> = cloud
                .points
                .iter()
                .enumerate()
                .map(|(i, p)| color_for_point(p, positions[i][2], 0.0, height_range, cloud.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), cloud.time_min, time_range, color_mode, &class_defs, None, ClassVis::ALL))
                .collect();

            let pipeline = create_pipeline(ctx);
            let total_points = colors.len() as i32;
            let chunks = build_chunks(ctx, &positions, &colors);
            let index = SpatialGrid::build(&positions);

            info!("{} points in {} chunks ({}m cells), height range: {:.1}m, index cell {:.2}m",
                total_points, chunks.len(), CHUNK_SIZE, height_range, index.cell_size());

            let mut stage = Stage {
                pipeline,
                filepath: filename.to_string(),
                chunks,
                total_points,
                cloud,
                positions,
                viewer_origin,
                class_defs,
                rgb_override: None,
                visible: true,
                color_h_min: 0.0,
                color_h_max: height_range,
                z_clip: ZClip::OPEN,
                // Starts unfiltered; `apply_class_vis` pushes the live set on
                // the next frame, which is also how a stage loaded while a
                // filter is in force picks it up.
                class_vis: ClassVis::ALL,
                edits: ClassEdits::default(),
                index,
            };
            // Replay any sidecar written in an earlier session before the
            // cloud is first shown, so the view opens on the edited state.
            stage.edits = ClassEdits::load(filename);
            if !stage.edits.is_empty() {
                let n = stage.edits.apply_all(&mut stage.cloud.points);
                stage.cloud.recompute_max_class();
                info!("{}: replayed {} ops, {} points reclassified", filename, stage.edits.ops.len(), n);
                stage.rebuild_colors(ctx, color_mode);
            }
            stage
        }

        /// Append every resident point inside the XY box and Z band, in
        /// absolute world coordinates. Used to gather evidence for a ground
        /// plane fit; the caller does the polygon test.
        pub fn collect_points_in_bbox(
            &self,
            min: (f64, f64),
            max: (f64, f64),
            z_min: f64,
            z_max: f64,
            out: &mut Vec<(f64, f64, f64)>,
        ) {
            for p in &self.cloud.points {
                if !self.class_vis.contains(effective_class(p)) { continue; }
                if p.x >= min.0 && p.x <= max.0 && p.y >= min.1 && p.y <= max.1
                    && p.z >= z_min && p.z <= z_max {
                    out.push((p.x, p.y, p.z));
                }
            }
        }

        /// Re-read the points from disk and replay the current op list over
        /// them. This is how an undo is honoured: an op records what it
        /// assigns, not what it overwrote, so the only way back is to start
        /// from the file again. Point order is deterministic, so `positions`
        /// and the GPU chunks stay valid — only colours need refreshing.
        pub fn reload_points(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let fresh = VizPointcloud::load(&self.filepath);
            self.cloud.points = fresh.points;
            self.edits.apply_all(&mut self.cloud.points);
            self.cloud.recompute_max_class();
            self.rebuild_colors(ctx, color_mode);
        }

        /// Re-run the op list over all resident points and refresh the colours.
        /// Returns the number of points changed.
        pub fn apply_edits(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode, from_idx: usize) -> usize {
            let bmin = self.cloud.min;
            let bmax = self.cloud.max;
            let changed = self.edits.apply_from(from_idx, &mut self.cloud.points, &bmin, &bmax);
            self.cloud.recompute_max_class();
            if changed > 0 {
                self.rebuild_colors(ctx, color_mode);
            }
            changed
        }

        /// Free all GPU buffers held by this stage. Consumes the stage.
        pub fn unload(mut self, ctx: &mut dyn RenderingBackend) {
            for chunk in self.chunks.drain(..) {
                free_chunk(ctx, chunk);
            }
        }

        pub fn find_nearest_point(
            &self,
            view_proj: &glam::Mat4,
            screen_x: f32,
            screen_y: f32,
            screen_w: f32,
            screen_h: f32,
        ) -> Option<usize> {
            let ndc_x = screen_x / screen_w * 2.0 - 1.0;
            let ndc_y = 1.0 - screen_y / screen_h * 2.0;

            let mut best_idx = None;
            let mut best_dist = f32::MAX;

            for (i, pos) in self.positions.iter().enumerate() {
                // A clipped or hidden point is not merely invisible — it must
                // not be pickable either, or Measure and Point Info would
                // silently snap to geometry the operator cannot see.
                if !self.z_clip.contains(pos[2]) { continue; }
                if !self.class_vis.contains(effective_class(&self.cloud.points[i])) { continue; }
                let p = *view_proj * glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                let px = p.x / p.w;
                let py = p.y / p.w;
                let dist = (px - ndc_x) * (px - ndc_x) + (py - ndc_y) * (py - ndc_y);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(i);
                }
            }

            best_idx
        }

        /// Point-visibility test for every indexed query on this stage.
        ///
        /// The `ZClip` doctrine — a clipped point cannot be measured, inspected
        /// or lassoed either — reaches the snap queries here, and it is passed
        /// *into* the grid walk rather than applied to the result: a hidden
        /// point that is nearer than every visible one would otherwise win the
        /// contest and be discarded, leaving the snap empty over data the
        /// operator can plainly see. Class visibility joins this test in
        /// phase 1.
        #[allow(dead_code)]
        fn snap_filter(&self) -> impl Fn(u32) -> bool + '_ {
            move |i: u32| {
                self.z_clip.contains(self.positions[i as usize][2])
                    && self.class_vis.contains(effective_class(&self.cloud.points[i as usize]))
            }
        }

        /// Frontmost point within `max_dist` of a viewer-space ray.
        ///
        /// Unlike `find_nearest_point`, which ranks by screen distance, this is
        /// a true 3D query — which is what makes digitising work from an
        /// oblique camera instead of only from directly above.
        #[allow(dead_code)]
        pub fn snap_nearest_ray(&self, origin: [f32; 3], dir: [f32; 3], max_dist: f32) -> Option<RayHit> {
            self.index.nearest_ray(&self.positions, origin, dir, max_dist, self.snap_filter())
        }

        /// Append every visible point within `r` metres of a viewer-space XY
        /// position as `(z, position)`, for a percentile taken across backends.
        ///
        /// Candidates rather than an answer, deliberately: a percentile over a
        /// union is not the union of percentiles, so the pooling has to happen
        /// before the rank is taken. A snap radius straddling two tiles, or a
        /// stage and a COPC file loaded together, is the normal case.
        #[allow(dead_code)]
        pub fn snap_z_candidates(&self, x: f32, y: f32, r: f32, out: &mut Vec<(f32, [f32; 3])>) {
            if !self.index.circle_overlaps(x, y, r) { return; }
            let mut buf = Vec::new();
            self.index.points_in_radius(&self.positions, x, y, r, self.snap_filter(), &mut buf);
            out.extend(buf.iter().map(|&i| {
                let p = self.positions[i as usize];
                (p[2], p)
            }));
        }

        /// Every visible point within `r` metres of a viewer-space XY position,
        /// appended to `out` as indices into `positions` / `cloud.points`.
        #[allow(dead_code)]
        pub fn snap_points_in_radius(&self, x: f32, y: f32, r: f32, out: &mut Vec<u32>) {
            self.index.points_in_radius(&self.positions, x, y, r, self.snap_filter(), out);
        }

        pub fn height_range(&self) -> f32 {
            (self.cloud.max[2] - self.cloud.min[2]) as f32
        }

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);
            let intensity_range = (self.cloud.intensity_max - self.cloud.intensity_min) as f32;
            let time_range = self.cloud.time_max - self.cloud.time_min;

            for chunk in &mut self.chunks {
                let chunk_colors: Vec<[f32; 3]> = chunk
                    .point_indices
                    .iter()
                    .map(|&i| {
                        let p = &self.cloud.points[i];
                        color_for_point(p, self.positions[i][2], h_lo, h_hi, self.cloud.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), self.cloud.time_min, time_range, color_mode, &self.class_defs, self.rgb_override, self.class_vis)
                    })
                    .collect();

                ctx.buffer_update(
                    chunk.bindings.vertex_buffers[1],
                    BufferSource::slice(&chunk_colors),
                );
            }
        }

        /// Extract section vertices for this stage (no GPU upload), so several
        /// stages can be combined into a single SectionData.
        pub fn extract_section_verts(
            &self,
            start: (f32, f32),
            end: (f32, f32),
            depth: f32,
            color_mode: ColorMode,
        ) -> (Vec<Vertex>, f32, f32, f32) {
            let dx = end.0 - start.0;
            let dy = end.1 - start.1;
            let len = (dx * dx + dy * dy).sqrt();
            let dir = (dx / len, dy / len);
            let normal = (-dir.1, dir.0);

            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);
            let intensity_range = (self.cloud.intensity_max - self.cloud.intensity_min) as f32;
            let time_range = self.cloud.time_max - self.cloud.time_min;

            let mut section_verts = Vec::new();
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;

            for (i, pos) in self.positions.iter().enumerate() {
                let rel_x = pos[0] - start.0;
                let rel_y = pos[1] - start.1;
                let along = rel_x * dir.0 + rel_y * dir.1;
                let perp = rel_x * normal.0 + rel_y * normal.1;

                if along >= 0.0 && along <= len && perp.abs() <= depth
                    && self.z_clip.contains(pos[2])
                    && self.class_vis.contains(effective_class(&self.cloud.points[i])) {
                    let p = &self.cloud.points[i];
                    h_min = h_min.min(pos[2]);
                    h_max = h_max.max(pos[2]);
                    section_verts.push(Vertex {
                        pos: [along, pos[2], 0.0],
                        color: color_for_point(p, pos[2], h_lo, h_hi, self.cloud.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), self.cloud.time_min, time_range, color_mode, &self.class_defs, self.rgb_override, self.class_vis),
                    });
                }
            }

            (section_verts, len, h_min, h_max)
        }

        pub fn draw_section(&self, ctx: &mut dyn RenderingBackend, section: &SectionData, mvp: &glam::Mat4, point_size: f32) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp, point_size, height_range: section.height_range,
                    z_clip_min: ZClip::OPEN.min, z_clip_max: ZClip::OPEN.max },
            ));
            ctx.draw(0, section.num_points, 1);
        }

        pub fn draw_visible(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, point_size: f32) -> i32 {
            if !self.visible { return 0; }
            let planes = extract_frustum_planes(view_proj);
            let mut drawn = 0;
            let height_range = self.height_range();

            ctx.apply_pipeline(&self.pipeline);
            for chunk in &self.chunks {
                if aabb_outside_frustum(&planes, &chunk.aabb_min, &chunk.aabb_max) {
                    continue;
                }
                ctx.apply_bindings(&chunk.bindings);
                ctx.apply_uniforms(miniquad::UniformsSource::table(
                    &shader::Uniforms { mvp: *view_proj, point_size, height_range,
                        z_clip_min: self.z_clip.min, z_clip_max: self.z_clip.max },
                ));
                ctx.draw(0, chunk.num_points, 1);
                drawn += chunk.num_points;
            }
            drawn
        }
    }

    // ---- Tile management ----

    pub struct TileInfo {
        pub filename: String,
        pub filepath: String,
        pub xmin: f64, pub ymin: f64, pub zmin: f64,
        pub xmax: f64, pub ymax: f64, pub zmax: f64,
    }

    pub struct LoadedTile {
        pub cloud: VizPointcloud,
        pub positions: Vec<[f32; 3]>,
        chunks: Vec<Chunk>,
        pub total_points: i32,
        pub rgb_override: Option<[f32; 3]>,
        /// XY index over this tile's `positions`. Built on load, dropped with
        /// the tile on unload — there is nothing to invalidate, because a tile
        /// never changes shape while it is resident.
        /// Unused until phase 4 wires up snapping — see spatial.rs.
        #[allow(dead_code)]
        pub index: SpatialGrid,
    }

    pub struct TileStageManager {
        pipeline: Pipeline,
        pub tiles: Vec<TileInfo>,
        pub loaded: Vec<Option<LoadedTile>>,
        pub global_min: [f64; 3],
        pub global_max: [f64; 3],
        pub class_defs: Option<ClassDefs>,
        /// Height colour-bar bounds, relative to `global_min[2]`. See `Stage::color_h_min`.
        pub color_h_min: f32,
        pub color_h_max: f32,
        /// Height band to draw; see [`ZClip`].
        pub z_clip: ZClip,
        /// Which classes are shown; see [`ClassVis`].
        pub class_vis: ClassVis,
    }

    fn read_index(dir_path: &str) -> Vec<TileInfo> {
        let index_path = Path::new(dir_path).join("index.txt");
        let content = std::fs::read_to_string(index_path).unwrap();
        content.lines()
            .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                TileInfo {
                    filename: parts[0].to_string(),
                    filepath: Path::new(dir_path).join(parts[0]).to_string_lossy().to_string(),
                    xmin: parts[1].parse().unwrap(),
                    ymin: parts[2].parse().unwrap(),
                    zmin: parts[3].parse().unwrap(),
                    xmax: parts[4].parse().unwrap(),
                    ymax: parts[5].parse().unwrap(),
                    zmax: parts[6].parse().unwrap(),
                }
            })
            .collect()
    }

    /// Return paths to all COPC files in `dir_path`.
    pub fn scan_dir_for_copc(dir_path: &str) -> Vec<String> {
        let mut paths = Vec::new();
        let Ok(read_dir) = std::fs::read_dir(dir_path) else { return paths; };
        let mut entries: Vec<_> = read_dir.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "laz" || ext_lower == "las" {
                    let is_copc = Reader::from_path(&path)
                        .map(|r| r.header().copc_info_vlr().is_some())
                        .unwrap_or(false);
                    if is_copc {
                        paths.push(path.to_string_lossy().into_owned());
                    }
                }
            }
        }
        paths
    }

    fn generate_index(dir_path: &str) {
        info!("Generating index.txt for {}", dir_path);
        let mut entries = Vec::new();
        let mut dir_entries: Vec<_> = std::fs::read_dir(dir_path).unwrap()
            .filter_map(|e| e.ok())
            .collect();
        dir_entries.sort_by_key(|e| e.file_name());

        for entry in dir_entries {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "laz" || ext_lower == "las" {
                    let reader = Reader::from_path(&path).unwrap();
                    // Skip COPC files — they are handled separately as CopcStage instances.
                    if reader.header().copc_info_vlr().is_some() { continue; }
                    let bounds = reader.header().bounds();
                    let filename = path.file_name().unwrap().to_string_lossy();
                    entries.push(format!("{} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3}",
                        filename,
                        bounds.min.x, bounds.min.y, bounds.min.z,
                        bounds.max.x, bounds.max.y, bounds.max.z,
                    ));
                }
            }
        }
        let index_path = Path::new(dir_path).join("index.txt");
        std::fs::write(&index_path, entries.join("\n")).unwrap();
        info!("Wrote {} tile entries to index.txt", entries.len());
    }

    impl TileStageManager {
        pub fn new(ctx: &mut dyn RenderingBackend, dir_path: &str, class_defs: Option<ClassDefs>) -> TileStageManager {
            // Generate index if missing
            let index_path = Path::new(dir_path).join("index.txt");
            if !index_path.exists() {
                generate_index(dir_path);
            }

            let tiles = read_index(dir_path);

            // Compute global bounds
            let mut global_min = [f64::MAX; 3];
            let mut global_max = [f64::MIN; 3];
            for t in &tiles {
                global_min[0] = global_min[0].min(t.xmin);
                global_min[1] = global_min[1].min(t.ymin);
                global_min[2] = global_min[2].min(t.zmin);
                global_max[0] = global_max[0].max(t.xmax);
                global_max[1] = global_max[1].max(t.ymax);
                global_max[2] = global_max[2].max(t.zmax);
            }

            let n = tiles.len();
            let loaded = (0..n).map(|_| None).collect();

            if n > 0 {
            info!("{} tiles, bounds: x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}",
                n, global_min[0], global_max[0], global_min[1], global_max[1], global_min[2], global_max[2]);
            }
            
            let pipeline = create_pipeline(ctx);

            TileStageManager {
                pipeline,
                tiles,
                loaded,
                global_min,
                global_max,
                class_defs,
                color_h_min: 0.0,
                color_h_max: (global_max[2] - global_min[2]) as f32,
                z_clip: ZClip::OPEN,
                class_vis: ClassVis::ALL,
            }
        }

        pub fn height_range(&self) -> f32 {
            (self.global_max[2] - self.global_min[2]) as f32
        }

        /// Highest classification id present across the currently loaded tiles.
        pub fn max_class(&self) -> u8 {
            self.loaded.iter().flatten().map(|lt| lt.cloud.max_class).max().unwrap_or(0)
        }

        /// Tile boundary corners in viewer coordinates (at z=0)
        pub fn tile_corners(&self, idx: usize) -> [(f32, f32); 4] {
            let t = &self.tiles[idx];
            let x0 = (t.xmin - self.global_min[0]) as f32;
            let y0 = (t.ymin - self.global_min[1]) as f32;
            let x1 = (t.xmax - self.global_min[0]) as f32;
            let y1 = (t.ymax - self.global_min[1]) as f32;
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        }

        /// Tile center in viewer coordinates
        pub fn tile_center(&self, idx: usize) -> (f32, f32) {
            let t = &self.tiles[idx];
            let cx = ((t.xmin + t.xmax) / 2.0 - self.global_min[0]) as f32;
            let cy = ((t.ymin + t.ymax) / 2.0 - self.global_min[1]) as f32;
            (cx, cy)
        }

        /// Find which tile contains the given viewer-space XY coordinate
        pub fn tile_at(&self, wx: f32, wy: f32) -> Option<usize> {
            for (i, t) in self.tiles.iter().enumerate() {
                let x0 = (t.xmin - self.global_min[0]) as f32;
                let y0 = (t.ymin - self.global_min[1]) as f32;
                let x1 = (t.xmax - self.global_min[0]) as f32;
                let y1 = (t.ymax - self.global_min[1]) as f32;
                if wx >= x0 && wx <= x1 && wy >= y0 && wy <= y1 {
                    return Some(i);
                }
            }
            None
        }

        pub fn load_tile(&mut self, ctx: &mut dyn RenderingBackend, index: usize, color_mode: ColorMode, override_rgb: Option<[f32; 3]>) {
            if self.loaded[index].is_some() { return; }

            let cloud = VizPointcloud::load(&self.tiles[index].filepath);
            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);
            let intensity_range = (cloud.intensity_max - cloud.intensity_min) as f32;
            let time_range = cloud.time_max - cloud.time_min;

            let positions: Vec<[f32; 3]> = cloud.points.iter().map(|p| [
                (p.x - self.global_min[0]) as f32,
                (p.y - self.global_min[1]) as f32,
                (p.z - self.global_min[2]) as f32,
            ]).collect();

            let colors: Vec<[f32; 3]> = cloud.points.iter().enumerate().map(|(i, p)| {
                color_for_point(p, positions[i][2], h_lo, h_hi, cloud.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), cloud.time_min, time_range, color_mode, &self.class_defs, override_rgb, self.class_vis)
            }).collect();

            let total_points = colors.len() as i32;
            let chunks = build_chunks(ctx, &positions, &colors);
            let grid = SpatialGrid::build(&positions);

            info!("Tile {} loaded: {} points, {} chunks, index cell {:.2}m",
                self.tiles[index].filename, total_points, chunks.len(), grid.cell_size());

            self.loaded[index] = Some(LoadedTile {
                cloud,
                positions,
                chunks,
                total_points,
                rgb_override: override_rgb,
                index: grid,
            });
        }

        pub fn unload_tile(&mut self, ctx: &mut dyn RenderingBackend, index: usize) {
            if let Some(tile) = self.loaded[index].take() {
                for chunk in tile.chunks {
                    free_chunk(ctx, chunk);
                }
                info!("Tile {} unloaded", self.tiles[index].filename);
            }
        }

        pub fn draw_visible(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, point_size: f32) -> (i32, i32) {
            let planes = extract_frustum_planes(view_proj);
            let mut drawn = 0;
            let mut total = 0;
            let height_range = self.height_range();

            ctx.apply_pipeline(&self.pipeline);
            for loaded in self.loaded.iter().flatten() {
                total += loaded.total_points;
                for chunk in &loaded.chunks {
                    if aabb_outside_frustum(&planes, &chunk.aabb_min, &chunk.aabb_max) {
                        continue;
                    }
                    ctx.apply_bindings(&chunk.bindings);
                    ctx.apply_uniforms(miniquad::UniformsSource::table(
                        &shader::Uniforms { mvp: *view_proj, point_size, height_range,
                        z_clip_min: self.z_clip.min, z_clip_max: self.z_clip.max },
                    ));
                    ctx.draw(0, chunk.num_points, 1);
                    drawn += chunk.num_points;
                }
            }
            (drawn, total)
        }

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);
            let class_defs = self.class_defs.clone();
            let class_vis = self.class_vis;
            for loaded in self.loaded.iter_mut().flatten() {
                let intensity_range = (loaded.cloud.intensity_max - loaded.cloud.intensity_min) as f32;
                let time_range = loaded.cloud.time_max - loaded.cloud.time_min;
                let override_rgb = loaded.rgb_override;
                for chunk in &mut loaded.chunks {
                    let chunk_colors: Vec<[f32; 3]> = chunk.point_indices.iter().map(|&i| {
                        let p = &loaded.cloud.points[i];
                        color_for_point(p, loaded.positions[i][2], h_lo, h_hi, loaded.cloud.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), loaded.cloud.time_min, time_range, color_mode, &class_defs, override_rgb, class_vis)
                    }).collect();
                    ctx.buffer_update(chunk.bindings.vertex_buffers[1], BufferSource::slice(&chunk_colors));
                }
            }
        }

        pub fn rebuild_tile_colors(&mut self, ctx: &mut dyn RenderingBackend, index: usize, color_mode: ColorMode) {
            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);
            let class_defs = self.class_defs.clone();
            let class_vis = self.class_vis;
            if let Some(ref mut loaded) = self.loaded[index] {
                let intensity_range = (loaded.cloud.intensity_max - loaded.cloud.intensity_min) as f32;
                let time_range = loaded.cloud.time_max - loaded.cloud.time_min;
                let override_rgb = loaded.rgb_override;
                for chunk in &mut loaded.chunks {
                    let chunk_colors: Vec<[f32; 3]> = chunk.point_indices.iter().map(|&i| {
                        let p = &loaded.cloud.points[i];
                        color_for_point(p, loaded.positions[i][2], h_lo, h_hi, loaded.cloud.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), loaded.cloud.time_min, time_range, color_mode, &class_defs, override_rgb, class_vis)
                    }).collect();
                    ctx.buffer_update(chunk.bindings.vertex_buffers[1], BufferSource::slice(&chunk_colors));
                }
            }
        }

        pub fn find_nearest_point(
            &self,
            view_proj: &glam::Mat4,
            screen_x: f32, screen_y: f32,
            screen_w: f32, screen_h: f32,
        ) -> Option<(usize, usize)> {
            let ndc_x = screen_x / screen_w * 2.0 - 1.0;
            let ndc_y = 1.0 - screen_y / screen_h * 2.0;

            let mut best: Option<(usize, usize)> = None;
            let mut best_dist = f32::MAX;

            for (tile_idx, loaded) in self.loaded.iter().enumerate() {
                if let Some(lt) = loaded {
                    for (i, pos) in lt.positions.iter().enumerate() {
                        if !self.z_clip.contains(pos[2]) { continue; }
                        if !self.class_vis.contains(effective_class(&lt.cloud.points[i])) { continue; }
                        let p = *view_proj * glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                        let px = p.x / p.w;
                        let py = p.y / p.w;
                        let dist = (px - ndc_x) * (px - ndc_x) + (py - ndc_y) * (py - ndc_y);
                        if dist < best_dist {
                            best_dist = dist;
                            best = Some((tile_idx, i));
                        }
                    }
                }
            }
            best
        }

        /// Frontmost point within `max_dist` of a viewer-space ray, across
        /// every loaded tile. Returns the tile and the hit within it.
        ///
        /// Each tile answers for its own points and the frontmost of those
        /// answers wins, which is why `RayHit` carries `t`: a ray crossing a
        /// tile boundary must not prefer whichever tile happened to be visited
        /// first.
        #[allow(dead_code)]
        pub fn snap_nearest_ray(
            &self,
            origin: [f32; 3],
            dir: [f32; 3],
            max_dist: f32,
        ) -> Option<(usize, RayHit)> {
            let mut best: Option<(usize, RayHit)> = None;
            for (tile_idx, loaded) in self.loaded.iter().enumerate() {
                let Some(lt) = loaded else { continue };
                let z_clip = self.z_clip;
                let class_vis = self.class_vis;
                let positions = &lt.positions;
                let points = &lt.cloud.points;
                let hit = lt.index.nearest_ray(
                    positions, origin, dir, max_dist,
                    move |i: u32| z_clip.contains(positions[i as usize][2])
                        && class_vis.contains(effective_class(&points[i as usize])),
                );
                if let Some(h) = hit
                    && best.as_ref().is_none_or(|(_, b)| h.t < b.t) {
                    best = Some((tile_idx, h));
                }
            }
            best
        }

        /// Append every visible point within `r` metres of a viewer-space XY
        /// position as `(z, position)`. See `Stage::snap_z_candidates`.
        #[allow(dead_code)]
        pub fn snap_z_candidates(&self, x: f32, y: f32, r: f32, out: &mut Vec<(f32, [f32; 3])>) {
            let mut buf: Vec<u32> = Vec::new();
            for loaded in self.loaded.iter().flatten() {
                if !loaded.index.circle_overlaps(x, y, r) { continue; }
                buf.clear();
                let z_clip = self.z_clip;
                let class_vis = self.class_vis;
                let positions = &loaded.positions;
                let points = &loaded.cloud.points;
                loaded.index.points_in_radius(
                    positions, x, y, r,
                    move |i: u32| z_clip.contains(positions[i as usize][2])
                        && class_vis.contains(effective_class(&points[i as usize])),
                    &mut buf,
                );
                out.extend(buf.iter().map(|&i| {
                    let p = loaded.positions[i as usize];
                    (p[2], p)
                }));
            }
        }

        /// Append every resident point inside the XY box and Z band, in
        /// absolute world coordinates. Used to gather evidence for a ground
        /// plane fit; the caller does the polygon test.
        pub fn collect_points_in_bbox(
            &self,
            min: (f64, f64),
            max: (f64, f64),
            z_min: f64,
            z_max: f64,
            out: &mut Vec<(f64, f64, f64)>,
        ) {
            for loaded in self.loaded.iter().flatten() {
                for p in &loaded.cloud.points {
                    if !self.class_vis.contains(effective_class(p)) { continue; }
                    if p.x >= min.0 && p.x <= max.0 && p.y >= min.1 && p.y <= max.1
                        && p.z >= z_min && p.z <= z_max {
                        out.push((p.x, p.y, p.z));
                    }
                }
            }
        }

        pub fn loaded_count(&self) -> usize {
            self.loaded.iter().filter(|l| l.is_some()).count()
        }

        pub fn total_chunks(&self) -> usize {
            self.loaded.iter().flatten().map(|l| l.chunks.len()).sum()
        }

        pub fn extract_section(
            &self,
            ctx: &mut dyn RenderingBackend,
            start: (f32, f32),
            end: (f32, f32),
            depth: f32,
            color_mode: ColorMode,
        ) -> SectionData {
            let dx = end.0 - start.0;
            let dy = end.1 - start.1;
            let len = (dx * dx + dy * dy).sqrt();
            let dir = (dx / len, dy / len);
            let normal = (-dir.1, dir.0);

            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);

            let mut section_verts = Vec::new();
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;

            for loaded in self.loaded.iter().flatten() {
                let intensity_range = (loaded.cloud.intensity_max - loaded.cloud.intensity_min) as f32;
                let time_range = loaded.cloud.time_max - loaded.cloud.time_min;
                for (i, pos) in loaded.positions.iter().enumerate() {
                    let rel_x = pos[0] - start.0;
                    let rel_y = pos[1] - start.1;
                    let along = rel_x * dir.0 + rel_y * dir.1;
                    let perp = rel_x * normal.0 + rel_y * normal.1;

                    if along >= 0.0 && along <= len && perp.abs() <= depth
                        && self.z_clip.contains(pos[2])
                        && self.class_vis.contains(effective_class(&loaded.cloud.points[i])) {
                        let p = &loaded.cloud.points[i];
                        h_min = h_min.min(pos[2]);
                        h_max = h_max.max(pos[2]);
                        section_verts.push(Vertex {
                            pos: [along, pos[2], 0.0],
                            color: color_for_point(p, pos[2], h_lo, h_hi, loaded.cloud.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), loaded.cloud.time_min, time_range, color_mode, &self.class_defs, loaded.rgb_override, self.class_vis),
                        });
                    }
                }
            }

            info!("Section: {:.1}m long, {:.1}m depth, {} points, height {:.1}..{:.1}m",
                len, depth, section_verts.len(), h_min, h_max);
            make_section_data(ctx, section_verts, len, h_min, h_max, self.height_range())
        }

        pub fn draw_section(&self, ctx: &mut dyn RenderingBackend, section: &SectionData, mvp: &glam::Mat4, point_size: f32) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp, point_size, height_range: section.height_range,
                    z_clip_min: ZClip::OPEN.min, z_clip_max: ZClip::OPEN.max },
            ));
            ctx.draw(0, section.num_points, 1);
        }
    }

    // ---- EDL (Eye-Dome Lighting) renderer ----

    pub struct EdlRenderer {
        offscreen_pass: RenderPass,
        color_texture: TextureId,
        depth_texture: TextureId,
        edl_pipeline: Pipeline,
        quad_bindings: Bindings,
        pub width: u32,
        pub height: u32,
    }

    impl EdlRenderer {
        pub fn new(ctx: &mut dyn RenderingBackend, width: u32, height: u32) -> Self {
            let (offscreen_pass, color_texture, depth_texture, edl_pipeline, quad_bindings) =
                Self::create_resources(ctx, width, height);
            Self { offscreen_pass, color_texture, depth_texture, edl_pipeline, quad_bindings, width, height }
        }

        fn create_resources(
            ctx: &mut dyn RenderingBackend,
            width: u32,
            height: u32,
        ) -> (RenderPass, TextureId, TextureId, Pipeline, Bindings) {
            let color_texture = ctx.new_render_texture(TextureParams {
                format: TextureFormat::RGBA8,
                wrap: TextureWrap::Clamp,
                min_filter: FilterMode::Nearest,
                mag_filter: FilterMode::Nearest,
                width,
                height,
                ..Default::default()
            });
            let depth_texture = ctx.new_render_texture(TextureParams {
                format: TextureFormat::Depth,
                wrap: TextureWrap::Clamp,
                min_filter: FilterMode::Nearest,
                mag_filter: FilterMode::Nearest,
                width,
                height,
                ..Default::default()
            });
            let offscreen_pass = ctx.new_render_pass(color_texture, Some(depth_texture));

            let verts: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            let indices: [u32; 6] = [0, 1, 2, 0, 2, 3];
            let vbuf = ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&verts),
            );
            let ibuf = ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&indices),
            );
            let shader = ctx.new_shader(
                ShaderSource::Glsl {
                    vertex: edl_shader::VERTEX,
                    fragment: edl_shader::FRAGMENT,
                },
                edl_shader::meta(),
            ).unwrap();
            let edl_pipeline = ctx.new_pipeline(
                &[BufferLayout::default()],
                &[VertexAttribute::new("pos", VertexFormat::Float2)],
                shader,
                PipelineParams::default(),
            );
            let quad_bindings = Bindings {
                vertex_buffers: vec![vbuf],
                index_buffer: ibuf,
                images: vec![color_texture],
            };
            (offscreen_pass, color_texture, depth_texture, edl_pipeline, quad_bindings)
        }

        pub fn resize_if_needed(&mut self, ctx: &mut dyn RenderingBackend, width: u32, height: u32) {
            if self.width == width && self.height == height { return; }
            ctx.delete_render_pass(self.offscreen_pass);
            ctx.delete_texture(self.color_texture);
            ctx.delete_texture(self.depth_texture);
            let (offscreen_pass, color_texture, depth_texture, edl_pipeline, quad_bindings) =
                Self::create_resources(ctx, width, height);
            self.offscreen_pass = offscreen_pass;
            self.color_texture = color_texture;
            self.depth_texture = depth_texture;
            self.edl_pipeline = edl_pipeline;
            self.quad_bindings = quad_bindings;
            self.width = width;
            self.height = height;
        }

        /// Begin rendering point cloud to offscreen texture (replaces begin_default_pass).
        pub fn begin(&self, ctx: &mut dyn RenderingBackend) {
            ctx.begin_pass(
                Some(self.offscreen_pass),
                PassAction::Clear {
                    color: Some((0.0, 0.0, 0.0, 0.0)),
                    depth: Some(1.0),
                    stencil: None,
                },
            );
        }

        /// Apply EDL post-process to the default framebuffer.
        /// Call after the draw_visible call; replaces end_render_pass.
        /// Composite the shaded offscreen image into `viewport` of the default
        /// framebuffer.
        ///
        /// The target rect is passed in rather than assumed to be the whole
        /// window: the docked panel takes the left edge, and the offscreen
        /// target is sized to what is left, so the full-screen quad has to land
        /// on exactly that region or the image is stretched across the panel.
        pub fn apply(
            &self,
            ctx: &mut dyn RenderingBackend,
            strength: f32,
            viewport: (i32, i32, i32, i32),
        ) {
            ctx.end_render_pass();
            ctx.begin_default_pass(PassAction::Nothing);
            ctx.apply_viewport(viewport.0, viewport.1, viewport.2, viewport.3);
            ctx.apply_pipeline(&self.edl_pipeline);
            ctx.apply_bindings(&self.quad_bindings);
            ctx.apply_uniforms(UniformsSource::table(&edl_shader::Uniforms {
                px_size: [1.0 / self.width as f32, 1.0 / self.height as f32],
                strength,
            }));
            ctx.draw(0, 6, 1);
            ctx.end_render_pass();
        }
    }

    pub mod edl_shader {
        use miniquad::*;

        pub const VERTEX: &str = r#"#version 100
            attribute vec2 pos;
            varying lowp vec2 v_uv;
            void main() {
                gl_Position = vec4(pos, 0.0, 1.0);
                v_uv = pos * 0.5 + 0.5;
            }"#;

        pub const FRAGMENT: &str = r#"#version 100
            uniform sampler2D tex;
            uniform mediump vec2 px_size;
            uniform mediump float strength;
            varying lowp vec2 v_uv;
            void main() {
                lowp vec4 c = texture2D(tex, v_uv);
                mediump float d = c.a;
                if (d < 0.001) {
                    discard;
                }
                mediump float d_l = texture2D(tex, v_uv + vec2(-px_size.x, 0.0)).a;
                mediump float d_r = texture2D(tex, v_uv + vec2( px_size.x, 0.0)).a;
                mediump float d_u = texture2D(tex, v_uv + vec2(0.0,  px_size.y)).a;
                mediump float d_d = texture2D(tex, v_uv + vec2(0.0, -px_size.y)).a;
                if (d_l < 0.001) d_l = d;
                if (d_r < 0.001) d_r = d;
                if (d_u < 0.001) d_u = d;
                if (d_d < 0.001) d_d = d;
                mediump float r = 0.0;
                r += max(0.0, d_l - d);
                r += max(0.0, d_r - d);
                r += max(0.0, d_u - d);
                r += max(0.0, d_d - d);
                mediump float shade = exp(-strength * r);
                gl_FragColor = vec4(c.rgb * shade, 1.0);
            }"#;

        pub fn meta() -> ShaderMeta {
            ShaderMeta {
                images: vec!["tex".to_string()],
                uniforms: UniformBlockLayout {
                    uniforms: vec![
                        UniformDesc::new("px_size", UniformType::Float2),
                        UniformDesc::new("strength", UniformType::Float1),
                    ],
                },
            }
        }

        #[repr(C)]
        pub struct Uniforms {
            pub px_size: [f32; 2],
            pub strength: f32,
        }
    }

    // ---- COPC Stage ----

    /// Upper bound on the points a snap neighbourhood pin may hold.
    ///
    /// Analogous to `CORRIDOR_POINT_LIMIT`, but far smaller: a section corridor
    /// is pinned once, deliberately, and the operator waits for it. A snap
    /// neighbourhood follows the cursor, so it must stay small enough to stream
    /// in without the tool ever feeling like it stalled.
    #[allow(dead_code)]
    pub const SNAP_PIN_POINT_LIMIT: i64 = 5_000_000;

    struct CopcEntry {
        key: VoxelKey,
        offset: u64,
        byte_size: i32,
        point_count: i32,
        xmin: f64, ymin: f64, zmin: f64,
        xmax: f64, ymax: f64, zmax: f64,
    }

    struct CopcLoadedNode {
        chunks: Vec<Chunk>,
        positions: Vec<[f32; 3]>,
        points: Vec<Point>,
        total_points: i32,
        /// XY index over this node's `positions`.
        ///
        /// One grid per node, built at ingest. `loaded_nodes` churns
        /// continuously as the camera moves, so a single grid over the resident
        /// set would have to be rebuilt every time a node arrived or was
        /// evicted — a rebuild storm exactly when the operator is moving. A
        /// per-node grid is built once, dropped with the node, and never
        /// invalidated; a query walks only the nodes whose bounds it meets.
        #[allow(dead_code)]
        index: SpatialGrid,
    }

    /// The neighbourhood `CopcStage::pin_neighbourhood` last pinned for, in
    /// world coordinates, and what it cost.
    #[derive(Clone, Copy)]
    struct PinBox {
        cx: f64,
        cy: f64,
        half: f64,
        points: i64,
    }

    /// Request sent to the background loader thread.
    struct CopcLoadReq {
        key: VoxelKey,
        offset: u64,
        byte_size: i32,
        point_count: i32,
        generation: u64,
        /// Pinned loads are exempt from generation filtering — they are
        /// requested for a section corridor, not for the current view, so a
        /// camera nudge must not discard them.
        pinned: bool,
    }

    /// Result returned by the background loader thread.
    struct CopcLoadResult {
        key: VoxelKey,
        points: Vec<Point>,
        generation: u64,
        pinned: bool,
    }

    pub struct CopcStage {
        pipeline: Pipeline,
        /// Source file path — used as the UI label and to identify the stage.
        pub filepath: String,
        /// Root spacing in metres (halved at each level)
        pub spacing: f64,
        entries: Vec<CopcEntry>,
        loaded_nodes: HashMap<VoxelKey, CopcLoadedNode>,
        pub global_min: [f64; 3],
        pub global_max: [f64; 3],
        /// World-space origin used for viewer-space coordinate conversion.
        /// Equals `global_min` for single-file mode; set to a shared minimum
        /// across all files in directory mode so all stages share one coordinate space.
        pub viewer_origin: [f64; 3],
        pub intensity_min: u16,
        pub intensity_max: u16,
        pub time_min: f64,
        pub time_max: f64,
        pub class_defs: Option<ClassDefs>,
        pub rgb_override: Option<[f32; 3]>,
        /// Whether this cloud is drawn. Hidden clouds keep loading their LOD in
        /// the background and keep their index in `copc_stages`, they are only
        /// skipped when rendering, picking and cutting sections.
        pub visible: bool,
        /// Highest classification id seen across all ingested nodes.
        pub max_class: u8,
        /// Height colour-bar bounds, relative to `global_min[2]`. See `Stage::color_h_min`.
        pub color_h_min: f32,
        pub color_h_max: f32,
        /// Height band to draw; see [`ZClip`].
        pub z_clip: ZClip,
        /// Which classes are shown; see [`ClassVis`]. Applied in
        /// `ingest_node`, so a node arriving long after a class was switched
        /// off is coloured to match what is already on screen.
        pub class_vis: ClassVis,
        /// Nodes dispatched to the loader thread but not yet ingested.
        in_flight: HashSet<VoxelKey>,
        /// Nodes force-loaded for the active section corridor. These are held
        /// regardless of LOD, frustum and point budget, so that what the
        /// operator classifies is the full-resolution data rather than
        /// whatever sample the LOD happened to pick.
        pinned: HashSet<VoxelKey>,
        /// The box `pin_neighbourhood` last pinned for. Lets a cursor that has
        /// barely moved reuse the existing pin set instead of rescanning the
        /// hierarchy every frame. `None` when the pins came from somewhere
        /// else, or from nowhere.
        pin_box: Option<PinBox>,
        /// Manual classification ops, applied to every node as it is ingested.
        pub edits: ClassEdits,
        /// Send requests to the background loader thread.
        load_tx: std::sync::mpsc::Sender<CopcLoadReq>,
        /// Receive completed results from the background loader thread.
        result_rx: std::sync::mpsc::Receiver<CopcLoadResult>,
        /// Time of the last view change; dispatch is deferred until the view settles.
        last_view_change: std::time::Instant,
        /// View-projection matrix from the previous frame, used to detect view changes.
        prev_view_proj: glam::Mat4,
        /// Incremented on every view change; results from older generations are discarded.
        generation: u64,
    }

    /// Flat snapshot of one COPC hierarchy entry: a leaf pointing at a data chunk.
    pub struct RawCopcEntry {
        pub key: VoxelKey,
        pub offset: u64,
        pub byte_size: i32,
        pub point_count: i32,
    }

    /// Walk the COPC hierarchy EVLR recursively and return every leaf entry
    /// (point_count > 0). The `las` 0.9 crate's `CopcEntryReader::hierarchy_entries`
    /// only resolves one level of sub-pages, so files written with deeply nested
    /// pages (e.g. copc-converter, which boundaries every 3 octree levels) appear
    /// to contain only a handful of coarse-LOD nodes. We bypass it by parsing the
    /// raw EVLR bytes ourselves and following page pointers to any depth.
    pub fn read_full_hierarchy(reader: &CopcEntryReader<BufReader<File>>) -> Vec<RawCopcEntry> {
        let header = reader.header();
        let Some(info_vlr) = header.vlrs().iter().find(|v| v.is_copc_info()) else {
            return Vec::new();
        };
        // CopcInfo VLR payload layout (160 bytes): the page-table fields we need are
        //   root_hier_offset: u64 @ byte 40,  root_hier_size: u64 @ byte 48.
        // These are private on las::copc::CopcInfoVlr so we read them from the raw payload.
        if info_vlr.data.len() < 56 { return Vec::new(); }
        let root_hier_offset = u64::from_le_bytes(info_vlr.data[40..48].try_into().unwrap());
        let root_hier_size = u64::from_le_bytes(info_vlr.data[48..56].try_into().unwrap());

        let Some(hier_vlr) = header.evlrs().iter().find(|v| v.is_copc_hierarchy()) else {
            return Vec::new();
        };
        let data = &hier_vlr.data;

        let mut out = Vec::new();
        let mut stack: Vec<(u64, u64)> = vec![(root_hier_offset, root_hier_size)];
        while let Some((page_offset, page_size)) = stack.pop() {
            // Page pointer offsets are absolute file offsets; the EVLR payload begins
            // at root_hier_offset (the writer lays the root page first in the EVLR data,
            // and all sub-pages live in the same payload).
            let Some(start) = page_offset.checked_sub(root_hier_offset).map(|v| v as usize) else { continue; };
            let end = start.saturating_add(page_size as usize);
            if end > data.len() { continue; }
            for chunk in data[start..end].chunks_exact(32) {
                let l = i32::from_le_bytes(chunk[0..4].try_into().unwrap());
                let x = i32::from_le_bytes(chunk[4..8].try_into().unwrap());
                let y = i32::from_le_bytes(chunk[8..12].try_into().unwrap());
                let z = i32::from_le_bytes(chunk[12..16].try_into().unwrap());
                let offset = u64::from_le_bytes(chunk[16..24].try_into().unwrap());
                let byte_size = i32::from_le_bytes(chunk[24..28].try_into().unwrap());
                let point_count = i32::from_le_bytes(chunk[28..32].try_into().unwrap());
                if point_count == -1 {
                    stack.push((offset, byte_size as u64));
                } else if point_count > 0 {
                    out.push(RawCopcEntry {
                        key: VoxelKey { l, x, y, z },
                        offset, byte_size, point_count,
                    });
                }
            }
        }
        out
    }

    /// Compute (xmin,ymin,zmin,xmax,ymax,zmax) for a VoxelKey given COPC octree params.
    pub fn voxel_bounds(key: &VoxelKey, cx: f64, cy: f64, cz: f64, halfsize: f64) -> (f64, f64, f64, f64, f64, f64) {
        let root_min_x = cx - halfsize;
        let root_min_y = cy - halfsize;
        let root_min_z = cz - halfsize;
        let span = halfsize * 2.0 / (1i64 << key.l) as f64;
        let xmin = root_min_x + span * key.x as f64;
        let ymin = root_min_y + span * key.y as f64;
        let zmin = root_min_z + span * key.z as f64;
        (xmin, ymin, zmin, xmin + span, ymin + span, zmin + span)
    }

    /// 2D separating-axis test between the section corridor (an oriented box:
    /// the segment `a`..`b` widened by `depth` on each side) and an axis-aligned
    /// rectangle. Used to decide which octree nodes a corridor can touch.
    ///
    /// Exact rather than conservative — a bounding-box approximation of a
    /// diagonal corridor pulls in far more nodes than it needs to, and force-
    /// loading is already the expensive path.
    pub fn corridor_meets_aabb_2d(
        a: (f64, f64),
        b: (f64, f64),
        depth: f64,
        rect_min: (f64, f64),
        rect_max: (f64, f64),
    ) -> bool {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 0.0 { return false; }
        let dir = (dx / len, dy / len);
        let normal = (-dir.1, dir.0);

        let obb_center = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
        let obb_half = (len * 0.5, depth);
        let rect_center = ((rect_min.0 + rect_max.0) * 0.5, (rect_min.1 + rect_max.1) * 0.5);
        let rect_half = ((rect_max.0 - rect_min.0) * 0.5, (rect_max.1 - rect_min.1) * 0.5);

        let d = (rect_center.0 - obb_center.0, rect_center.1 - obb_center.1);

        // Corridor axes: project the rectangle's extent onto dir and normal.
        for (axis, half) in [(dir, obb_half.0), (normal, obb_half.1)] {
            let dist = (d.0 * axis.0 + d.1 * axis.1).abs();
            let r = rect_half.0 * axis.0.abs() + rect_half.1 * axis.1.abs();
            if dist > half + r { return false; }
        }
        // World axes: project the corridor's extent onto x and y.
        for (axis, half) in [((1.0, 0.0), rect_half.0), ((0.0, 1.0), rect_half.1)] {
            let dist: f64 = (d.0 * axis.0 + d.1 * axis.1).abs();
            let r = obb_half.0 * (dir.0 * axis.0 + dir.1 * axis.1).abs()
                  + obb_half.1 * (normal.0 * axis.0 + normal.1 * axis.1).abs();
            if dist > half + r { return false; }
        }
        true
    }

    /// Number of background threads decompressing COPC nodes in parallel.
    /// LAZ decompression is CPU-bound and per-node independent, so a small pool
    /// gives near-linear speedup on cold zoom-ins. Capped to avoid oversubscribing
    /// a typical workstation when other stages are also active.
    const COPC_LOADER_THREADS: usize = 4;

    /// Spawn a pool of background threads that service COPC load requests.
    /// Each thread reopens the file per request so the LAZ decompressor starts
    /// from a clean state — each COPC node is a self-contained LAZ chunk and
    /// reusing a reader across nodes can leave stale decompressor state.
    /// All threads share one request receiver via `Arc<Mutex<_>>`; the lock is
    /// only held during `recv()`, so decompression itself runs concurrently.
    fn spawn_copc_loader_pool(
        filepath: String,
        rx: std::sync::mpsc::Receiver<CopcLoadReq>,
        tx: std::sync::mpsc::Sender<CopcLoadResult>,
        worker_count: usize,
    ) {
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        for i in 0..worker_count {
            let filepath = filepath.clone();
            let rx = rx.clone();
            let tx = tx.clone();
            std::thread::Builder::new()
                .name(format!("copc-loader-{i}"))
                .spawn(move || {
                    loop {
                        let req = match rx.lock().unwrap().recv() {
                            Ok(req) => req,
                            Err(_) => break,
                        };
                        let file = match File::open(&filepath) {
                            Ok(f) => BufReader::new(f),
                            Err(e) => { error!("COPC loader: open error: {}", e); continue; }
                        };
                        let mut reader = match CopcEntryReader::new(file) {
                            Ok(r) => r,
                            Err(e) => { error!("COPC loader: reader error: {}", e); continue; }
                        };
                        let las_entry = las::copc::Entry {
                            key: req.key,
                            offset: req.offset,
                            byte_size: req.byte_size,
                            point_count: req.point_count,
                        };
                        let mut points = Vec::new();
                        match reader.read_entry_points(&las_entry, &mut points) {
                            Ok(_) => { let _ = tx.send(CopcLoadResult { key: req.key, points, generation: req.generation, pinned: req.pinned }); }
                            Err(e) => error!("COPC loader error for {:?}: {}", req.key, e),
                        }
                    }
                })
                .expect("Cannot spawn COPC loader thread");
        }
    }

    impl CopcStage {
        pub fn new(
            ctx: &mut dyn RenderingBackend,
            filepath: &str,
            viewer_origin: Option<[f64; 3]>,
            _color_mode: ColorMode,
            class_defs: Option<ClassDefs>,
        ) -> Self {
            let file = BufReader::new(File::open(filepath).expect("Cannot open COPC file"));
            let reader = CopcEntryReader::new(file).expect("Cannot create COPC reader");
            let header = reader.header();

            let copc_info = header.copc_info_vlr().expect("Not a COPC file");
            let (cx, cy, cz, halfsize, spacing) = (
                copc_info.center_x,
                copc_info.center_y,
                copc_info.center_z,
                copc_info.halfsize,
                copc_info.spacing,
            );

            let bounds = header.bounds();
            let global_min = [bounds.min.x, bounds.min.y, bounds.min.z];
            let global_max = [bounds.max.x, bounds.max.y, bounds.max.z];
            let viewer_origin = viewer_origin.unwrap_or(global_min);

            let hier_entries = read_full_hierarchy(&reader);

            let entries: Vec<CopcEntry> = hier_entries.iter()
                .map(|e| {
                    let (xmin, ymin, zmin, xmax, ymax, zmax) =
                        voxel_bounds(&e.key, cx, cy, cz, halfsize);
                    CopcEntry {
                        key: e.key,
                        offset: e.offset,
                        byte_size: e.byte_size,
                        point_count: e.point_count,
                        xmin, ymin, zmin, xmax, ymax, zmax,
                    }
                })
                .collect();

            let max_level = entries.iter().map(|e| e.key.l).max().unwrap_or(0);

            // Normalise `spacing` to the spec convention (root-level spacing,
            // halved at each octree level). The COPC spec says the field IS the
            // root spacing, but copc-converter writes the finest-level voxel
            // size instead (halfsize / 2^max_depth). Detect that case: if
            // scaling the written value up by 2^max_depth still fits inside
            // the root cube it must have been a fine-level value; otherwise
            // trust it as-is.
            let scaled = spacing * (1u64 << max_level) as f64;
            let root_spacing = if max_level > 0 && scaled <= halfsize * 2.0 {
                scaled
            } else {
                spacing
            };

            info!(
                "COPC: {} entries (max depth {}), spacing {:.4}m (root {:.2}m), \
                 bounds x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}",
                entries.len(), max_level, spacing, root_spacing,
                global_min[0], global_max[0],
                global_min[1], global_max[1],
                global_min[2], global_max[2],
            );

            let pipeline = create_pipeline(ctx);

            let (load_tx, load_rx) = std::sync::mpsc::channel::<CopcLoadReq>();
            let (result_tx, result_rx) = std::sync::mpsc::channel::<CopcLoadResult>();
            spawn_copc_loader_pool(filepath.to_string(), load_rx, result_tx, COPC_LOADER_THREADS);

            let mut stage = CopcStage {
                pipeline,
                filepath: filepath.to_string(),
                spacing: root_spacing,
                entries,
                loaded_nodes: HashMap::new(),
                global_min,
                global_max,
                viewer_origin,
                intensity_min: u16::MAX,
                intensity_max: 0,
                time_min: f64::MAX,
                time_max: f64::MIN,
                class_defs,
                rgb_override: None,
                visible: true,
                max_class: 0,
                color_h_min: 0.0,
                color_h_max: (global_max[2] - global_min[2]) as f32,
                z_clip: ZClip::OPEN,
                class_vis: ClassVis::ALL,
                in_flight: HashSet::new(),
                pinned: HashSet::new(),
                pin_box: None,
                edits: ClassEdits::load(filepath),
                load_tx,
                result_rx,
                last_view_change: std::time::Instant::now(),
                prev_view_proj: glam::Mat4::IDENTITY,
                generation: 0,
            };

            // Dispatch the root node immediately so the viewer shows something on startup.
            stage.dispatch_load(VoxelKey::ROOT);

            stage
        }

        pub fn height_range(&self) -> f32 {
            (self.global_max[2] - self.global_min[2]) as f32
        }

        /// How many nodes are dispatched to the loader but not yet on the GPU.
        pub fn pending_count(&self) -> usize {
            self.in_flight.len()
        }

        /// Send a load request to the background thread (no-op if already dispatched).
        fn dispatch_load(&mut self, key: VoxelKey) {
            self.dispatch_load_inner(key, false)
        }

        fn dispatch_load_inner(&mut self, key: VoxelKey, pinned: bool) {
            if self.in_flight.contains(&key) { return; }
            if let Some(entry) = self.entries.iter().find(|e| e.key == key) {
                let _ = self.load_tx.send(CopcLoadReq {
                    key,
                    offset: entry.offset,
                    byte_size: entry.byte_size,
                    point_count: entry.point_count,
                    generation: self.generation,
                    pinned,
                });
                self.in_flight.insert(key);
            }
        }

        /// Force-load every node whose octree cube meets the section corridor,
        /// at every level. COPC puts each point in exactly one node, so a
        /// corridor's full-resolution content is spread across all levels that
        /// overlap it — pinning only the deepest level would miss the points
        /// held by coarser ancestors.
        ///
        /// `start`/`end` are viewer-space XY, matching `section_start` /
        /// `section_end`. Returns `(node_count, point_count)`.
        pub fn pin_corridor(&mut self, start: (f32, f32), end: (f32, f32), depth: f32) -> (usize, i64) {
            let ax = start.0 as f64 + self.viewer_origin[0];
            let ay = start.1 as f64 + self.viewer_origin[1];
            let bx = end.0 as f64 + self.viewer_origin[0];
            let by = end.1 as f64 + self.viewer_origin[1];
            let depth = depth as f64;

            self.pinned.clear();
            self.pin_box = None;
            let mut points: i64 = 0;
            for e in &self.entries {
                if corridor_meets_aabb_2d(
                    (ax, ay), (bx, by), depth,
                    (e.xmin, e.ymin), (e.xmax, e.ymax),
                ) {
                    self.pinned.insert(e.key);
                    points += e.point_count as i64;
                }
            }
            info!(
                "COPC {}: pinned {} nodes ({} points) for section corridor",
                self.filepath, self.pinned.len(), points,
            );
            (self.pinned.len(), points)
        }

        /// Append every resident point inside the XY box and Z band, in
        /// absolute world coordinates. Used to gather evidence for a ground
        /// plane fit; the caller does the polygon test.
        ///
        /// Only resident nodes are searched. A ground plane is a low-order fit
        /// and the surface is dense, so the LOD sample defines it perfectly
        /// well — force-loading a whole plan-view footprint would be far more
        /// data than the fit needs.
        pub fn collect_points_in_bbox(
            &self,
            min: (f64, f64),
            max: (f64, f64),
            z_min: f64,
            z_max: f64,
            out: &mut Vec<(f64, f64, f64)>,
        ) {
            for node in self.loaded_nodes.values() {
                for p in &node.points {
                    if !self.class_vis.contains(effective_class(p)) { continue; }
                    if p.x >= min.0 && p.x <= max.0 && p.y >= min.1 && p.y <= max.1
                        && p.z >= z_min && p.z <= z_max {
                        out.push((p.x, p.y, p.z));
                    }
                }
            }
        }

        /// Drain finished loads and keep the pinned set fed, without touching
        /// LOD, the frustum, or the eviction pass.
        ///
        /// The section view never calls `update_view` — it has no 3D camera to
        /// drive it — so this is what lets a corridor stream in while the slice
        /// stays on screen, instead of bouncing the operator back to the 3D
        /// view to watch a progress bar.
        pub fn service_pins(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            while let Ok(result) = self.result_rx.try_recv() {
                self.in_flight.remove(&result.key);
                if !result.pinned && result.generation != self.generation { continue; }
                if !self.loaded_nodes.contains_key(&result.key) {
                    self.ingest_node(ctx, result.key, result.points, color_mode);
                }
            }
            if self.pinned.is_empty() { return; }
            let mut want: Vec<VoxelKey> = self.pinned.iter()
                .filter(|k| !self.loaded_nodes.contains_key(k) && !self.in_flight.contains(k))
                .copied()
                .collect();
            want.sort_by_key(|k| k.l);
            let room = (COPC_LOADER_THREADS * 2).saturating_sub(self.in_flight.len());
            for key in want.into_iter().take(room) {
                self.dispatch_load_inner(key, true);
            }
        }

        /// Release the corridor or neighbourhood pins, letting normal LOD resume.
        pub fn clear_pins(&mut self) {
            self.pinned.clear();
            self.pin_box = None;
        }

        /// Pinned nodes not yet resident. Zero means the corridor is fully loaded.
        pub fn pins_pending(&self) -> usize {
            self.pinned.iter().filter(|k| !self.loaded_nodes.contains_key(k)).count()
        }

        pub fn pinned_count(&self) -> usize {
            self.pinned.len()
        }

        /// Drop every resident node so it is re-read from disk. Used after the
        /// op list changes in a way that cannot be applied incrementally — an
        /// undo, since an op does not record the classes it overwrote.
        pub fn invalidate_loaded(&mut self, ctx: &mut dyn RenderingBackend) {
            let keys: Vec<VoxelKey> = self.loaded_nodes.keys().copied().collect();
            for key in keys {
                if let Some(node) = self.loaded_nodes.remove(&key) {
                    for chunk in node.chunks {
                        free_chunk(ctx, chunk);
                    }
                }
            }
            self.in_flight.clear();
            self.generation = self.generation.wrapping_add(1);
        }

        /// Apply the current op list to the points already resident, and push
        /// the resulting colours to the GPU. Cheap counterpart to
        /// `invalidate_loaded` for the append-an-op case, where re-reading from
        /// disk would be wasted work.
        pub fn apply_edits_to_loaded(
            &mut self,
            ctx: &mut dyn RenderingBackend,
            color_mode: ColorMode,
            from_idx: usize,
        ) -> usize {
            let mut changed = 0;
            let keys: Vec<VoxelKey> = self.loaded_nodes.keys().copied().collect();
            for key in keys {
                let Some(node) = self.loaded_nodes.get_mut(&key) else { continue };
                let mut nmin = [f64::MAX; 3];
                let mut nmax = [f64::MIN; 3];
                for p in &node.points {
                    for (i, v) in [p.x, p.y, p.z].iter().enumerate() {
                        nmin[i] = nmin[i].min(*v);
                        nmax[i] = nmax[i].max(*v);
                    }
                }
                changed += self.edits.apply_from(from_idx, &mut node.points, &nmin, &nmax);
            }
            for node in self.loaded_nodes.values() {
                for p in &node.points {
                    let id: u8 = if p.is_overlap { 12 } else { p.classification.into() };
                    self.max_class = self.max_class.max(id);
                }
            }
            if changed > 0 {
                self.rebuild_colors(ctx, color_mode);
            }
            changed
        }

        /// Convert loaded points into GPU chunks and store them.
        fn ingest_node(
            &mut self,
            ctx: &mut dyn RenderingBackend,
            key: VoxelKey,
            mut pts: Vec<Point>,
            color_mode: ColorMode,
        ) {
            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);

            // Manual classification is applied here, on the way in, rather than
            // to whatever happened to be on screen when the operator drew it.
            // A node arriving for the first time long after the edit was made
            // gets exactly the same treatment as one that was already resident.
            if !self.edits.is_empty() && !pts.is_empty() {
                let mut nmin = [f64::MAX; 3];
                let mut nmax = [f64::MIN; 3];
                for p in &pts {
                    for (i, v) in [p.x, p.y, p.z].iter().enumerate() {
                        nmin[i] = nmin[i].min(*v);
                        nmax[i] = nmax[i].max(*v);
                    }
                }
                self.edits.apply_to_node(&mut pts, &nmin, &nmax);
            }

            for p in &pts {
                self.intensity_min = self.intensity_min.min(p.intensity);
                self.intensity_max = self.intensity_max.max(p.intensity);

                let time = p.gps_time.unwrap_or(0.);
                self.time_min = self.time_min.min(time);
                self.time_max = self.time_max.max(time);

                let class_id: u8 = if p.is_overlap { 12 } else { p.classification.into() };
                self.max_class = self.max_class.max(class_id);
            }

            // Ranges must be read after the accumulation above, otherwise this
            // node is coloured against the bounds it just widened.
            let intensity_range = if self.intensity_max > self.intensity_min {
                (self.intensity_max - self.intensity_min) as f32
            } else {
                65535.0
            };
            let time_range = if self.time_max > self.time_min {
                self.time_max - self.time_min
            }
            else {
                1.0
            };

            let positions: Vec<[f32; 3]> = pts.iter().map(|p| [
                (p.x - self.viewer_origin[0]) as f32,
                (p.y - self.viewer_origin[1]) as f32,
                (p.z - self.viewer_origin[2]) as f32,
            ]).collect();

            let colors: Vec<[f32; 3]> = pts.iter().enumerate().map(|(i, p)| {
                color_for_point(
                    p, positions[i][2], h_lo, h_hi,
                    self.intensity_min, intensity_range, p.gps_time.unwrap_or(0.), self.time_min, time_range,
                    color_mode, &self.class_defs, self.rgb_override, self.class_vis,
                )
            }).collect();

            let total_points = colors.len() as i32;
            let chunks = build_chunks(ctx, &positions, &colors);
            let index = SpatialGrid::build(&positions);

            self.loaded_nodes.insert(key, CopcLoadedNode { chunks, positions, points: pts, total_points, index });
        }

        /// Called every frame. Drains completed results, then dispatches new requests.
        /// `already_loaded` is the total point count already loaded by all other stages
        /// that share the same global point budget.
        pub fn update_view(
            &mut self,
            ctx: &mut dyn RenderingBackend,
            view_proj: &glam::Mat4,
            ortho_scale: f32,
            screen_h: f32,
            color_mode: ColorMode,
            already_loaded: i32,
        ) {
            const GLOBAL_POINT_BUDGET: i32 = 20_000_000;
            // Detect view changes: cancel all queued loads and bump the generation so
            // any results that were already in-flight get discarded when they arrive.
            if *view_proj != self.prev_view_proj {
                self.prev_view_proj = *view_proj;
                self.last_view_change = std::time::Instant::now();
                self.generation = self.generation.wrapping_add(1);
                // Pinned loads survive a view change; only speculative LOD
                // requests are abandoned.
                self.in_flight.retain(|k| self.pinned.contains(k));
            }
            const SETTLE_MS: u64 = 300;
            let view_settled = self.last_view_change.elapsed()
                >= std::time::Duration::from_millis(SETTLE_MS);

            // Drain all results that the background thread has finished.
            // Discard results that belong to a previous generation (view has moved on).
            // GPU upload (ingest_node) is cheap — just vertex buffer creation.
            while let Ok(result) = self.result_rx.try_recv() {
                self.in_flight.remove(&result.key);
                if !result.pinned && result.generation != self.generation { continue; }
                // Only ingest if not already loaded (might have been unloaded mid-flight).
                if !self.loaded_nodes.contains_key(&result.key) {
                    //let n = result.points.len();
                    self.ingest_node(ctx, result.key, result.points, color_mode);
                    // println!("COPC ingested ({},{},{},{}) — {} pts, {} in-flight",
                    //     result.key.l, result.key.x, result.key.y, result.key.z,
                    //     n, self.in_flight.len());
                }
            }

            // Meters per screen pixel determines the required LOD.
            let mpp = (ortho_scale * 2.0 / screen_h.max(1.0)) as f64;
            let target_depth = (if mpp > 0.0 && self.spacing > 0.0 {
                 (self.spacing / mpp).log2().ceil() as i32
             } else {
                 0
            }).clamp(0, 20);
            let planes = extract_frustum_planes(view_proj);

            let desired: HashSet<VoxelKey> = self.entries.iter()
                .filter(|e| e.key.l <= target_depth)
                .filter(|e| {
                    let vmin = [
                        (e.xmin - self.viewer_origin[0]) as f32,
                        (e.ymin - self.viewer_origin[1]) as f32,
                        (e.zmin - self.viewer_origin[2]) as f32,
                    ];
                    let vmax = [
                        (e.xmax - self.viewer_origin[0]) as f32,
                        (e.ymax - self.viewer_origin[1]) as f32,
                        (e.zmax - self.viewer_origin[2]) as f32,
                    ];
                    !aabb_outside_frustum(&planes, &vmin, &vmax)
                })
                .map(|e| e.key)
                .collect();
            // Corridor pins are held on top of whatever the LOD wants, and are
            // never evicted while the section is open.
            let mut desired = desired;
            desired.extend(self.pinned.iter().copied());

            // Unload nodes that are no longer desired and not in flight.
            let to_unload: Vec<VoxelKey> = self.loaded_nodes.keys()
                .filter(|k| !desired.contains(k))
                .copied()
                .collect();
            for key in to_unload {
                if let Some(node) = self.loaded_nodes.remove(&key) {
                    for chunk in node.chunks {
                        free_chunk(ctx, chunk);
                    }
                }
            }

            // Dispatch loads for desired nodes that are not loaded and not in flight.
            // Only dispatch once the view has settled (no changes for SETTLE_MS).
            // Sort by NDC distance from screen centre so the most visible nodes load first.
            // Cap the queue depth so we never flood the channel on a sudden zoom-in.
            // Also respect the global point budget shared across all COPC stages.
            // Keep the pool fed: 2× the worker count means each thread always
            // has a follow-up node queued when it finishes its current one.
            let max_in_flight = COPC_LOADER_THREADS * 2;

            // Pinned corridor nodes are dispatched ahead of everything else and
            // ignore both the settle delay and the global point budget: the
            // operator is waiting on exactly these, and a partial corridor is
            // worse than a slow one.
            if !self.pinned.is_empty() {
                let mut want: Vec<VoxelKey> = self.pinned.iter()
                    .filter(|k| !self.loaded_nodes.contains_key(k) && !self.in_flight.contains(k))
                    .copied()
                    .collect();
                want.sort_by_key(|k| k.l);
                let room = max_in_flight.saturating_sub(self.in_flight.len());
                for key in want.into_iter().take(room) {
                    self.dispatch_load_inner(key, true);
                }
            }

            let budget = max_in_flight.saturating_sub(self.in_flight.len());
            let points_remaining = GLOBAL_POINT_BUDGET - already_loaded - self.total_loaded_points();
            if view_settled && budget > 0 && points_remaining > 0 {
                // Build a lookup from key → entry for centre-distance sorting.
                let entry_map: HashMap<VoxelKey, &CopcEntry> =
                    self.entries.iter().map(|e| (e.key, e)).collect();

                let mut to_dispatch: Vec<VoxelKey> = desired.iter()
                    .filter(|k| !self.loaded_nodes.contains_key(k) && !self.in_flight.contains(k))
                    .copied()
                    .collect();

                // Project each voxel's world-space centre through view_proj and measure
                // its NDC distance from the screen centre (0,0).  Coarser levels are
                // given a small bonus so LOD still progresses coarse→fine overall.
                to_dispatch.sort_by(|a, b| {
                    let ndc_dist = |key: &VoxelKey| -> f32 {
                        if let Some(e) = entry_map.get(key) {
                            let cx = ((e.xmin + e.xmax) * 0.5 - self.viewer_origin[0]) as f32;
                            let cy = ((e.ymin + e.ymax) * 0.5 - self.viewer_origin[1]) as f32;
                            let cz = ((e.zmin + e.zmax) * 0.5 - self.viewer_origin[2]) as f32;
                            let p = view_proj.project_point3(glam::Vec3::new(cx, cy, cz));
                            // Combine screen distance with a per-level bias so coarse
                            // nodes are preferred when they are roughly as close.
                            p.x * p.x + p.y * p.y + key.l as f32 * 0.1
                        } else {
                            f32::MAX
                        }
                    };
                    ndc_dist(a).partial_cmp(&ndc_dist(b)).unwrap_or(std::cmp::Ordering::Equal)
                });

                for key in to_dispatch.into_iter().take(budget) {
                    self.dispatch_load(key);
                }
            }
        }

        pub fn draw_visible(
            &self,
            ctx: &mut dyn RenderingBackend,
            view_proj: &glam::Mat4,
            point_size: f32,
        ) -> i32 {
            if !self.visible { return 0; }
            let planes = extract_frustum_planes(view_proj);
            let height_range = self.height_range();
            let mut drawn = 0;

            ctx.apply_pipeline(&self.pipeline);
            for node in self.loaded_nodes.values() {
                for chunk in &node.chunks {
                    if aabb_outside_frustum(&planes, &chunk.aabb_min, &chunk.aabb_max) {
                        continue;
                    }
                    ctx.apply_bindings(&chunk.bindings);
                    ctx.apply_uniforms(miniquad::UniformsSource::table(
                        &shader::Uniforms { mvp: *view_proj, point_size, height_range,
                        z_clip_min: self.z_clip.min, z_clip_max: self.z_clip.max },
                    ));
                    ctx.draw(0, chunk.num_points, 1);
                    drawn += chunk.num_points;
                }
            }
            drawn
        }

        pub fn total_loaded_points(&self) -> i32 {
            self.loaded_nodes.values().map(|n| n.total_points).sum()
        }

        /// Free all GPU buffers held by this stage and shut down the loader pool.
        /// Consumes the stage; the loader threads exit when `load_tx` drops and
        /// their `recv()` returns `Err`.
        pub fn unload(mut self, ctx: &mut dyn RenderingBackend) {
            for (_, node) in self.loaded_nodes.drain() {
                for chunk in node.chunks {
                    free_chunk(ctx, chunk);
                }
            }
            // `self` drops here, which drops `load_tx`, signalling workers to exit.
        }

        pub fn loaded_node_count(&self) -> usize {
            self.loaded_nodes.len()
        }

        /// Returns `(point, viewer_pos, ndc_dist_sq)` for the loaded point closest
        /// to the given screen coordinate.  The distance is returned so callers with
        /// multiple stages can pick the globally nearest point.
        pub fn find_nearest_point(
            &self,
            view_proj: &glam::Mat4,
            screen_x: f32, screen_y: f32,
            screen_w: f32, screen_h: f32,
        ) -> Option<(&Point, glam::Vec3, f32)> {
            let ndc_x = screen_x / screen_w * 2.0 - 1.0;
            let ndc_y = 1.0 - screen_y / screen_h * 2.0;
            let mut best_dist = f32::MAX;
            let mut best: Option<(&Point, glam::Vec3, f32)> = None;
            for node in self.loaded_nodes.values() {
                for (i, pos) in node.positions.iter().enumerate() {
                    if !self.z_clip.contains(pos[2]) { continue; }
                    if !self.class_vis.contains(effective_class(&node.points[i])) { continue; }
                    let p = *view_proj * glam::Vec4::new(pos[0], pos[1], pos[2], 1.0);
                    let px = p.x / p.w;
                    let py = p.y / p.w;
                    let dist = (px - ndc_x) * (px - ndc_x) + (py - ndc_y) * (py - ndc_y);
                    if dist < best_dist {
                        best_dist = dist;
                        best = Some((&node.points[i], glam::Vec3::from(*pos), dist));
                    }
                }
            }
            best
        }

        /// Frontmost point within `max_dist` of a viewer-space ray, across
        /// every resident node. Returns the hit and the LAS point behind it.
        ///
        /// Each node answers over its own grid and the frontmost answer wins.
        /// COPC puts every point in exactly one node at exactly one level, so a
        /// surface under the cursor is generally spread across several resident
        /// nodes — comparing `t` across them is not an optimisation, it is what
        /// makes the answer right.
        #[allow(dead_code)]
        pub fn snap_nearest_ray(
            &self,
            origin: [f32; 3],
            dir: [f32; 3],
            max_dist: f32,
        ) -> Option<(RayHit, &Point)> {
            let mut best: Option<(RayHit, &Point)> = None;
            for node in self.loaded_nodes.values() {
                let z_clip = self.z_clip;
                let class_vis = self.class_vis;
                let positions = &node.positions;
                let points = &node.points;
                let hit = node.index.nearest_ray(
                    positions, origin, dir, max_dist,
                    move |i: u32| z_clip.contains(positions[i as usize][2])
                        && class_vis.contains(effective_class(&points[i as usize])),
                );
                if let Some(h) = hit
                    && best.as_ref().is_none_or(|(b, _)| h.t < b.t) {
                    best = Some((h, &node.points[h.idx as usize]));
                }
            }
            best
        }

        /// Append every visible point within `r` metres of a viewer-space XY
        /// position as `(z, position)`. See `Stage::snap_z_candidates`.
        ///
        /// Read [`CopcStage::lod_spacing_near`] before ranking these: at a
        /// coarse LOD they are a thinned sample and any percentile over them is
        /// quietly wrong by decimetres.
        #[allow(dead_code)]
        pub fn snap_z_candidates(&self, x: f32, y: f32, r: f32, out: &mut Vec<(f32, [f32; 3])>) {
            let mut buf: Vec<u32> = Vec::new();
            for node in self.loaded_nodes.values() {
                if !node.index.circle_overlaps(x, y, r) { continue; }
                buf.clear();
                let z_clip = self.z_clip;
                let class_vis = self.class_vis;
                let positions = &node.positions;
                let points = &node.points;
                node.index.points_in_radius(
                    positions, x, y, r,
                    move |i: u32| z_clip.contains(positions[i as usize][2])
                        && class_vis.contains(effective_class(&points[i as usize])),
                    &mut buf,
                );
                out.extend(buf.iter().map(|&i| {
                    let p = node.positions[i as usize];
                    (p[2], p)
                }));
            }
        }

        /// Point spacing, in metres, of the finest resident node near a
        /// viewer-space XY position — or `None` when nothing is resident there.
        ///
        /// This is the spacing guard. Zoomed out, the resident set is a thinned
        /// LOD sample: "5th percentile of Z within 0.2 m" would then be taken
        /// over whatever sparse subset happened to be loaded, and be wrong by
        /// decimetres without anything looking wrong. Comparing this against
        /// the snap radius is how the tool knows to refuse rather than to
        /// guess. It is measured *near the cursor* rather than over the whole
        /// resident set, because the finest node loaded somewhere off screen
        /// says nothing about the data under the crosshair.
        #[allow(dead_code)]
        pub fn lod_spacing_near(&self, x: f32, y: f32, r: f32) -> Option<f64> {
            let deepest = self.loaded_nodes.iter()
                .filter(|(_, node)| node.index.circle_overlaps(x, y, r))
                .map(|(key, _)| key.l)
                .max()?;
            Some(self.spacing / 2f64.powi(deepest))
        }

        /// Force-load every node whose octree cube meets a box of half-extent
        /// `half` metres around a viewer-space XY position, so that snapping
        /// under the cursor sees full-resolution data rather than the LOD
        /// sample. The digitising counterpart of [`CopcStage::pin_corridor`],
        /// and like it the sole owner of `pinned`: a call replaces whatever was
        /// pinned before, so the two must not be in force at once.
        ///
        /// Returns `(node_count, point_count)`. When the box would pull in more
        /// than [`SNAP_PIN_POINT_LIMIT`] points nothing is pinned and
        /// `(0, point_count)` comes back, so the caller can say why rather than
        /// letting a dense cloud page itself into memory behind a moving
        /// cursor.
        ///
        /// Cheap to call every frame: the pin set is recomputed only once the
        /// cursor has left the middle of the box it was built for. Without that
        /// hysteresis this would rescan the whole hierarchy on every mouse
        /// move, and re-dispatch loads for nodes that were already pinned.
        #[allow(dead_code)]
        pub fn pin_neighbourhood(&mut self, x: f32, y: f32, half: f64) -> (usize, i64) {
            let cx = x as f64 + self.viewer_origin[0];
            let cy = y as f64 + self.viewer_origin[1];

            if let Some(b) = self.pin_box
                && (b.half - half).abs() < 1e-6
                && (cx - b.cx).abs() < half * 0.5
                && (cy - b.cy).abs() < half * 0.5 {
                return (self.pinned.len(), b.points);
            }

            let mut keys = Vec::new();
            let mut points: i64 = 0;
            for e in &self.entries {
                if e.xmax >= cx - half && e.xmin <= cx + half
                    && e.ymax >= cy - half && e.ymin <= cy + half {
                    keys.push(e.key);
                    points += e.point_count as i64;
                }
            }
            if points > SNAP_PIN_POINT_LIMIT {
                self.pinned.clear();
                self.pin_box = None;
                return (0, points);
            }
            self.pinned = keys.into_iter().collect();
            self.pin_box = Some(PinBox { cx, cy, half, points });
            info!(
                "COPC {}: pinned {} nodes ({} points) for a {:.0}m snap neighbourhood",
                self.filepath, self.pinned.len(), points, half * 2.0,
            );
            (self.pinned.len(), points)
        }

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);
            let intensity_range = if self.intensity_max > self.intensity_min {
                (self.intensity_max - self.intensity_min) as f32
            } else {
                65535.0
            };
            let time_range = if self.time_max > self.time_min {
                self.time_max - self.time_min
            }
            else {
                1.0
            };
            for node in self.loaded_nodes.values_mut() {
                for chunk in &mut node.chunks {
                    let colors: Vec<[f32; 3]> = chunk.point_indices.iter().map(|&i| {
                        color_for_point(
                            &node.points[i], node.positions[i][2], h_lo, h_hi,
                            self.intensity_min, intensity_range, node.points[i].gps_time.unwrap_or(0.), self.time_min, time_range,
                            color_mode, &self.class_defs, self.rgb_override, self.class_vis,
                        )
                    }).collect();
                    ctx.buffer_update(chunk.bindings.vertex_buffers[1], BufferSource::slice(&colors));
                }
            }
        }

        /// Collect section vertices from currently loaded nodes.
        /// Returns `(verts, length, height_min, height_max)`.
        #[allow(clippy::type_complexity)]
        pub fn extract_section_verts(
            &self,
            start: (f32, f32),
            end: (f32, f32),
            depth: f32,
            color_mode: ColorMode,
        ) -> (Vec<Vertex>, Vec<(Point, f32, f32)>, f32, f32, f32) {
            let dx = end.0 - start.0;
            let dy = end.1 - start.1;
            let len = (dx * dx + dy * dy).sqrt();
            let dir = (dx / len, dy / len);
            let normal = (-dir.1, dir.0);

            let (h_lo, h_hi) = (self.color_h_min, self.color_h_max);
            let intensity_range = if self.intensity_max > self.intensity_min {
                (self.intensity_max - self.intensity_min) as f32
            } else {
                65535.0
            };
            let time_range = if self.time_max > self.time_min {
                self.time_max - self.time_min
            }
            else {
                1.0
            };

            let mut verts = Vec::new();
            let mut raw: Vec<(Point, f32, f32)> = Vec::new();
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;

            for node in self.loaded_nodes.values() {
                for (i, pos) in node.positions.iter().enumerate() {
                    let rel_x = pos[0] - start.0;
                    let rel_y = pos[1] - start.1;
                    let along = rel_x * dir.0 + rel_y * dir.1;
                    let perp = rel_x * normal.0 + rel_y * normal.1;
                    if along >= 0.0 && along <= len && perp.abs() <= depth
                        && self.z_clip.contains(pos[2])
                        && self.class_vis.contains(effective_class(&node.points[i])) {
                        h_min = h_min.min(pos[2]);
                        h_max = h_max.max(pos[2]);
                        raw.push((node.points[i].clone(), along, pos[2]));
                        verts.push(Vertex {
                            pos: [along, pos[2], 0.0],
                            color: color_for_point(
                                &node.points[i], pos[2], h_lo, h_hi,
                                self.intensity_min, intensity_range, node.points[i].gps_time.unwrap_or(0.), self.time_min, time_range,
                                color_mode, &self.class_defs, self.rgb_override, self.class_vis,
                            ),
                        });
                    }
                }
            }

            let h_min = if h_min == f32::MAX { 0.0 } else { h_min };
            let h_max = if h_max == f32::MIN { 0.0 } else { h_max };
            (verts, raw, len, h_min, h_max)
        }

        /*pub fn extract_section(
            &self,
            ctx: &mut dyn RenderingBackend,
            start: (f32, f32),
            end: (f32, f32),
            depth: f32,
            color_mode: ColorMode,
        ) -> SectionData {
            let (verts, len, h_min, h_max) = self.extract_section_verts(start, end, depth, color_mode);
            info!("COPC Section: {:.1}m long, {:.1}m depth, {} points", len, depth, verts.len());
            make_section_data(ctx, verts, len, h_min, h_max, self.height_range())
        }*/

        pub fn draw_section(
            &self,
            ctx: &mut dyn RenderingBackend,
            section: &SectionData,
            mvp: &glam::Mat4,
            point_size: f32,
        ) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp, point_size, height_range: section.height_range,
                    z_clip_min: ZClip::OPEN.min, z_clip_max: ZClip::OPEN.max },
            ));
            ctx.draw(0, section.num_points, 1);
        }
    }

    /// Build a `SectionData` GPU object from raw vertices.
    pub fn make_section_data(
        ctx: &mut dyn RenderingBackend,
        verts: Vec<Vertex>,
        length: f32,
        height_min: f32,
        height_max: f32,
        height_range: f32,
    ) -> SectionData {
        let num_points = verts.len() as i32;
        let (pos_buffer, color_buffer) = upload_split_verts(ctx, &verts);
        let indices: Vec<u32> = (0..num_points as u32).collect();
        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );
        SectionData {
            bindings: Bindings {
                vertex_buffers: vec![pos_buffer, color_buffer],
                index_buffer,
                images: vec![],
            },
            num_points,
            length,
            height_min,
            height_max,
            height_range,
            points: verts.iter().map(|v| v.pos).collect(),
            index: std::cell::OnceCell::new(),
        }
    }

    /// Build one combined `SectionData` from all manually-loaded raw `Stage`s,
    /// so a cross section shows every loaded cloud, not just the first.
    pub fn build_section_from_stages(
        ctx: &mut dyn RenderingBackend,
        stages: &[Stage],
        start: (f32, f32),
        end: (f32, f32),
        depth: f32,
        color_mode: ColorMode,
    ) -> SectionData {
        let height_range = stages.first().map(|s| s.height_range()).unwrap_or(1.0);
        let mut all_verts = Vec::new();
        let (mut h_min, mut h_max) = (f32::MAX, f32::MIN);
        let mut length = 0.0_f32;
        for s in stages {
            if !s.visible { continue; }
            let (verts, len, hmin, hmax) = s.extract_section_verts(start, end, depth, color_mode);
            all_verts.extend(verts);
            h_min = h_min.min(hmin);
            h_max = h_max.max(hmax);
            length = length.max(len);
        }
        let h_min = if h_min == f32::MAX { 0.0 } else { h_min };
        let h_max = if h_max == f32::MIN { 0.0 } else { h_max };
        info!("Section: {:.1}m long, {:.1}m depth, {} points, height {:.1}..{:.1}m",
            length, depth, all_verts.len(), h_min, h_max);
        make_section_data(ctx, all_verts, length, h_min, h_max, height_range)
    }

    /// Rebuild a combined COPC `SectionData` from cached raw point data with a new colour mode.
    /// `per_stage_raw[i]` contains `(Point, along, pos_z)` triples extracted from `stages[i]`.
    pub fn build_section_from_raw_stages(
        ctx: &mut dyn RenderingBackend,
        stages: &[CopcStage],
        per_stage_raw: &[Vec<(Point, f32, f32)>],
        length: f32,
        h_min: f32,
        h_max: f32,
        color_mode: ColorMode,
    ) -> SectionData {
        let height_range = stages.first().map(|s| s.height_range()).unwrap_or(1.0);
        let mut all_verts = Vec::new();
        for (stage_idx, raw) in per_stage_raw.iter().enumerate() {
            if let Some(cs) = stages.get(stage_idx) {
                let intensity_range = if cs.intensity_max > cs.intensity_min {
                    (cs.intensity_max - cs.intensity_min) as f32
                } else {
                    65535.0
                };
                let time_range = if cs.time_max > cs.time_min {
                    cs.time_max - cs.time_min
                }
                else {
                    1.0
                };
                for (pt, along, pos_z) in raw {
                    all_verts.push(Vertex {
                        pos: [*along, *pos_z, 0.0],
                        color: color_for_point(
                            pt, *pos_z, cs.color_h_min, cs.color_h_max,
                            cs.intensity_min, intensity_range, pt.gps_time.unwrap_or(0.), cs.time_min, time_range,
                            color_mode, &cs.class_defs, cs.rgb_override, cs.class_vis,
                        ),
                    });
                }
            }
        }
        make_section_data(ctx, all_verts, length, h_min, h_max, height_range)
    }

    pub mod shader {
        use miniquad::*;

        pub const VERTEX: &str = r#"#version 100
            attribute vec3 pos;
            attribute vec3 color;

            uniform mat4 mvp;
            uniform float point_size;
            uniform float height_range;
            // Height band to draw, in the same space as pos.z. Points outside
            // it are pushed past the far plane so the rasteriser drops them —
            // GLSL ES 100 has no vertex-stage discard, and culling here costs
            // nothing per frame, where re-uploading filtered buffers would.
            uniform float z_clip_min;
            uniform float z_clip_max;

            varying lowp vec3 v_color;
            varying mediump float v_depth;

            void main() {
                // A negative red is the sentinel a hidden class is coloured
                // with (see HIDDEN_COLOR): no real colour can be negative, so
                // no uniform and no extra attribute are needed to carry
                // visibility. Culled the same way as the height band.
                if (color.r < 0.0 || pos.z < z_clip_min || pos.z > z_clip_max) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                    gl_PointSize = 0.0;
                    v_color = color;
                    v_depth = 1.0;
                    return;
                }
                gl_Position = mvp * vec4(pos, 1.0);
                gl_PointSize = point_size;
                v_color = color;
                // Use model-space Z (height) as depth signal for EDL.
                // height_range gives full [0.002, 1.0] coverage so depth
                // differences between neighbors are large and EDL is visible.
                mediump float t = (height_range > 0.0) ? (pos.z / height_range) : 0.5;
                v_depth = t * 0.998 + 0.002;
            }"#;

        pub const FRAGMENT: &str = r#"#version 100
            varying lowp vec3 v_color;
            varying mediump float v_depth;

            void main() {
                gl_FragColor = vec4(v_color, v_depth);
            }"#;

        pub fn meta() -> ShaderMeta {
            ShaderMeta {
                images: vec![],
                uniforms: UniformBlockLayout {
                    uniforms: vec![
                        UniformDesc::new("mvp", UniformType::Mat4),
                        UniformDesc::new("point_size", UniformType::Float1),
                        UniformDesc::new("height_range", UniformType::Float1),
                        UniformDesc::new("z_clip_min", UniformType::Float1),
                        UniformDesc::new("z_clip_max", UniformType::Float1),
                    ],
                },
            }
        }

        #[repr(C)]
        pub struct Uniforms {
            pub mvp: glam::Mat4,
            pub point_size: f32,
            pub height_range: f32,
            pub z_clip_min: f32,
            pub z_clip_max: f32,
        }
    }
//}
