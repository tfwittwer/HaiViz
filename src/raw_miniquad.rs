//pub mod raw_miniquad {
    use std::collections::{HashMap, HashSet};
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    use las::{Point, Reader};
    use las::copc::{CopcEntryReader, VoxelKey};
    use miniquad::*;

    use crate::ColorMode;

    const CHUNK_SIZE: f32 = 50.0;

    /// Custom class definitions: id -> (label, [r,g,b])
    pub type ClassDefs = HashMap<u8, (String, [f32; 3])>;

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
            println!("Loaded {} class definitions from classes.txt", defs.len());
            Some(defs)
        }
    }

    /// Point cloud backed by las::Point for full attribute access.
    pub struct VizPointcloud {
        pub points: Vec<Point>,
        pub min: [f64; 3],
        pub max: [f64; 3],
        pub intensity_min: u16,
        pub intensity_max: u16,
    }

    impl VizPointcloud {
        pub fn load(filename: &str) -> VizPointcloud {
            let mut reader = Reader::from_path(filename).unwrap();
            let n = reader.header().number_of_points() as usize;

            let mut points = Vec::with_capacity(n);
            reader.read_all_points_into(&mut points).unwrap();

            let mut min = [f64::MAX; 3];
            let mut max = [f64::MIN; 3];
            let mut intensity_min = u16::MAX;
            let mut intensity_max = u16::MIN;

            for point in &points {
                min[0] = min[0].min(point.x);
                min[1] = min[1].min(point.y);
                min[2] = min[2].min(point.z);
                max[0] = max[0].max(point.x);
                max[1] = max[1].max(point.y);
                max[2] = max[2].max(point.z);
                intensity_min = intensity_min.min(point.intensity);
                intensity_max = intensity_max.max(point.intensity);
            }

            println!("{} points loaded from {}", points.len(), filename);
            println!("Intensity range: {}..{}", intensity_min, intensity_max);
            VizPointcloud { points, min, max, intensity_min, intensity_max }
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

    pub struct SectionData {
        bindings: Bindings,
        pub num_points: i32,
        pub length: f32,
        pub height_min: f32,
        pub height_max: f32,
        /// Full Z span of the source dataset — used by the shader for height-based colouring.
        pub height_range: f32,
    }

    pub struct Stage {
        pipeline: Pipeline,
        pub chunks: Vec<Chunk>,
        pub total_points: i32,
        pub cloud: VizPointcloud,
        pub positions: Vec<[f32; 3]>,
        pub class_defs: Option<ClassDefs>,
    }

    fn color_for_point(
        point: &Point, pos_z: f32, height_range: f32,
        intensity_min: u16, intensity_range: f32,
        mode: ColorMode, class_defs: &Option<ClassDefs>,
    ) -> [f32; 3] {
        match mode {
            ColorMode::Height => {
                let h = if height_range > 0.0 { (pos_z / height_range).clamp(0.0, 1.0) } else { 0.5 };
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
                let class_id: u8 = point.classification.into();
                if let Some(defs) = class_defs &&
                    let Some((_label, color)) = defs.get(&class_id) {
                        return *color;
                    
                }
                classification_color(class_id)
            }
            ColorMode::Rgb => {
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

    fn classification_color(class: u8) -> [f32; 3] {
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

        ctx.new_pipeline(
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("pos", VertexFormat::Float3),
                VertexAttribute::new("color", VertexFormat::Float3),
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

    fn build_chunks(
        ctx: &mut dyn RenderingBackend,
        positions: &[[f32; 3]],
        vertices: &[Vertex],
    ) -> Vec<Chunk> {
        let mut chunk_map: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (i, pos) in positions.iter().enumerate() {
            let cx = (pos[0] / CHUNK_SIZE).floor() as i32;
            let cy = (pos[1] / CHUNK_SIZE).floor() as i32;
            chunk_map.entry((cx, cy)).or_default().push(i);
        }

        chunk_map.into_values()
            .map(|point_indices| {
                let chunk_verts: Vec<Vertex> =
                point_indices.iter().map(|&i| vertices[i]).collect();
                let mut aabb_min = [f32::MAX; 3];
                let mut aabb_max = [f32::MIN; 3];
                for v in &chunk_verts {
                    for j in 0..3 {
                        aabb_min[j] = aabb_min[j].min(v.pos[j]);
                        aabb_max[j] = aabb_max[j].max(v.pos[j]);
                    }
                }

                let n = chunk_verts.len() as i32;
                let vertex_buffer = ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Dynamic,
                    BufferSource::slice(&chunk_verts),
                );
                let indices: Vec<u32> = (0..n as u32).collect();
                let index_buffer = ctx.new_buffer(
                    BufferType::IndexBuffer,
                    BufferUsage::Immutable,
                    BufferSource::slice(&indices),
                );

                Chunk {
                    bindings: Bindings {
                        vertex_buffers: vec![vertex_buffer],
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
        pub fn new(ctx: &mut dyn RenderingBackend, filename: &str, color_mode: ColorMode, class_defs: Option<ClassDefs>) -> Stage {
            let cloud = VizPointcloud::load(filename);

            let height_range = (cloud.max[2] - cloud.min[2]) as f32;
            let intensity_range = (cloud.intensity_max - cloud.intensity_min) as f32;

            let positions: Vec<[f32; 3]> = cloud
                .points
                .iter()
                .map(|p| [
                    (p.x - cloud.min[0]) as f32,
                    (p.y - cloud.min[1]) as f32,
                    (p.z - cloud.min[2]) as f32,
                ])
                .collect();

            let vertices: Vec<Vertex> = cloud
                .points
                .iter()
                .enumerate()
                .map(|(i, p)| Vertex {
                    pos: positions[i],
                    color: color_for_point(p, positions[i][2], height_range, cloud.intensity_min, intensity_range, color_mode, &class_defs),
                })
                .collect();

            let pipeline = create_pipeline(ctx);
            let total_points = vertices.len() as i32;
            let chunks = build_chunks(ctx, &positions, &vertices);

            println!("{} points in {} chunks ({}m cells), height range: {:.1}m",
                total_points, chunks.len(), CHUNK_SIZE, height_range);

            Stage {
                pipeline,
                chunks,
                total_points,
                cloud,
                positions,
                class_defs,
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

        pub fn height_range(&self) -> f32 {
            (self.cloud.max[2] - self.cloud.min[2]) as f32
        }

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let height_range = self.height_range();
            let intensity_range = (self.cloud.intensity_max - self.cloud.intensity_min) as f32;

            for chunk in &mut self.chunks {
                let chunk_verts: Vec<Vertex> = chunk
                    .point_indices
                    .iter()
                    .map(|&i| {
                        let p = &self.cloud.points[i];
                        Vertex {
                            pos: self.positions[i],
                            color: color_for_point(p, self.positions[i][2], height_range, self.cloud.intensity_min, intensity_range, color_mode, &self.class_defs),
                        }
                    })
                    .collect();

                ctx.buffer_update(
                    chunk.bindings.vertex_buffers[0],
                    BufferSource::slice(&chunk_verts),
                );
            }
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

            let height_range = self.height_range();
            let intensity_range = (self.cloud.intensity_max - self.cloud.intensity_min) as f32;

            let mut section_verts = Vec::new();
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;

            for (i, pos) in self.positions.iter().enumerate() {
                let rel_x = pos[0] - start.0;
                let rel_y = pos[1] - start.1;
                let along = rel_x * dir.0 + rel_y * dir.1;
                let perp = rel_x * normal.0 + rel_y * normal.1;

                if along >= 0.0 && along <= len && perp.abs() <= depth {
                    let p = &self.cloud.points[i];
                    h_min = h_min.min(pos[2]);
                    h_max = h_max.max(pos[2]);
                    section_verts.push(Vertex {
                        pos: [along, pos[2], 0.0],
                        color: color_for_point(p, pos[2], height_range, self.cloud.intensity_min, intensity_range, color_mode, &self.class_defs),
                    });
                }
            }

            let num_points = section_verts.len() as i32;
            println!("Section: {:.1}m long, {:.1}m depth, {} points, height {:.1}..{:.1}m",
                len, depth, num_points, h_min, h_max);

            let vertex_buffer = ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&section_verts),
            );
            let indices: Vec<u32> = (0..num_points as u32).collect();
            let index_buffer = ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&indices),
            );

            SectionData {
                bindings: Bindings {
                    vertex_buffers: vec![vertex_buffer],
                    index_buffer,
                    images: vec![],
                },
                num_points,
                length: len,
                height_min: h_min,
                height_max: h_max,
                height_range: self.height_range(),
            }
        }

        pub fn draw_section(&self, ctx: &mut dyn RenderingBackend, section: &SectionData, mvp: &glam::Mat4, point_size: f32) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp, point_size, height_range: section.height_range },
            ));
            ctx.draw(0, section.num_points, 1);
        }

        pub fn draw_visible(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, point_size: f32) -> i32 {
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
                    &shader::Uniforms { mvp: *view_proj, point_size, height_range },
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
    }

    pub struct TileStageManager {
        pipeline: Pipeline,
        pub tiles: Vec<TileInfo>,
        pub loaded: Vec<Option<LoadedTile>>,
        pub global_min: [f64; 3],
        pub global_max: [f64; 3],
        pub class_defs: Option<ClassDefs>,
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
        println!("Generating index.txt for {}", dir_path);
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
        println!("Wrote {} tile entries to index.txt", entries.len());
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

            println!("{} tiles, bounds: x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}",
                n, global_min[0], global_max[0], global_min[1], global_max[1], global_min[2], global_max[2]);

            let pipeline = create_pipeline(ctx);

            TileStageManager {
                pipeline,
                tiles,
                loaded,
                global_min,
                global_max,
                class_defs,
            }
        }

        pub fn height_range(&self) -> f32 {
            (self.global_max[2] - self.global_min[2]) as f32
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

        pub fn load_tile(&mut self, ctx: &mut dyn RenderingBackend, index: usize, color_mode: ColorMode) {
            if self.loaded[index].is_some() { return; }

            let cloud = VizPointcloud::load(&self.tiles[index].filepath);
            let height_range = self.height_range();
            let intensity_range = (cloud.intensity_max - cloud.intensity_min) as f32;

            let positions: Vec<[f32; 3]> = cloud.points.iter().map(|p| [
                (p.x - self.global_min[0]) as f32,
                (p.y - self.global_min[1]) as f32,
                (p.z - self.global_min[2]) as f32,
            ]).collect();

            let vertices: Vec<Vertex> = cloud.points.iter().enumerate().map(|(i, p)| Vertex {
                pos: positions[i],
                color: color_for_point(p, positions[i][2], height_range, cloud.intensity_min, intensity_range, color_mode, &self.class_defs),
            }).collect();

            let total_points = vertices.len() as i32;
            let chunks = build_chunks(ctx, &positions, &vertices);

            println!("Tile {} loaded: {} points, {} chunks",
                self.tiles[index].filename, total_points, chunks.len());

            self.loaded[index] = Some(LoadedTile {
                cloud,
                positions,
                chunks,
                total_points,
            });
        }

        pub fn unload_tile(&mut self, ctx: &mut dyn RenderingBackend, index: usize) {
            if let Some(tile) = self.loaded[index].take() {
                for chunk in tile.chunks {
                    ctx.delete_buffer(chunk.bindings.vertex_buffers[0]);
                    ctx.delete_buffer(chunk.bindings.index_buffer);
                }
                println!("Tile {} unloaded", self.tiles[index].filename);
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
                        &shader::Uniforms { mvp: *view_proj, point_size, height_range },
                    ));
                    ctx.draw(0, chunk.num_points, 1);
                    drawn += chunk.num_points;
                }
            }
            (drawn, total)
        }

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let height_range = self.height_range();
            for loaded in self.loaded.iter_mut().flatten() {
                let intensity_range = (loaded.cloud.intensity_max - loaded.cloud.intensity_min) as f32;
                for chunk in &mut loaded.chunks {
                    let chunk_verts: Vec<Vertex> = chunk.point_indices.iter().map(|&i| {
                        let p = &loaded.cloud.points[i];
                        Vertex {
                            pos: loaded.positions[i],
                            color: color_for_point(p, loaded.positions[i][2], height_range, loaded.cloud.intensity_min, intensity_range, color_mode, &self.class_defs),
                        }
                    }).collect();
                    ctx.buffer_update(chunk.bindings.vertex_buffers[0], BufferSource::slice(&chunk_verts));
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

            let height_range = self.height_range();

            let mut section_verts = Vec::new();
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;

            for loaded in self.loaded.iter().flatten() {
                let intensity_range = (loaded.cloud.intensity_max - loaded.cloud.intensity_min) as f32;
                for (i, pos) in loaded.positions.iter().enumerate() {
                    let rel_x = pos[0] - start.0;
                    let rel_y = pos[1] - start.1;
                    let along = rel_x * dir.0 + rel_y * dir.1;
                    let perp = rel_x * normal.0 + rel_y * normal.1;

                    if along >= 0.0 && along <= len && perp.abs() <= depth {
                        let p = &loaded.cloud.points[i];
                        h_min = h_min.min(pos[2]);
                        h_max = h_max.max(pos[2]);
                        section_verts.push(Vertex {
                            pos: [along, pos[2], 0.0],
                            color: color_for_point(p, pos[2], height_range, loaded.cloud.intensity_min, intensity_range, color_mode, &self.class_defs),
                        });
                    }
                }
            }

            let num_points = section_verts.len() as i32;
            println!("Section: {:.1}m long, {:.1}m depth, {} points, height {:.1}..{:.1}m",
                len, depth, num_points, h_min, h_max);

            let vertex_buffer = ctx.new_buffer(
                BufferType::VertexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&section_verts),
            );
            let indices: Vec<u32> = (0..num_points as u32).collect();
            let index_buffer = ctx.new_buffer(
                BufferType::IndexBuffer,
                BufferUsage::Immutable,
                BufferSource::slice(&indices),
            );

            SectionData {
                bindings: Bindings {
                    vertex_buffers: vec![vertex_buffer],
                    index_buffer,
                    images: vec![],
                },
                num_points,
                length: len,
                height_min: h_min,
                height_max: h_max,
                height_range: self.height_range(),
            }
        }

        pub fn draw_section(&self, ctx: &mut dyn RenderingBackend, section: &SectionData, mvp: &glam::Mat4, point_size: f32) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp, point_size, height_range: section.height_range },
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
        pub fn apply(&self, ctx: &mut dyn RenderingBackend, strength: f32) {
            ctx.end_render_pass();
            ctx.begin_default_pass(PassAction::Nothing);
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
    }

    /// Request sent to the background loader thread.
    struct CopcLoadReq {
        key: VoxelKey,
        offset: u64,
        byte_size: i32,
        point_count: i32,
        generation: u64,
    }

    /// Result returned by the background loader thread.
    struct CopcLoadResult {
        key: VoxelKey,
        points: Vec<Point>,
        generation: u64,
    }

    pub struct CopcStage {
        pipeline: Pipeline,
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
        pub class_defs: Option<ClassDefs>,
        /// Nodes dispatched to the loader thread but not yet ingested.
        in_flight: HashSet<VoxelKey>,
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

    /// Compute (xmin,ymin,zmin,xmax,ymax,zmax) for a VoxelKey given COPC octree params.
    fn voxel_bounds(key: &VoxelKey, cx: f64, cy: f64, cz: f64, halfsize: f64) -> (f64, f64, f64, f64, f64, f64) {
        let root_min_x = cx - halfsize;
        let root_min_y = cy - halfsize;
        let root_min_z = cz - halfsize;
        let span = halfsize * 2.0 / (1i64 << key.l) as f64;
        let xmin = root_min_x + span * key.x as f64;
        let ymin = root_min_y + span * key.y as f64;
        let zmin = root_min_z + span * key.z as f64;
        (xmin, ymin, zmin, xmin + span, ymin + span, zmin + span)
    }

    /// Spawn a background thread that services COPC load requests.
    /// The file is reopened for every request so that the LAZ decompressor
    /// starts from a clean state — each COPC node is a self-contained LAZ chunk
    /// and reusing the reader across nodes can leave stale decompressor state.
    fn spawn_copc_loader(
        filepath: String,
        rx: std::sync::mpsc::Receiver<CopcLoadReq>,
        tx: std::sync::mpsc::Sender<CopcLoadResult>,
    ) {
        std::thread::Builder::new()
            .name("copc-loader".into())
            .spawn(move || {
                for req in rx {
                    let file = match File::open(&filepath) {
                        Ok(f) => BufReader::new(f),
                        Err(e) => { eprintln!("COPC loader: open error: {}", e); continue; }
                    };
                    let mut reader = match CopcEntryReader::new(file) {
                        Ok(r) => r,
                        Err(e) => { eprintln!("COPC loader: reader error: {}", e); continue; }
                    };
                    let las_entry = las::copc::Entry {
                        key: req.key,
                        offset: req.offset,
                        byte_size: req.byte_size,
                        point_count: req.point_count,
                    };
                    let mut points = Vec::new();
                    match reader.read_entry_points(&las_entry, &mut points) {
                        Ok(_) => { let _ = tx.send(CopcLoadResult { key: req.key, points, generation: req.generation }); }
                        Err(e) => eprintln!("COPC loader error for {:?}: {}", req.key, e),
                    }
                }
            })
            .expect("Cannot spawn COPC loader thread");
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

            let hier_entries = reader.hierarchy_entries()
                .expect("No COPC hierarchy");

            let entries: Vec<CopcEntry> = hier_entries.iter()
                .filter(|e| e.point_count > 0)
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

            println!(
                "COPC: {} entries (max depth {}), root spacing {:.2}m, \
                 bounds x {:.1}..{:.1}  y {:.1}..{:.1}  z {:.1}..{:.1}",
                entries.len(), max_level, spacing,
                global_min[0], global_max[0],
                global_min[1], global_max[1],
                global_min[2], global_max[2],
            );

            let pipeline = create_pipeline(ctx);

            let (load_tx, load_rx) = std::sync::mpsc::channel::<CopcLoadReq>();
            let (result_tx, result_rx) = std::sync::mpsc::channel::<CopcLoadResult>();
            spawn_copc_loader(filepath.to_string(), load_rx, result_tx);

            let mut stage = CopcStage {
                pipeline,
                spacing,
                entries,
                loaded_nodes: HashMap::new(),
                global_min,
                global_max,
                viewer_origin,
                intensity_min: u16::MAX,
                intensity_max: 0,
                class_defs,
                in_flight: HashSet::new(),
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
            if self.in_flight.contains(&key) { return; }
            if let Some(entry) = self.entries.iter().find(|e| e.key == key) {
                let _ = self.load_tx.send(CopcLoadReq {
                    key,
                    offset: entry.offset,
                    byte_size: entry.byte_size,
                    point_count: entry.point_count,
                    generation: self.generation,
                });
                self.in_flight.insert(key);
            }
        }

        /// Convert loaded points into GPU chunks and store them.
        fn ingest_node(
            &mut self,
            ctx: &mut dyn RenderingBackend,
            key: VoxelKey,
            pts: Vec<Point>,
            color_mode: ColorMode,
        ) {
            let height_range = self.height_range();
            let intensity_range = if self.intensity_max > self.intensity_min {
                (self.intensity_max - self.intensity_min) as f32
            } else {
                65535.0
            };

            for p in &pts {
                self.intensity_min = self.intensity_min.min(p.intensity);
                self.intensity_max = self.intensity_max.max(p.intensity);
            }

            let positions: Vec<[f32; 3]> = pts.iter().map(|p| [
                (p.x - self.viewer_origin[0]) as f32,
                (p.y - self.viewer_origin[1]) as f32,
                (p.z - self.viewer_origin[2]) as f32,
            ]).collect();

            let vertices: Vec<Vertex> = pts.iter().enumerate().map(|(i, p)| Vertex {
                pos: positions[i],
                color: color_for_point(
                    p, positions[i][2], height_range,
                    self.intensity_min, intensity_range,
                    color_mode, &self.class_defs,
                ),
            }).collect();

            let total_points = vertices.len() as i32;
            let chunks = build_chunks(ctx, &positions, &vertices);

            self.loaded_nodes.insert(key, CopcLoadedNode { chunks, positions, points: pts, total_points });
        }

        /// Called every frame. Drains completed results, then dispatches new requests.
        pub fn update_view(
            &mut self,
            ctx: &mut dyn RenderingBackend,
            view_proj: &glam::Mat4,
            ortho_scale: f32,
            screen_h: f32,
            color_mode: ColorMode,
        ) {
            // Detect view changes: cancel all queued loads and bump the generation so
            // any results that were already in-flight get discarded when they arrive.
            if *view_proj != self.prev_view_proj {
                self.prev_view_proj = *view_proj;
                self.last_view_change = std::time::Instant::now();
                self.generation = self.generation.wrapping_add(1);
                self.in_flight.clear();
            }
            const SETTLE_MS: u64 = 300;
            let view_settled = self.last_view_change.elapsed()
                >= std::time::Duration::from_millis(SETTLE_MS);

            // Drain all results that the background thread has finished.
            // Discard results that belong to a previous generation (view has moved on).
            // GPU upload (ingest_node) is cheap — just vertex buffer creation.
            while let Ok(result) = self.result_rx.try_recv() {
                self.in_flight.remove(&result.key);
                if result.generation != self.generation { continue; }
                // Only ingest if not already loaded (might have been unloaded mid-flight).
                if !self.loaded_nodes.contains_key(&result.key) {
                    let n = result.points.len();
                    self.ingest_node(ctx, result.key, result.points, color_mode);
                    println!("COPC ingested ({},{},{},{}) — {} pts, {} in-flight",
                        result.key.l, result.key.x, result.key.y, result.key.z,
                        n, self.in_flight.len());
                }
            }

            // Meters per screen pixel determines the required LOD.
            let mpp = (ortho_scale * 2.0 / screen_h.max(1.0)) as f64;
            let target_depth = if mpp > 0.0 && self.spacing > 0.0 {
                (self.spacing / mpp).log2().ceil() as i32
            } else {
                0
            }.max(0).min(20);

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

            // Unload nodes that are no longer desired and not in flight.
            let to_unload: Vec<VoxelKey> = self.loaded_nodes.keys()
                .filter(|k| !desired.contains(k))
                .copied()
                .collect();
            for key in to_unload {
                if let Some(node) = self.loaded_nodes.remove(&key) {
                    for chunk in node.chunks {
                        ctx.delete_buffer(chunk.bindings.vertex_buffers[0]);
                        ctx.delete_buffer(chunk.bindings.index_buffer);
                    }
                }
            }

            // Dispatch loads for desired nodes that are not loaded and not in flight.
            // Only dispatch once the view has settled (no changes for SETTLE_MS).
            // Sort by NDC distance from screen centre so the most visible nodes load first.
            // Cap the queue depth so we never flood the channel on a sudden zoom-in.
            const MAX_IN_FLIGHT: usize = 4;
            let budget = MAX_IN_FLIGHT.saturating_sub(self.in_flight.len());
            if view_settled && budget > 0 {
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
                        &shader::Uniforms { mvp: *view_proj, point_size, height_range },
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

        pub fn rebuild_colors(&mut self, ctx: &mut dyn RenderingBackend, color_mode: ColorMode) {
            let height_range = self.height_range();
            let intensity_range = if self.intensity_max > self.intensity_min {
                (self.intensity_max - self.intensity_min) as f32
            } else {
                65535.0
            };
            for node in self.loaded_nodes.values_mut() {
                for chunk in &mut node.chunks {
                    let verts: Vec<Vertex> = chunk.point_indices.iter().map(|&i| Vertex {
                        pos: node.positions[i],
                        color: color_for_point(
                            &node.points[i], node.positions[i][2], height_range,
                            self.intensity_min, intensity_range,
                            color_mode, &self.class_defs,
                        ),
                    }).collect();
                    ctx.buffer_update(chunk.bindings.vertex_buffers[0], BufferSource::slice(&verts));
                }
            }
        }

        /// Collect section vertices from currently loaded nodes.
        /// Returns `(verts, length, height_min, height_max)`.
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

            let height_range = self.height_range();
            let intensity_range = if self.intensity_max > self.intensity_min {
                (self.intensity_max - self.intensity_min) as f32
            } else {
                65535.0
            };

            let mut verts = Vec::new();
            let mut h_min = f32::MAX;
            let mut h_max = f32::MIN;

            for node in self.loaded_nodes.values() {
                for (i, pos) in node.positions.iter().enumerate() {
                    let rel_x = pos[0] - start.0;
                    let rel_y = pos[1] - start.1;
                    let along = rel_x * dir.0 + rel_y * dir.1;
                    let perp = rel_x * normal.0 + rel_y * normal.1;
                    if along >= 0.0 && along <= len && perp.abs() <= depth {
                        h_min = h_min.min(pos[2]);
                        h_max = h_max.max(pos[2]);
                        verts.push(Vertex {
                            pos: [along, pos[2], 0.0],
                            color: color_for_point(
                                &node.points[i], pos[2], height_range,
                                self.intensity_min, intensity_range,
                                color_mode, &self.class_defs,
                            ),
                        });
                    }
                }
            }

            let h_min = if h_min == f32::MAX { 0.0 } else { h_min };
            let h_max = if h_max == f32::MIN { 0.0 } else { h_max };
            (verts, len, h_min, h_max)
        }

        pub fn extract_section(
            &self,
            ctx: &mut dyn RenderingBackend,
            start: (f32, f32),
            end: (f32, f32),
            depth: f32,
            color_mode: ColorMode,
        ) -> SectionData {
            let (verts, len, h_min, h_max) = self.extract_section_verts(start, end, depth, color_mode);
            println!("COPC Section: {:.1}m long, {:.1}m depth, {} points", len, depth, verts.len());
            make_section_data(ctx, verts, len, h_min, h_max, self.height_range())
        }

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
                &shader::Uniforms { mvp: *mvp, point_size, height_range: section.height_range },
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
        let vertex_buffer = ctx.new_buffer(
            BufferType::VertexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&verts),
        );
        let indices: Vec<u32> = (0..num_points as u32).collect();
        let index_buffer = ctx.new_buffer(
            BufferType::IndexBuffer,
            BufferUsage::Immutable,
            BufferSource::slice(&indices),
        );
        SectionData {
            bindings: Bindings {
                vertex_buffers: vec![vertex_buffer],
                index_buffer,
                images: vec![],
            },
            num_points,
            length,
            height_min,
            height_max,
            height_range,
        }
    }

    pub mod shader {
        use miniquad::*;

        pub const VERTEX: &str = r#"#version 100
            attribute vec3 pos;
            attribute vec3 color;

            uniform mat4 mvp;
            uniform float point_size;
            uniform float height_range;

            varying lowp vec3 v_color;
            varying mediump float v_depth;

            void main() {
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
                    ],
                },
            }
        }

        #[repr(C)]
        pub struct Uniforms {
            pub mvp: glam::Mat4,
            pub point_size: f32,
            pub height_range: f32,
        }
    }
//}
