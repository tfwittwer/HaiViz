//pub mod raw_miniquad {
    use std::collections::HashMap;
    use std::path::Path;

    use las::{Point, Reader};
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
            }
        }

        pub fn draw_section(&self, ctx: &mut dyn RenderingBackend, section: &SectionData, mvp: &glam::Mat4, point_size: f32) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp, point_size, height_range: self.height_range() },
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
            }
        }

        pub fn draw_section(&self, ctx: &mut dyn RenderingBackend, section: &SectionData, mvp: &glam::Mat4, point_size: f32) {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&section.bindings);
            ctx.apply_uniforms(miniquad::UniformsSource::table(
                &shader::Uniforms { mvp: *mvp, point_size, height_range: self.height_range() },
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
