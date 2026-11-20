//! Gaussian splats, from the PLY layout 3DGS trainers write (Inria, Brush,
//! gsplat, nerfstudio): position, log-scale, rotation quaternion, logit
//! opacity and spherical-harmonic colour per vertex.
//!
//! Rendering is the usual: each splat is a screen-space ellipse from its 3D
//! covariance, alpha-blended back to front. Two things make it cheaper here
//! than in a perspective viewer.
//!
//! - **The projection is exact.** Under an orthographic camera the projection
//!   is affine, so the 2D covariance is J Σ Jᵀ with J the camera's linear
//!   part — no per-splat Jacobian approximation.
//! - **The sort depends only on the view direction.** Depth along a fixed
//!   direction does not change when the camera pans or zooms, so only
//!   rotation re-sorts. That runs on a worker thread; the order on screen
//!   lags a rotation by a frame or two, which shows as nothing at all.
//!
//! Splat data lives in float textures, fetched in the vertex shader by an
//! index that is the only per-instance attribute. A re-sort then uploads 4
//! bytes per splat, not the 50 the splat itself takes.
//!
//! Colour is the SH DC term only; the view-dependent bands are read past.
//! For an inspection viewer that loses glints, not geometry.
//!
//! Georeferencing: a `scene.json` with `origin_rd` (and optionally
//! `horizontal_scale`) beside the PLY or up to three folders above it — the
//! file the haires splat pipeline trains from — places the splats in RD/NAP,
//! by `local = diag(s, s, 1) · (RD − origin)`. Without one the PLY's
//! coordinates are taken as world coordinates.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use glam::{DVec3, Mat3, Quat, Vec3};
use miniquad::*;
use rayon::prelude::*;

use crate::raw_miniquad::ZClip;

/// Width of the data textures. A power of two, so the index → texel
/// arithmetic in the shader is exact in `float`.
const TEX_W: usize = 4096;
const SH_C0: f32 = 0.282_094_8;

#[derive(Clone, Debug, PartialEq)]
pub struct Georef {
    pub offset: [f64; 3],
    pub horizontal_scale: f64,
    /// The file it came from; `None` means the PLY's own coordinates.
    pub source: Option<PathBuf>,
}

impl Georef {
    pub const IDENTITY: Georef = Georef { offset: [0.0; 3], horizontal_scale: 1.0, source: None };

    pub fn to_world(&self, p: [f32; 3]) -> DVec3 {
        let s = self.horizontal_scale;
        DVec3::new(
            p[0] as f64 / s + self.offset[0],
            p[1] as f64 / s + self.offset[1],
            p[2] as f64 + self.offset[2],
        )
    }

    /// Look for a `scene.json` with `origin_rd` beside `ply` or above it.
    pub fn find(ply: &Path) -> Georef {
        let mut dir = ply.parent();
        for _ in 0..4 {
            let Some(d) = dir else { break };
            let p = d.join("scene.json");
            if let Some(g) = std::fs::read(&p)
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .and_then(|j| Georef::from_scene_json(&j, &p))
            {
                return g;
            }
            dir = d.parent();
        }
        Georef::IDENTITY
    }

    fn from_scene_json(j: &serde_json::Value, path: &Path) -> Option<Georef> {
        let o = j.get("origin_rd")?.as_array()?;
        if o.len() < 2 {
            return None;
        }
        let z = o.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let s = j.get("horizontal_scale").and_then(|v| v.as_f64()).filter(|s| *s > 0.0).unwrap_or(1.0);
        Some(Georef { offset: [o[0].as_f64()?, o[1].as_f64()?, z], horizontal_scale: s, source: Some(path.to_path_buf()) })
    }
}

/// Splats as read from the file, in its own frame.
pub struct SplatFile {
    pub path: PathBuf,
    pub local: Vec<[f32; 3]>,
    /// Upper triangle of the 3D covariance: xx, xy, xz, yy, yz, zz.
    pub cov: Vec<[f32; 6]>,
    pub rgba: Vec<[u8; 4]>,
    /// Splats skipped as fully transparent.
    pub dropped: usize,
    pub georef: Georef,
}

#[derive(Clone, Copy)]
enum PlyType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl PlyType {
    fn parse(s: &str) -> Option<PlyType> {
        Some(match s {
            "char" | "int8" => PlyType::I8,
            "uchar" | "uint8" => PlyType::U8,
            "short" | "int16" => PlyType::I16,
            "ushort" | "uint16" => PlyType::U16,
            "int" | "int32" => PlyType::I32,
            "uint" | "uint32" => PlyType::U32,
            "float" | "float32" => PlyType::F32,
            "double" | "float64" => PlyType::F64,
            _ => return None,
        })
    }
    fn size(self) -> usize {
        match self {
            PlyType::I8 | PlyType::U8 => 1,
            PlyType::I16 | PlyType::U16 => 2,
            PlyType::I32 | PlyType::U32 | PlyType::F32 => 4,
            PlyType::F64 => 8,
        }
    }
    fn read(self, b: &[u8]) -> f32 {
        match self {
            PlyType::I8 => b[0] as i8 as f32,
            PlyType::U8 => b[0] as f32,
            PlyType::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            PlyType::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
            PlyType::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            PlyType::U32 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
            PlyType::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            PlyType::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        }
    }
}

struct PlyElement {
    name: String,
    count: usize,
    /// (name, type, byte offset in the record)
    props: Vec<(String, PlyType, usize)>,
    stride: usize,
    has_list: bool,
}

impl PlyElement {
    fn prop(&self, name: &str) -> Option<(PlyType, usize)> {
        self.props.iter().find(|p| p.0 == name).map(|p| (p.1, p.2))
    }
}

/// Whether a PLY is Gaussian splats rather than an ordinary point cloud or
/// mesh — what decides which loader a dropped `.ply` goes to.
pub fn is_splat_ply(path: &Path) -> bool {
    let Ok(f) = std::fs::File::open(path) else { return false };
    let mut r = BufReader::new(f);
    match read_header(&mut r) {
        Ok(elements) => elements
            .iter()
            .any(|e| e.name == "vertex" && e.prop("scale_0").is_some() && e.prop("rot_0").is_some() && e.prop("opacity").is_some()),
        Err(_) => false,
    }
}

fn read_header(r: &mut impl BufRead) -> Result<Vec<PlyElement>, String> {
    let mut line = String::new();
    let next = |r: &mut dyn BufRead, line: &mut String| -> Result<(), String> {
        line.clear();
        let n = r.read_line(line).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("PLY header ends early".into());
        }
        Ok(())
    };
    next(r, &mut line)?;
    if line.trim() != "ply" {
        return Err("not a PLY file".into());
    }
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut binary_le = false;
    loop {
        next(r, &mut line)?;
        let mut w = line.split_whitespace();
        match w.next() {
            Some("format") => {
                let f = w.next().unwrap_or("");
                if f != "binary_little_endian" {
                    return Err(format!("PLY format {f} is not supported, only binary_little_endian"));
                }
                binary_le = true;
            }
            Some("element") => {
                let name = w.next().unwrap_or("").to_string();
                let count = w.next().and_then(|c| c.parse().ok()).ok_or("bad PLY element line")?;
                elements.push(PlyElement { name, count, props: Vec::new(), stride: 0, has_list: false });
            }
            Some("property") => {
                let e = elements.last_mut().ok_or("PLY property before any element")?;
                let t = w.next().unwrap_or("");
                if t == "list" {
                    e.has_list = true;
                    continue;
                }
                let ty = PlyType::parse(t).ok_or_else(|| format!("unknown PLY type {t}"))?;
                let name = w.next().unwrap_or("").to_string();
                e.props.push((name, ty, e.stride));
                e.stride += ty.size();
            }
            Some("end_header") => break,
            _ => {}
        }
    }
    if !binary_le {
        return Err("PLY has no format line".into());
    }
    Ok(elements)
}

/// One decoded splat: centre, covariance, colour.
type SplatRow = ([f32; 3], [f32; 6], [u8; 4]);

pub fn read_ply(path: &Path) -> Result<SplatFile, String> {
    let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut r = BufReader::with_capacity(1 << 22, f);
    let elements = read_header(&mut r)?;
    if elements.iter().any(|e| e.name == "chunk") {
        return Err("compressed PLY (SuperSplat) is not supported yet — export an uncompressed PLY".into());
    }
    // Elements before the vertices are skipped; only fixed-size ones can be.
    for e in &elements {
        if e.name == "vertex" {
            break;
        }
        if e.has_list {
            return Err(format!("PLY element {} before the vertices has list properties", e.name));
        }
        std::io::copy(&mut (&mut r).take((e.count * e.stride) as u64), &mut std::io::sink()).map_err(|e| e.to_string())?;
    }
    let v = elements.iter().find(|e| e.name == "vertex").ok_or("PLY has no vertex element")?;
    if v.has_list {
        return Err("PLY vertices have list properties".into());
    }
    let need = |n: &str| v.prop(n).ok_or_else(|| format!("not a Gaussian-splat PLY: no {n} property"));
    let pos = [need("x")?, need("y")?, need("z")?];
    let scale = [need("scale_0")?, need("scale_1")?, need("scale_2")?];
    let rot = [need("rot_0")?, need("rot_1")?, need("rot_2")?, need("rot_3")?];
    let opacity = need("opacity")?;
    // DC colour; plain 8-bit colour as a fallback some exporters write.
    let dc = match (v.prop("f_dc_0"), v.prop("f_dc_1"), v.prop("f_dc_2")) {
        (Some(a), Some(b), Some(c)) => Some([a, b, c]),
        _ => None,
    };
    let rgb8 = match (v.prop("red"), v.prop("green"), v.prop("blue")) {
        (Some(a), Some(b), Some(c)) => Some([a, b, c]),
        _ => None,
    };

    let mut local = Vec::with_capacity(v.count);
    let mut cov = Vec::with_capacity(v.count);
    let mut rgba = Vec::with_capacity(v.count);
    let mut dropped = 0;
    const BATCH: usize = 1 << 16;
    let mut buf = vec![0u8; BATCH * v.stride];
    let mut left = v.count;
    while left > 0 {
        let n = left.min(BATCH);
        let bytes = &mut buf[..n * v.stride];
        r.read_exact(bytes).map_err(|e| format!("{}: {e} (file shorter than its header says)", path.display()))?;
        let rows: Vec<Option<SplatRow>> = bytes
            .par_chunks_exact(v.stride)
            .map(|rec| {
                let g = |(t, o): (PlyType, usize)| t.read(&rec[o..]);
                let alpha = 1.0 / (1.0 + (-g(opacity)).exp());
                // NaN opacity is dropped along with the transparent.
                if alpha.is_nan() || alpha < 1.0 / 255.0 {
                    return None;
                }
                let p = [g(pos[0]), g(pos[1]), g(pos[2])];
                if !p.iter().all(|c| c.is_finite()) {
                    return None;
                }
                let s = Vec3::new(g(scale[0]).exp(), g(scale[1]).exp(), g(scale[2]).exp());
                let q = Quat::from_xyzw(g(rot[1]), g(rot[2]), g(rot[3]), g(rot[0]));
                let q = if q.length_squared() > 0.0 { q.normalize() } else { Quat::IDENTITY };
                let m = Mat3::from_quat(q) * Mat3::from_diagonal(s);
                let c = m * m.transpose();
                let rgb = match (dc, rgb8) {
                    (Some(dc), _) => [0, 1, 2].map(|k| ((0.5 + SH_C0 * g(dc[k])).clamp(0.0, 1.0) * 255.0 + 0.5) as u8),
                    (None, Some(c8)) => [0, 1, 2].map(|k| g(c8[k]).clamp(0.0, 255.0) as u8),
                    _ => [200, 200, 200],
                };
                Some((
                    p,
                    [c.x_axis.x, c.y_axis.x, c.z_axis.x, c.y_axis.y, c.z_axis.y, c.z_axis.z],
                    [rgb[0], rgb[1], rgb[2], (alpha * 255.0 + 0.5) as u8],
                ))
            })
            .collect();
        for row in rows {
            match row {
                Some((p, c, col)) => {
                    local.push(p);
                    cov.push(c);
                    rgba.push(col);
                }
                None => dropped += 1,
            }
        }
        left -= n;
    }
    if local.is_empty() {
        return Err("the PLY has no visible splats".into());
    }
    Ok(SplatFile { path: path.to_path_buf(), local, cov, rgba, dropped, georef: Georef::find(path) })
}

impl SplatFile {
    /// World box of the bulk of the splats, 0.5th to 99.5th percentile per
    /// axis. Trainers leave a haze of floaters kilometres out; framing on the
    /// full extent would shrink the scene to a dot.
    pub fn core_bounds(&self) -> (DVec3, DVec3) {
        let mut lo = [0f32; 3];
        let mut hi = [0f32; 3];
        let n = self.local.len();
        let (a, b) = (n / 200, (n - 1) - n / 200);
        for k in 0..3 {
            let mut v: Vec<f32> = self.local.iter().map(|p| p[k]).collect();
            lo[k] = *v.select_nth_unstable_by(a, f32::total_cmp).1;
            hi[k] = *v.select_nth_unstable_by(b, f32::total_cmp).1;
        }
        (self.georef.to_world(lo), self.georef.to_world(hi))
    }
}

/// An RGBA32F texture, made by miniquad (which owns and frees it) and filled
/// through raw GL, since miniquad offers only half floats. Positions in half
/// float would be centimetres off a few tens of metres out.
fn float_texture(ctx: &mut dyn RenderingBackend, w: usize, h: usize, data: &[f32]) -> TextureId {
    assert_eq!(data.len(), w * h * 4);
    let tex = ctx.new_texture(
        TextureAccess::Static,
        TextureSource::Empty,
        TextureParams {
            format: TextureFormat::RGBA16F,
            wrap: TextureWrap::Clamp,
            min_filter: FilterMode::Nearest,
            mag_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::None,
            width: w as u32,
            height: h as u32,
            ..Default::default()
        },
    );
    const GL_TEXTURE_BINDING_2D: u32 = 0x8069;
    // Irrefutable where miniquad is built without Metal.
    #[allow(irrefutable_let_patterns)]
    if let RawId::OpenGl(id) = unsafe { ctx.texture_raw_id(tex) } {
        unsafe {
            use miniquad::gl::*;
            // Put the binding back exactly as found: miniquad caches it and
            // would otherwise skip a bind it believes is already in place.
            let mut prev: GLint = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &mut prev);
            glBindTexture(GL_TEXTURE_2D, id);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA32F as GLint,
                w as GLint,
                h as GLint,
                0,
                GL_RGBA,
                GL_FLOAT,
                data.as_ptr() as *const _,
            );
            glBindTexture(GL_TEXTURE_2D, prev as GLuint);
        }
    }
    tex
}

/// Back-to-front order along `front`, as the `float` indices the shader
/// reads. Keys pack the depth's sortable bits above the index, so one
/// parallel `u64` sort does it.
fn sort_order(centers: &[[f32; 3]], front: [f32; 3]) -> Vec<f32> {
    let f = Vec3::from(front);
    let mut keys: Vec<u64> = centers
        .par_iter()
        .enumerate()
        .map(|(i, c)| {
            // Farthest first: ascending in −depth.
            let d = -Vec3::from(*c).dot(f);
            let b = d.to_bits();
            let ord = if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 };
            ((ord as u64) << 32) | i as u64
        })
        .collect();
    keys.par_sort_unstable();
    keys.par_iter().map(|k| (*k & 0xFFFF_FFFF) as f32).collect()
}

pub struct SplatCloud {
    pub path: PathBuf,
    pub visible: bool,
    pub count: usize,
    pub dropped: usize,
    pub georef: Georef,
    /// Hide splats centred outside the scene's box; see `crop_box`.
    pub crop: bool,
    world_lo: DVec3,
    world_hi: DVec3,
    /// Viewer-space box splats must be centred in while `crop` is on.
    crop_lo: Vec3,
    crop_hi: Vec3,
    textures: [TextureId; 3],
    tex_h: usize,
    quad: BufferId,
    quad_idx: BufferId,
    order: BufferId,
    pipeline: Pipeline,
    sort_tx: Option<mpsc::Sender<[f32; 3]>>,
    sort_rx: mpsc::Receiver<(Vec<f32>, [f32; 3])>,
    sorted_for: [f32; 3],
    sort_pending: bool,
}

impl SplatCloud {
    pub fn attach(ctx: &mut dyn RenderingBackend, file: SplatFile, origin: [f64; 3], front: Vec3) -> SplatCloud {
        let (world_lo, world_hi) = file.core_bounds();
        let n = file.local.len();
        let o = DVec3::from(origin);
        // Trainers surround the scene with large, faint background splats.
        // A perspective viewer sits inside that shell and sees it as sky;
        // under an orthographic camera every one of them along the line of
        // sight lands in front of the scene. The crop box is the core box
        // grown by a tenth of its horizontal size on every side — upwards
        // too, so roofs and trees above the 99.5th height percentile stay.
        let ext = world_hi - world_lo;
        let margin = DVec3::splat(0.1 * ext.x.max(ext.y));
        let (crop_lo, crop_hi) = ((world_lo - margin - o).as_vec3(), (world_hi + margin - o).as_vec3());
        let g = &file.georef;
        let centers: Vec<[f32; 3]> = file
            .local
            .par_iter()
            .map(|p| {
                let w = g.to_world(*p) - o;
                [w.x as f32, w.y as f32, w.z as f32]
            })
            .collect();

        let tex_h = n.div_ceil(TEX_W);
        let mut t0 = vec![0f32; TEX_W * tex_h * 4];
        let mut t1 = vec![0f32; TEX_W * tex_h * 4];
        let mut t2 = vec![0f32; TEX_W * tex_h * 4];
        // The horizontal scale shrinks x and y on the way to world; the
        // covariance follows as D Σ Dᵀ.
        let s = g.horizontal_scale as f32;
        let (sxx, sxz) = (1.0 / (s * s), 1.0 / s);
        for i in 0..n {
            let c = centers[i];
            let [r, gr, b, a] = file.rgba[i];
            let packed = r as f32 * 65536.0 + gr as f32 * 256.0 + b as f32;
            let v = file.cov[i];
            t0[i * 4..i * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], packed]);
            t1[i * 4..i * 4 + 4].copy_from_slice(&[v[0] * sxx, v[1] * sxx, v[2] * sxz, a as f32 / 255.0]);
            t2[i * 4..i * 4 + 4].copy_from_slice(&[v[3] * sxx, v[4] * sxz, v[5], 0.0]);
        }
        let textures = [
            float_texture(ctx, TEX_W, tex_h, &t0),
            float_texture(ctx, TEX_W, tex_h, &t1),
            float_texture(ctx, TEX_W, tex_h, &t2),
        ];

        let corners: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let quad = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&corners));
        let quad_idx = ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&[0u16, 1, 2, 0, 2, 3]));
        let first = sort_order(&centers, front.to_array());
        let order = ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Dynamic, BufferSource::slice(&first));

        let centers = Arc::new(centers);
        let (sort_tx, req_rx) = mpsc::channel::<[f32; 3]>();
        let (res_tx, sort_rx) = mpsc::channel();
        {
            let centers = centers.clone();
            std::thread::Builder::new()
                .name("splat-sort".into())
                .spawn(move || {
                    while let Ok(mut f) = req_rx.recv() {
                        // Only the latest direction matters.
                        while let Ok(newer) = req_rx.try_recv() {
                            f = newer;
                        }
                        if res_tx.send((sort_order(&centers, f), f)).is_err() {
                            break;
                        }
                    }
                })
                .expect("cannot spawn the splat sort thread");
        }

        ::log::info!(
            "{} splats from {} ({} transparent ones skipped){}",
            n,
            file.path.display(),
            file.dropped,
            match &g.source {
                Some(p) => format!(", placed by {}", p.display()),
                None => String::new(),
            }
        );

        SplatCloud {
            path: file.path,
            visible: true,
            count: n,
            dropped: file.dropped,
            georef: file.georef,
            crop: true,
            world_lo,
            world_hi,
            crop_lo,
            crop_hi,
            textures,
            tex_h,
            quad,
            quad_idx,
            order,
            pipeline: splat_pipeline(ctx),
            sort_tx: Some(sort_tx),
            sort_rx,
            sorted_for: front.to_array(),
            sort_pending: false,
        }
    }

    pub fn name(&self) -> String {
        self.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
    }

    /// World box of the bulk of the splats; see [`SplatFile::core_bounds`].
    pub fn world_bounds(&self) -> (DVec3, DVec3) {
        (self.world_lo, self.world_hi)
    }

    /// Take a finished sort, and ask for another if the view has turned.
    pub fn update(&mut self, ctx: &mut dyn RenderingBackend, front: Vec3) {
        let mut latest = None;
        while let Ok(r) = self.sort_rx.try_recv() {
            latest = Some(r);
        }
        if let Some((order, f)) = latest {
            ctx.buffer_update(self.order, BufferSource::slice(&order));
            self.sorted_for = f;
            self.sort_pending = false;
        }
        if !self.visible || self.sort_pending {
            return;
        }
        // About a tenth of a degree.
        if Vec3::from(self.sorted_for).dot(front) < 0.999_998
            && let Some(tx) = &self.sort_tx
            && tx.send(front.to_array()).is_ok()
        {
            self.sort_pending = true;
        }
    }

    /// `viewport` is the view's size in physical pixels.
    pub fn draw(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, viewport: (f32, f32), z_clip: ZClip) {
        if !self.visible {
            return;
        }
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&Bindings {
            vertex_buffers: vec![self.quad, self.order],
            index_buffer: self.quad_idx,
            images: self.textures.to_vec(),
        });
        ctx.apply_uniforms(UniformsSource::table(&splat_shader::Uniforms {
            mvp: *view_proj,
            viewport: [viewport.0, viewport.1],
            tex_size: [TEX_W as f32, self.tex_h as f32],
            z_clip_min: z_clip.min,
            z_clip_max: z_clip.max,
            crop_lo: if self.crop { self.crop_lo.to_array() } else { [-1.0e9; 3] },
            crop_hi: if self.crop { self.crop_hi.to_array() } else { [1.0e9; 3] },
        }));
        // Test depth, do not write it; see `splat_pipeline`. Put back
        // straight after: with the mask off, the next frame's depth clear
        // would silently do nothing.
        unsafe { miniquad::gl::glDepthMask(miniquad::gl::GL_FALSE as _) };
        ctx.draw(0, 6, self.count as i32);
        unsafe { miniquad::gl::glDepthMask(miniquad::gl::GL_TRUE as _) };
    }

    pub fn unload(mut self, ctx: &mut dyn RenderingBackend) {
        self.sort_tx = None;
        for t in self.textures {
            ctx.delete_texture(t);
        }
        ctx.delete_buffer(self.quad);
        ctx.delete_buffer(self.quad_idx);
        ctx.delete_buffer(self.order);
        ctx.delete_pipeline(self.pipeline);
    }
}

fn splat_pipeline(ctx: &mut dyn RenderingBackend) -> Pipeline {
    let shader = ctx
        .new_shader(ShaderSource::Glsl { vertex: splat_shader::VERTEX, fragment: splat_shader::FRAGMENT }, splat_shader::meta())
        .expect("splat shader");
    ctx.new_pipeline(
        &[BufferLayout::default(), BufferLayout { step_func: VertexStep::PerInstance, ..Default::default() }],
        &[
            VertexAttribute::with_buffer("corner", VertexFormat::Float2, 0),
            VertexAttribute::with_buffer("splat_idx", VertexFormat::Float1, 1),
        ],
        shader,
        PipelineParams {
            primitive_type: PrimitiveType::Triangles,
            // Hidden behind points and meshes, but never hiding each other:
            // the sort and the blend do that. So depth is tested and not
            // written — but miniquad turns the depth *test* off along with
            // the write for a pipeline that does not write, so this one says
            // it writes and `draw` masks the writes off around the draw call.
            depth_test: Comparison::LessOrEqual,
            depth_write: true,
            cull_face: CullFace::Nothing,
            color_blend: Some(BlendState::new(
                Equation::Add,
                BlendFactor::Value(BlendValue::SourceAlpha),
                BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
            )),
            // Leave the framebuffer's alpha alone.
            alpha_blend: Some(BlendState::new(Equation::Add, BlendFactor::Zero, BlendFactor::One)),
            ..Default::default()
        },
    )
}

mod splat_shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
        attribute vec2 corner;
        attribute float splat_idx;

        uniform sampler2D tex_center;
        uniform sampler2D tex_cov_a;
        uniform sampler2D tex_cov_b;
        uniform mat4 mvp;
        uniform vec2 viewport;
        uniform vec2 tex_size;
        uniform float z_clip_min;
        uniform float z_clip_max;
        uniform vec3 crop_lo;
        uniform vec3 crop_hi;

        varying mediump vec4 v_color;
        varying mediump vec2 v_pos;

        void main() {
            v_color = vec4(0.0);
            v_pos = vec2(0.0);
            highp float row = floor((splat_idx + 0.5) / tex_size.x);
            highp float col = splat_idx - row * tex_size.x;
            highp vec2 uv = vec2((col + 0.5) / tex_size.x, (row + 0.5) / tex_size.y);
            highp vec4 t0 = texture2D(tex_center, uv);
            if (t0.z < z_clip_min || t0.z > z_clip_max
                || any(lessThan(t0.xyz, crop_lo)) || any(greaterThan(t0.xyz, crop_hi))) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }
            highp vec4 t1 = texture2D(tex_cov_a, uv);
            highp vec4 t2 = texture2D(tex_cov_b, uv);
            highp vec4 clip = mvp * vec4(t0.xyz, 1.0);

            // Orthographic, so the screen covariance is exactly J S J^T, J
            // the first two rows of the camera's linear part, in pixels.
            highp vec3 jx = vec3(mvp[0][0], mvp[1][0], mvp[2][0]) * (0.5 * viewport.x);
            highp vec3 jy = vec3(mvp[0][1], mvp[1][1], mvp[2][1]) * (0.5 * viewport.y);
            highp mat3 S = mat3(t1.x, t1.y, t1.z,
                                t1.y, t2.x, t2.y,
                                t1.z, t2.y, t2.z);
            highp vec3 sy = S * jy;
            // 0.3 px^2: the low-pass every 3DGS renderer adds, so a splat
            // never falls between pixels.
            highp float a = dot(jx, S * jx) + 0.3;
            highp float b = dot(jx, sy);
            highp float d = dot(jy, sy) + 0.3;
            highp float mid = 0.5 * (a + d);
            highp float rad = length(vec2(0.5 * (a - d), b));
            highp float l1 = mid + rad;
            highp float l2 = max(mid - rad, 0.1);
            // Eigenvector of the larger eigenvalue, from whichever of the two
            // textbook forms is better conditioned.
            highp vec2 va = vec2(b, l1 - a);
            highp vec2 vb = vec2(l1 - d, b);
            highp vec2 e1 = dot(va, va) > dot(vb, vb) ? va : vb;
            e1 = dot(e1, e1) > 1e-20 ? normalize(e1) : vec2(1.0, 0.0);
            highp vec2 e2 = vec2(-e1.y, e1.x);
            // Three sigma, capped so a splat filling the screen when zoomed
            // right in cannot cost the frame.
            highp float r1 = min(3.0 * sqrt(l1), 2048.0);
            highp float r2 = min(3.0 * sqrt(l2), 2048.0);
            if (abs(clip.x) > 1.0 + 2.0 * r1 / viewport.x || abs(clip.y) > 1.0 + 2.0 * r1 / viewport.y) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                return;
            }
            highp vec2 off = (corner.x * r1) * e1 + (corner.y * r2) * e2;
            gl_Position = vec4(clip.xy + off * 2.0 / viewport, clip.z, 1.0);
            v_pos = corner * 3.0;

            // Colour is packed as r * 65536 + g * 256 + b: exact in a float.
            highp float p = t0.w;
            highp float cr = floor(p / 65536.0);
            highp float cg = floor((p - cr * 65536.0) / 256.0);
            highp float cb = p - cr * 65536.0 - cg * 256.0;
            v_color = vec4(cr / 255.0, cg / 255.0, cb / 255.0, t1.w);
        }"#;

    pub const FRAGMENT: &str = r#"#version 100
        precision mediump float;
        varying mediump vec4 v_color;
        varying mediump vec2 v_pos;

        void main() {
            float r2 = dot(v_pos, v_pos);
            if (r2 > 9.0) discard;
            float a = v_color.a * exp(-0.5 * r2);
            if (a < 1.0 / 255.0) discard;
            gl_FragColor = vec4(v_color.rgb, a);
        }"#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec!["tex_center".to_string(), "tex_cov_a".to_string(), "tex_cov_b".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("mvp", UniformType::Mat4),
                    UniformDesc::new("viewport", UniformType::Float2),
                    UniformDesc::new("tex_size", UniformType::Float2),
                    UniformDesc::new("z_clip_min", UniformType::Float1),
                    UniformDesc::new("z_clip_max", UniformType::Float1),
                    UniformDesc::new("crop_lo", UniformType::Float3),
                    UniformDesc::new("crop_hi", UniformType::Float3),
                ],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub mvp: glam::Mat4,
        pub viewport: [f32; 2],
        pub tex_size: [f32; 2],
        pub z_clip_min: f32,
        pub z_clip_max: f32,
        pub crop_lo: [f32; 3],
        pub crop_hi: [f32; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_ply(path: &Path, rows: &[[f32; 14]]) {
        let names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"];
        let mut h = format!("ply\nformat binary_little_endian 1.0\ncomment test\nelement vertex {}\n", rows.len());
        for n in names {
            h += &format!("property float {n}\n");
        }
        h += "end_header\n";
        let mut b = h.into_bytes();
        for r in rows {
            for v in r {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(path, b).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("haiviz-splats-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn splat_attributes_are_activated() {
        let d = scratch("attrs");
        let p = d.join("s.ply");
        // Scale e^0 = 1 along x, e^-2 elsewhere, rotated 90° about z: the long
        // axis ends up along y. Opacity logit 0 is alpha one half.
        let h = std::f32::consts::FRAC_1_SQRT_2;
        write_ply(&p, &[
            [1.0, 2.0, 3.0, 0.0, -2.0, -2.0, 0.0, h, 0.0, 0.0, h, 0.0, 1.0, -10.0],
            // Fully transparent: dropped.
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        assert!(is_splat_ply(&p));
        let f = read_ply(&p).unwrap();
        assert_eq!(f.local, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(f.dropped, 1);
        let c = f.cov[0];
        assert!((c[3] - 1.0).abs() < 1e-5, "yy {c:?}");
        assert!((c[0] - (-4.0f32).exp()).abs() < 1e-5, "xx {c:?}");
        assert!(c[1].abs() < 1e-5);
        assert_eq!(f.rgba[0][3], 128);
        assert_eq!(f.rgba[0][0], 128); // 0.5 + C0·0
        assert_eq!(f.rgba[0][2], 0); // clamped
        assert_eq!(f.georef, Georef::IDENTITY);
    }

    #[test]
    fn scene_json_above_the_ply_places_it_in_rd() {
        let d = scratch("georef");
        let run = d.join("runs/brush");
        std::fs::create_dir_all(&run).unwrap();
        std::fs::write(d.join("scene.json"), r#"{"origin_rd":[149730.0,464777.0,0.0],"horizontal_scale":2.0}"#).unwrap();
        let g = Georef::find(&run.join("export.ply"));
        assert_eq!(g.offset, [149730.0, 464777.0, 0.0]);
        assert_eq!(g.to_world([10.0, -4.0, 12.0]), DVec3::new(149735.0, 464775.0, 12.0));
    }

    /// Against a real PLY: `HAIVIZ_SPLATS=/path/export.ply cargo test
    /// --release real_splats -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn real_splats_read_and_sort() {
        let Ok(p) = std::env::var("HAIVIZ_SPLATS") else { return };
        let t = std::time::Instant::now();
        let f = read_ply(Path::new(&p)).unwrap();
        println!("{} splats ({} dropped) read in {:?}, georef {:?}", f.local.len(), f.dropped, t.elapsed(), f.georef);
        let t = std::time::Instant::now();
        let (lo, hi) = f.core_bounds();
        println!("core bounds {lo} .. {hi} in {:?}", t.elapsed());
        let t = std::time::Instant::now();
        let order = sort_order(&f.local, [0.3, 0.2, -0.93]);
        println!("sorted in {:?}", t.elapsed());
        assert_eq!(order.len(), f.local.len());
    }

    #[test]
    fn sort_is_back_to_front() {
        let c = [[0.0, 0.0, 5.0], [0.0, 0.0, -3.0], [0.0, 0.0, 1.0]];
        // Looking down: the lowest splat is farthest and drawn first.
        assert_eq!(sort_order(&c, [0.0, 0.0, -1.0]), vec![1.0, 2.0, 0.0]);
        assert_eq!(sort_order(&c, [0.0, 0.0, 1.0]), vec![0.0, 2.0, 1.0]);
    }

    #[test]
    fn plain_point_cloud_ply_is_not_a_splat() {
        let d = scratch("plain");
        let p = d.join("pc.ply");
        std::fs::write(&p, b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n").unwrap();
        assert!(!is_splat_ply(&p));
    }
}
