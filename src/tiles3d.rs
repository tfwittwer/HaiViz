//! 3D Tiles tilesets — 1.1, and 1.0's content formats — streamed by
//! screen-space error under haiviz's orthographic camera.
//!
//! Orthographic makes the LOD rule simpler than in a perspective viewer: a
//! tile's screen-space error is its geometric error over the metres one
//! screen pixel spans, the same everywhere in the view. Refinement is
//! therefore a function of zoom alone, and frustum culling is what keeps the
//! working set to the tiles actually on screen.
//!
//! Coordinates. A tileset whose root lies near the Earth's centre-distance is
//! in ECEF (EPSG:4978), and is put in a local east-north-up frame at its
//! centre so that up is up; it will not line up with point clouds in a
//! projected CRS. Anything else is taken to be in the same CRS as the clouds
//! already loaded — as `haires amesh` writes them, in RD/NAP — and lands on
//! them directly.
//!
//! One deliberate deviation from the spec: a tile *without content* never
//! ends a traversal. Per spec it would, drawing nothing, and a tileset whose
//! upper levels are empty containers then shows a blank screen over a range of
//! zoom levels. Descending to the first tiles that have content instead draws
//! more detail than asked for, which beats drawing nothing.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};

use glam::{DMat3, DMat4, DVec3};
use miniquad::*;
use serde_json::Value;

use crate::gltf_load::{self, CpuMesh};
use crate::raw_miniquad::{ZClip, aabb_outside_frustum, extract_frustum_planes};

/// How meshes are shaded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lighting {
    /// Lit unless the material says `KHR_materials_unlit` — photogrammetry
    /// with baked light usually does.
    Material,
    /// Lit regardless — shows the shape of a mesh whose colours are flat.
    On,
    Off,
}

impl Lighting {
    pub const ALL: [Lighting; 3] = [Lighting::Material, Lighting::On, Lighting::Off];
    pub fn label(self) -> &'static str {
        match self {
            Lighting::Material => "As material",
            Lighting::On => "On",
            Lighting::Off => "Off",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Refine {
    Add,
    Replace,
}

enum State {
    /// The tile has nothing to draw of its own.
    Empty,
    Unloaded,
    Queued,
    Loaded(GpuTile),
    Failed,
}

struct Node {
    children: Vec<usize>,
    /// World-space box, f64, for framing and the viewer origin.
    world_lo: DVec3,
    world_hi: DVec3,
    /// Viewer-space box, for culling.
    lo: [f32; 3],
    hi: [f32; 3],
    geometric_error: f64,
    refine: Refine,
    /// Tile to world, accumulated down the tree.
    transform: DMat4,
    uris: Vec<PathBuf>,
    /// External tilesets this tile points at, grafted in on first visit.
    external: Vec<PathBuf>,
    depth: u32,
    state: State,
    last_used: u64,
}

struct GpuPrim {
    bindings: Bindings,
    num_indices: i32,
    base_color: [f32; 4],
    unlit: bool,
    has_normals: bool,
    alpha_cutoff: f32,
}

struct GpuTile {
    prims: Vec<GpuPrim>,
    textures: Vec<TextureId>,
    bytes: usize,
    triangles: usize,
}

impl GpuTile {
    fn upload(ctx: &mut dyn RenderingBackend, mesh: CpuMesh, white: TextureId) -> GpuTile {
        let mut bytes = 0;
        let textures: Vec<TextureId> = mesh
            .images
            .iter()
            .map(|img| {
                let t = ctx.new_texture_from_data_and_format(
                    &img.rgba,
                    TextureParams {
                        format: TextureFormat::RGBA8,
                        wrap: if img.repeat { TextureWrap::Repeat } else { TextureWrap::Clamp },
                        min_filter: FilterMode::Linear,
                        mag_filter: FilterMode::Linear,
                        mipmap_filter: MipmapFilterMode::Linear,
                        width: img.width,
                        height: img.height,
                        allocate_mipmaps: true,
                        ..Default::default()
                    },
                );
                ctx.texture_generate_mipmaps(t);
                // A full mip chain adds a third.
                bytes += img.rgba.len() * 4 / 3;
                t
            })
            .collect();
        let triangles = mesh.triangle_count();
        let prims = mesh
            .primitives
            .into_iter()
            .map(|p| {
                let vb = |ctx: &mut dyn RenderingBackend, src: BufferSource| {
                    ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, src)
                };
                bytes += p.positions.len() * (12 + 12 + 4 + 8) + p.indices.len() * 4;
                let pos = vb(ctx, BufferSource::slice(&p.positions));
                let nrm = vb(ctx, BufferSource::slice(&p.normals));
                let col = vb(ctx, BufferSource::slice(&p.colors));
                let uv = vb(ctx, BufferSource::slice(&p.uvs));
                let idx = ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&p.indices));
                let tex = p.texture.and_then(|t| textures.get(t).copied()).unwrap_or(white);
                GpuPrim {
                    bindings: Bindings { vertex_buffers: vec![pos, nrm, col, uv], index_buffer: idx, images: vec![tex] },
                    num_indices: p.indices.len() as i32,
                    base_color: p.base_color,
                    unlit: p.unlit,
                    has_normals: p.has_normals,
                    alpha_cutoff: p.alpha_cutoff.unwrap_or(-1.0),
                }
            })
            .collect();
        GpuTile { prims, textures, bytes, triangles }
    }

    fn free(self, ctx: &mut dyn RenderingBackend) {
        for p in self.prims {
            for vb in p.bindings.vertex_buffers {
                ctx.delete_buffer(vb);
            }
            ctx.delete_buffer(p.bindings.index_buffer);
        }
        for t in self.textures {
            ctx.delete_texture(t);
        }
    }
}

struct LoadReq {
    node: usize,
    uris: Vec<PathBuf>,
    transform: DMat4,
    origin: [f64; 3],
}

struct LoadResult {
    node: usize,
    result: Result<CpuMesh, String>,
}

fn spawn_loader_pool(rx: mpsc::Receiver<LoadReq>, tx: mpsc::Sender<LoadResult>, workers: usize) {
    let rx = Arc::new(Mutex::new(rx));
    for i in 0..workers {
        let rx = rx.clone();
        let tx = tx.clone();
        std::thread::Builder::new()
            .name(format!("tiles-loader-{i}"))
            .spawn(move || {
                loop {
                    // The lock is held only for the receive; a closed channel
                    // (the tileset was unloaded) ends the worker.
                    let req = match rx.lock().unwrap().recv() {
                        Ok(r) => r,
                        Err(_) => break,
                    };
                    let mut mesh = CpuMesh::default();
                    let mut result = Ok(());
                    for uri in &req.uris {
                        match gltf_load::load_content(uri, &req.transform, req.origin) {
                            Ok(m) => mesh_append(&mut mesh, m),
                            Err(e) => {
                                result = Err(format!("{}: {e}", uri.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()));
                                break;
                            }
                        }
                    }
                    let result = result.map(|_| mesh);
                    if tx.send(LoadResult { node: req.node, result }).is_err() {
                        break;
                    }
                }
            })
            .expect("cannot spawn a tiles loader thread");
    }
}

fn mesh_append(into: &mut CpuMesh, mut other: CpuMesh) {
    let base = into.images.len();
    for p in &mut other.primitives {
        if let Some(t) = p.texture.as_mut() {
            *t += base;
        }
    }
    into.images.append(&mut other.images);
    into.primitives.append(&mut other.primitives);
    into.warnings.append(&mut other.warnings);
}

// ---- Parsing ----

const WGS84_A: f64 = 6_378_137.0;
const WGS84_E2: f64 = 6.694_379_990_14e-3;

fn geodetic_to_ecef(lon: f64, lat: f64, h: f64) -> DVec3 {
    let (sl, cl) = lat.sin_cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sl * sl).sqrt();
    DVec3::new((n + h) * cl * lon.cos(), (n + h) * cl * lon.sin(), (n * (1.0 - WGS84_E2) + h) * sl)
}

/// Geodetic latitude and longitude of an ECEF position (Bowring, one step —
/// sub-millimetre near the surface, which is all a frame needs).
fn ecef_to_lon_lat(p: DVec3) -> (f64, f64) {
    let b = WGS84_A * (1.0 - WGS84_E2).sqrt();
    let ep2 = (WGS84_A * WGS84_A - b * b) / (b * b);
    let r = (p.x * p.x + p.y * p.y).sqrt();
    let th = (p.z * WGS84_A).atan2(r * b);
    let lat = (p.z + ep2 * b * th.sin().powi(3)).atan2(r - WGS84_E2 * WGS84_A * th.cos().powi(3));
    (p.y.atan2(p.x), lat)
}

/// ECEF to east-north-up at the ellipsoid point under `centre`, so local Z is
/// height above the ellipsoid.
fn enu_frame(centre: DVec3) -> DMat4 {
    let (lon, lat) = ecef_to_lon_lat(centre);
    let (so, co) = lon.sin_cos();
    let (sa, ca) = lat.sin_cos();
    let east = DVec3::new(-so, co, 0.0);
    let north = DVec3::new(-sa * co, -sa * so, ca);
    let up = DVec3::new(ca * co, ca * so, sa);
    let base = geodetic_to_ecef(lon, lat, 0.0);
    // Rows east/north/up: the transpose of the matrix with them as columns.
    let rot = DMat3::from_cols(east, north, up).transpose();
    DMat4::from_mat3(rot) * DMat4::from_translation(-base)
}

fn f64_array(v: Option<&Value>) -> Option<Vec<f64>> {
    v?.as_array()?.iter().map(Value::as_f64).collect()
}

/// World AABB of a bounding volume. `transform` is the tile's (which
/// includes the ENU frame, if any); `frame` alone applies to `region`, which
/// is always geodetic and never moved by tile transforms.
fn volume_aabb(bv: &Value, transform: &DMat4, frame: &DMat4) -> Option<(DVec3, DVec3)> {
    if let Some(b) = f64_array(bv.get("box")).filter(|b| b.len() == 12) {
        let c = transform.transform_point3(DVec3::new(b[0], b[1], b[2]));
        let lin = DMat3::from_mat4(*transform);
        let mut ext = DVec3::ZERO;
        for k in 0..3 {
            let a = lin * DVec3::new(b[3 + 3 * k], b[4 + 3 * k], b[5 + 3 * k]);
            ext += a.abs();
        }
        return Some((c - ext, c + ext));
    }
    if let Some(s) = f64_array(bv.get("sphere")).filter(|s| s.len() == 4) {
        let c = transform.transform_point3(DVec3::new(s[0], s[1], s[2]));
        let lin = DMat3::from_mat4(*transform);
        let scale = lin.x_axis.length().max(lin.y_axis.length()).max(lin.z_axis.length());
        let r = DVec3::splat(s[3] * scale);
        return Some((c - r, c + r));
    }
    if let Some(r) = f64_array(bv.get("region")).filter(|r| r.len() == 6) {
        let (w, s, e, n, h0, h1) = (r[0], r[1], r[2], r[3], r[4], r[5]);
        let mut lo = DVec3::splat(f64::MAX);
        let mut hi = DVec3::splat(f64::MIN);
        // Corners and edge midpoints at both heights: the region's surface
        // bulges between its corners, by metres over a city.
        for &lon in &[w, (w + e) * 0.5, e] {
            for &lat in &[s, (s + n) * 0.5, n] {
                for &h in &[h0, h1] {
                    let p = frame.transform_point3(geodetic_to_ecef(lon, lat, h));
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        return Some((lo, hi));
    }
    None
}

/// Strip the query a web tileset hangs on its URIs and resolve the rest
/// against the referring file's directory.
fn resolve_uri(base_dir: &Path, uri: &str) -> PathBuf {
    let uri = uri.split(['?', '#']).next().unwrap_or(uri);
    let uri = uri.strip_prefix("file://").unwrap_or(uri);
    let mut out = Vec::new();
    let b = uri.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(&uri[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    base_dir.join(String::from_utf8_lossy(&out).as_ref())
}

fn is_tileset_json(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("json"))
}

struct Parser<'a> {
    nodes: &'a mut Vec<Node>,
    frame: DMat4,
}

impl Parser<'_> {
    /// Append `tile` and its subtree; returns its index.
    fn tile(&mut self, tile: &Value, parent_transform: &DMat4, parent_refine: Refine, base_dir: &Path, depth: u32) -> Result<usize, String> {
        if tile.get("implicitTiling").is_some() {
            return Err("implicit tiling is not supported yet".into());
        }
        let local = f64_array(tile.get("transform"))
            .filter(|m| m.len() == 16)
            .map(|m| DMat4::from_cols_slice(&m))
            .unwrap_or(DMat4::IDENTITY);
        let transform = *parent_transform * local;
        let (world_lo, world_hi) = tile
            .get("boundingVolume")
            .and_then(|bv| volume_aabb(bv, &transform, &self.frame))
            .unwrap_or((DVec3::splat(-1.0e12), DVec3::splat(1.0e12)));
        let refine = match tile.get("refine").and_then(Value::as_str) {
            Some(r) if r.eq_ignore_ascii_case("add") => Refine::Add,
            Some(r) if r.eq_ignore_ascii_case("replace") => Refine::Replace,
            _ => parent_refine,
        };
        let mut uris = Vec::new();
        let mut external = Vec::new();
        let mut contents: Vec<&Value> = Vec::new();
        if let Some(c) = tile.get("content") {
            contents.push(c);
        }
        if let Some(cs) = tile.get("contents").and_then(Value::as_array) {
            contents.extend(cs.iter());
        }
        for c in contents {
            // 1.0 spelled it `url`.
            if let Some(u) = c.get("uri").or_else(|| c.get("url")).and_then(Value::as_str) {
                let p = resolve_uri(base_dir, u);
                if is_tileset_json(&p) { external.push(p) } else { uris.push(p) }
            }
        }
        let idx = self.nodes.len();
        self.nodes.push(Node {
            children: Vec::new(),
            world_lo,
            world_hi,
            lo: [f32::MIN; 3],
            hi: [f32::MAX; 3],
            geometric_error: tile.get("geometricError").and_then(Value::as_f64).unwrap_or(0.0),
            refine,
            transform,
            state: if uris.is_empty() { State::Empty } else { State::Unloaded },
            uris,
            external,
            depth,
            last_used: 0,
        });
        if let Some(children) = tile.get("children").and_then(Value::as_array) {
            for c in children {
                let ci = self.tile(c, &transform, refine, base_dir, depth + 1)?;
                self.nodes[idx].children.push(ci);
            }
        }
        Ok(idx)
    }
}

fn read_tileset_json(path: &Path) -> Result<Value, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let json: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    if json.get("root").is_none() {
        return Err(format!("{} has no root tile — is it a 3D Tiles tileset?", path.display()));
    }
    Ok(json)
}

// ---- The tileset ----

pub struct TilesetStats {
    pub loaded: usize,
    pub queued: usize,
    pub failed: usize,
    pub gpu_bytes: usize,
    pub drawn_tiles: usize,
    pub drawn_triangles: usize,
    /// The first failure, so the panel can say why rather than only how many.
    pub first_error: Option<String>,
    pub warnings: usize,
    pub first_warning: Option<String>,
}

pub struct Tileset {
    pub path: PathBuf,
    pub visible: bool,
    /// Placed in a local ENU frame because it was in ECEF.
    pub ecef: bool,
    /// ECEF to the local frame, or identity. Region volumes in external
    /// tilesets need it as much as the root's do.
    frame: DMat4,
    nodes: Vec<Node>,
    origin: [f64; 3],
    pipeline: Pipeline,
    white: TextureId,
    req_tx: Option<mpsc::Sender<LoadReq>>,
    res_rx: mpsc::Receiver<LoadResult>,
    ready: VecDeque<LoadResult>,
    workers: usize,
    in_flight: usize,
    render: Vec<usize>,
    wishes: Vec<(u32, f32, usize)>,
    pub stats: TilesetStats,
}

/// Per-frame inputs to the traversal.
struct View {
    planes: [[f32; 4]; 6],
    view_proj: glam::Mat4,
    mpp: f64,
    max_sse: f64,
    frame: u64,
}

impl Tileset {
    /// Read `tileset.json` and build the tile tree. No GPU work and no viewer
    /// origin yet: `world_bounds` is what the origin gets pinned from, and
    /// [`Tileset::attach`] finishes the job.
    pub fn open(path: &Path) -> Result<TilesetDesc, String> {
        let json = read_tileset_json(path)?;
        let version = json.get("asset").and_then(|a| a.get("version")).and_then(Value::as_str).unwrap_or("?");
        if !matches!(version, "1.0" | "1.1") {
            ::log::warn!("{}: 3D Tiles version {version}, reading it as 1.1", path.display());
        }
        let base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let root = &json["root"];

        // Decide the frame from the root alone: a centre near the Earth's
        // radius from the origin, or a region volume, means ECEF.
        let root_local = f64_array(root.get("transform"))
            .filter(|m| m.len() == 16)
            .map(|m| DMat4::from_cols_slice(&m))
            .unwrap_or(DMat4::IDENTITY);
        let is_region = root.get("boundingVolume").and_then(|b| b.get("region")).is_some();
        let (lo, hi) = root
            .get("boundingVolume")
            .and_then(|bv| volume_aabb(bv, &root_local, &DMat4::IDENTITY))
            .ok_or("the root tile has no usable boundingVolume")?;
        let centre = (lo + hi) * 0.5;
        let ecef = is_region || centre.length() > 6.0e6;
        let frame = if ecef { enu_frame(centre) } else { DMat4::IDENTITY };

        let mut nodes = Vec::new();
        Parser { nodes: &mut nodes, frame }.tile(root, &frame, Refine::Replace, &base_dir, 0)?;
        Ok(TilesetDesc { path: path.to_path_buf(), nodes, ecef, frame })
    }

    pub fn attach(ctx: &mut dyn RenderingBackend, desc: TilesetDesc, origin: [f64; 3]) -> Tileset {
        let TilesetDesc { path, mut nodes, ecef, frame } = desc;
        for n in &mut nodes {
            set_viewer_box(n, origin);
        }
        let (req_tx, req_rx) = mpsc::channel();
        let (res_tx, res_rx) = mpsc::channel();
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 8);
        spawn_loader_pool(req_rx, res_tx, workers);
        let white = ctx.new_texture_from_rgba8(1, 1, &[255, 255, 255, 255]);
        info_tiles(&path, &nodes, ecef);
        Tileset {
            path,
            visible: true,
            ecef,
            frame,
            nodes,
            origin,
            pipeline: mesh_pipeline(ctx),
            white,
            req_tx: Some(req_tx),
            res_rx,
            ready: VecDeque::new(),
            workers,
            in_flight: 0,
            render: Vec::new(),
            wishes: Vec::new(),
            stats: TilesetStats {
                loaded: 0,
                queued: 0,
                failed: 0,
                gpu_bytes: 0,
                drawn_tiles: 0,
                drawn_triangles: 0,
                first_error: None,
                warnings: 0,
                first_warning: None,
            },
        }
    }

    pub fn name(&self) -> String {
        // `tileset.json` says nothing; its folder says which tileset it is.
        let file = self.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        match self.path.parent().and_then(|p| p.file_name()) {
            Some(dir) => format!("{}/{}", dir.to_string_lossy(), file),
            None => file,
        }
    }

    /// World-space box of the root.
    pub fn world_bounds(&self) -> (DVec3, DVec3) {
        (self.nodes[0].world_lo, self.nodes[0].world_hi)
    }

    /// Stream and select tiles for this frame. `mpp` is metres per physical
    /// pixel — the unit `max_sse` is in.
    pub fn update(&mut self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, mpp: f64, max_sse: f64, budget_bytes: usize, frame: u64) {
        while let Ok(r) = self.res_rx.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            self.ready.push_back(r);
        }
        // Uploads are capped per frame so a burst of arrivals is spread over
        // a few frames instead of stalling one.
        let mut uploaded = 0usize;
        while uploaded < 96 << 20 {
            let Some(r) = self.ready.pop_front() else { break };
            let node = &mut self.nodes[r.node];
            match r.result {
                Ok(mesh) => {
                    if let Some(w) = mesh.warnings.first() {
                        self.stats.warnings += mesh.warnings.len();
                        if self.stats.first_warning.is_none() {
                            ::log::warn!("{}: {w}", self.path.display());
                            self.stats.first_warning = Some(w.clone());
                        }
                    }
                    let tile = GpuTile::upload(ctx, mesh, self.white);
                    uploaded += tile.bytes;
                    self.stats.gpu_bytes += tile.bytes;
                    self.stats.loaded += 1;
                    node.state = State::Loaded(tile);
                }
                Err(e) => {
                    self.stats.failed += 1;
                    if self.stats.first_error.is_none() {
                        ::log::error!("{}: {e}", self.path.display());
                        self.stats.first_error = Some(e);
                    }
                    node.state = State::Failed;
                }
            }
        }

        self.render.clear();
        self.wishes.clear();
        if self.visible {
            let view = View { planes: extract_frustum_planes(view_proj), view_proj: *view_proj, mpp: mpp.max(1e-9), max_sse, frame };
            self.visit(0, &view);
        }

        // Coarse tiles first — they cover the most screen per byte — then
        // the ones nearest the middle of the view.
        self.wishes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let cap = self.workers * 2;
        if let Some(tx) = &self.req_tx {
            for &(_, _, i) in &self.wishes {
                if self.in_flight >= cap {
                    break;
                }
                let n = &mut self.nodes[i];
                if !matches!(n.state, State::Unloaded) {
                    continue;
                }
                if tx.send(LoadReq { node: i, uris: n.uris.clone(), transform: n.transform, origin: self.origin }).is_ok() {
                    n.state = State::Queued;
                    self.in_flight += 1;
                }
            }
        }
        self.stats.queued = self.in_flight + self.ready.len();

        // Evict the least recently used tiles not drawn this frame.
        if self.stats.gpu_bytes > budget_bytes {
            let mut cands: Vec<(u64, usize)> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, n)| matches!(n.state, State::Loaded(_)) && n.last_used != frame)
                .map(|(i, n)| (n.last_used, i))
                .collect();
            cands.sort_unstable();
            for (_, i) in cands {
                if self.stats.gpu_bytes <= budget_bytes {
                    break;
                }
                if let State::Loaded(t) = std::mem::replace(&mut self.nodes[i].state, State::Unloaded) {
                    self.stats.gpu_bytes -= t.bytes;
                    self.stats.loaded -= 1;
                    t.free(ctx);
                }
            }
        }

        self.stats.drawn_tiles = self.render.len();
        self.stats.drawn_triangles = self
            .render
            .iter()
            .map(|&i| if let State::Loaded(t) = &self.nodes[i].state { t.triangles } else { 0 })
            .sum();
    }

    /// Ask for a tile; true if it can be drawn now.
    fn want(&mut self, i: usize, view: &View) -> bool {
        let n = &mut self.nodes[i];
        n.last_used = view.frame;
        match n.state {
            State::Loaded(_) => true,
            State::Unloaded => {
                let c = glam::Vec3::new((n.lo[0] + n.hi[0]) * 0.5, (n.lo[1] + n.hi[1]) * 0.5, (n.lo[2] + n.hi[2]) * 0.5);
                let p = view.view_proj * c.extend(1.0);
                self.wishes.push((n.depth, p.x * p.x + p.y * p.y, i));
                false
            }
            _ => false,
        }
    }

    /// Select what to draw under tile `i`. Returns whether everything the
    /// subtree should show is on screen — a REPLACE parent stays up until its
    /// children are complete, so zooming in never opens a hole.
    fn visit(&mut self, i: usize, view: &View) -> bool {
        if aabb_outside_frustum(&view.planes, &self.nodes[i].lo, &self.nodes[i].hi) {
            return true;
        }
        if !self.nodes[i].external.is_empty() {
            self.graft_external(i);
        }
        let n = &self.nodes[i];
        let has_content = !matches!(n.state, State::Empty);
        let has_children = !n.children.is_empty();
        let refine = has_children && (!has_content || n.geometric_error / view.mpp > view.max_sse);
        // A failed tile is as complete as it will ever be.
        let failed = matches!(n.state, State::Failed);

        if !refine {
            if !has_content || failed {
                return true;
            }
            if self.want(i, view) {
                self.render.push(i);
                return true;
            }
            return false;
        }

        let children = self.nodes[i].children.clone();
        if self.nodes[i].refine == Refine::Add {
            let mut complete = true;
            if has_content && !failed {
                if self.want(i, view) {
                    self.render.push(i);
                } else {
                    complete = false;
                }
            }
            for c in children {
                complete &= self.visit(c, view);
            }
            return complete;
        }

        let mark = self.render.len();
        let mut complete = true;
        for c in children {
            complete &= self.visit(c, view);
        }
        if complete || !has_content || failed {
            return complete;
        }
        // Children not all in yet: show this tile instead, while they load.
        if self.want(i, view) {
            self.render.truncate(mark);
            self.render.push(i);
            return true;
        }
        false
    }

    fn graft_external(&mut self, i: usize) {
        let ext = std::mem::take(&mut self.nodes[i].external);
        let parent_transform = self.nodes[i].transform;
        let refine = self.nodes[i].refine;
        let depth = self.nodes[i].depth + 1;
        for path in ext {
            let base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            let first_new = self.nodes.len();
            let result = read_tileset_json(&path).and_then(|json| {
                // Region volumes inside an external tileset are geodetic like
                // any other; the frame is whatever the parent tileset chose.
                let frame = self.frame;
                Parser { nodes: &mut self.nodes, frame }.tile(&json["root"], &parent_transform, refine, &base_dir, depth)
            });
            match result {
                Ok(root) => {
                    for k in first_new..self.nodes.len() {
                        set_viewer_box(&mut self.nodes[k], self.origin);
                    }
                    self.nodes[i].children.push(root);
                }
                Err(e) => {
                    self.nodes.truncate(first_new);
                    self.stats.failed += 1;
                    if self.stats.first_error.is_none() {
                        ::log::error!("{e}");
                        self.stats.first_error = Some(e);
                    }
                }
            }
        }
    }

    pub fn draw(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, light_dir: glam::Vec3, lighting: Lighting, z_clip: ZClip) {
        if !self.visible || self.render.is_empty() {
            return;
        }
        ctx.apply_pipeline(&self.pipeline);
        for &i in &self.render {
            let State::Loaded(tile) = &self.nodes[i].state else { continue };
            for p in &tile.prims {
                let lit = match lighting {
                    Lighting::Material => !p.unlit && p.has_normals,
                    Lighting::On => p.has_normals,
                    Lighting::Off => false,
                };
                ctx.apply_bindings(&p.bindings);
                ctx.apply_uniforms(UniformsSource::table(&mesh_shader::Uniforms {
                    mvp: *view_proj,
                    base_color: p.base_color,
                    light_dir: light_dir.to_array(),
                    lit: if lit { 1.0 } else { 0.0 },
                    alpha_cutoff: p.alpha_cutoff,
                    z_clip_min: z_clip.min,
                    z_clip_max: z_clip.max,
                }));
                ctx.draw(0, p.num_indices, 1);
            }
        }
    }

    pub fn unload(mut self, ctx: &mut dyn RenderingBackend) {
        // Closing the request channel lets the workers run dry and exit.
        self.req_tx = None;
        for n in self.nodes.drain(..) {
            if let State::Loaded(t) = n.state {
                t.free(ctx);
            }
        }
        ctx.delete_texture(self.white);
        ctx.delete_pipeline(self.pipeline);
    }
}

/// A parsed tileset, waiting for the viewer origin.
pub struct TilesetDesc {
    pub path: PathBuf,
    nodes: Vec<Node>,
    pub ecef: bool,
    frame: DMat4,
}

impl TilesetDesc {
    pub fn world_bounds(&self) -> (DVec3, DVec3) {
        (self.nodes[0].world_lo, self.nodes[0].world_hi)
    }
    #[cfg(test)]
    fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

fn set_viewer_box(n: &mut Node, origin: [f64; 3]) {
    let o = DVec3::from(origin);
    // Unbounded tiles keep the sentinel so they are never culled.
    if n.world_lo.x > -1.0e11 {
        let lo = n.world_lo - o;
        let hi = n.world_hi - o;
        n.lo = [lo.x as f32, lo.y as f32, lo.z as f32];
        n.hi = [hi.x as f32, hi.y as f32, hi.z as f32];
    }
}

fn info_tiles(path: &Path, nodes: &[Node], ecef: bool) {
    let with_content = nodes.iter().filter(|n| !n.uris.is_empty()).count();
    let external = nodes.iter().filter(|n| !n.external.is_empty()).count();
    ::log::info!(
        "3D Tiles {}: {} tiles ({} with content, {} external tilesets){}",
        path.display(),
        nodes.len(),
        with_content,
        external,
        if ecef { ", ECEF — shown in a local east-north-up frame" } else { "" }
    );
}

fn mesh_pipeline(ctx: &mut dyn RenderingBackend) -> Pipeline {
    let shader = ctx
        .new_shader(ShaderSource::Glsl { vertex: mesh_shader::VERTEX, fragment: mesh_shader::FRAGMENT }, mesh_shader::meta())
        .expect("mesh shader");
    ctx.new_pipeline(
        &[BufferLayout::default(), BufferLayout::default(), BufferLayout::default(), BufferLayout::default()],
        &[
            VertexAttribute::with_buffer("pos", VertexFormat::Float3, 0),
            VertexAttribute::with_buffer("normal", VertexFormat::Float3, 1),
            VertexAttribute::with_buffer("color", VertexFormat::Byte4, 2),
            VertexAttribute::with_buffer("uv", VertexFormat::Float2, 3),
        ],
        shader,
        PipelineParams {
            primitive_type: PrimitiveType::Triangles,
            depth_test: Comparison::LessOrEqual,
            depth_write: true,
            // Winding in tilesets is not to be trusted, and a roof seen from
            // below should still occlude.
            cull_face: CullFace::Nothing,
            ..Default::default()
        },
    )
}

mod mesh_shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
        attribute vec3 pos;
        attribute vec3 normal;
        attribute vec4 color;
        attribute vec2 uv;

        uniform mat4 mvp;
        uniform vec4 base_color;
        uniform vec3 light_dir;
        uniform float lit;

        varying lowp vec4 v_color;
        varying highp vec2 v_uv;
        varying mediump float v_shade;
        varying highp float v_z;

        void main() {
            gl_Position = mvp * vec4(pos, 1.0);
            // Byte attributes arrive unnormalised.
            v_color = color / 255.0 * base_color;
            v_uv = uv;
            v_z = pos.z;
            // Two-sided: the side facing away from the light is as lit as the
            // side facing it, since winding cannot be relied on.
            mediump float d = abs(dot(normal, light_dir));
            v_shade = mix(1.0, 0.35 + 0.65 * d, lit);
        }"#;

    pub const FRAGMENT: &str = r#"#version 100
        precision mediump float;
        uniform sampler2D tex;
        uniform float alpha_cutoff;
        uniform float z_clip_min;
        uniform float z_clip_max;

        varying lowp vec4 v_color;
        varying highp vec2 v_uv;
        varying mediump float v_shade;
        varying highp float v_z;

        void main() {
            // The height band cuts meshes as it culls points, so a storey can
            // be looked into the same way in both.
            if (v_z < z_clip_min || v_z > z_clip_max) discard;
            vec4 c = texture2D(tex, v_uv) * v_color;
            if (c.a < alpha_cutoff) discard;
            gl_FragColor = vec4(c.rgb * v_shade, 1.0);
        }"#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec!["tex".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("mvp", UniformType::Mat4),
                    UniformDesc::new("base_color", UniformType::Float4),
                    UniformDesc::new("light_dir", UniformType::Float3),
                    UniformDesc::new("lit", UniformType::Float1),
                    UniformDesc::new("alpha_cutoff", UniformType::Float1),
                    UniformDesc::new("z_clip_min", UniformType::Float1),
                    UniformDesc::new("z_clip_max", UniformType::Float1),
                ],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub mvp: glam::Mat4,
        pub base_color: [f32; 4],
        pub light_dir: [f32; 3],
        pub lit: f32,
        pub alpha_cutoff: f32,
        pub z_clip_min: f32,
        pub z_clip_max: f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("haiviz-tiles-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn box_volumes_are_moved_by_the_accumulated_transform() {
        let d = scratch("box");
        let p = d.join("tileset.json");
        std::fs::write(
            &p,
            r#"{"asset":{"version":"1.1"},"geometricError":10,
               "root":{"transform":[1,0,0,0,0,1,0,0,0,0,1,0,185735,428319,0,1],"refine":"REPLACE",
                 "boundingVolume":{"box":[0,0,50,100,0,0,0,200,0,0,0,50]},"geometricError":8,
                 "children":[{"boundingVolume":{"box":[10,0,50,10,0,0,0,10,0,0,0,10]},"geometricError":0,
                   "content":{"uri":"tiles/a%20b.glb?v=2"}}]}}"#,
        )
        .unwrap();
        let desc = Tileset::open(&p).unwrap();
        assert!(!desc.ecef);
        assert_eq!(desc.node_count(), 2);
        let (lo, hi) = desc.world_bounds();
        assert_eq!(lo, DVec3::new(185635.0, 428119.0, 0.0));
        assert_eq!(hi, DVec3::new(185835.0, 428519.0, 100.0));
        assert_eq!(desc.nodes[1].uris[0], d.join("tiles/a b.glb"));
        assert_eq!(desc.nodes[1].refine, Refine::Replace);
        assert_eq!(desc.nodes[1].world_lo, DVec3::new(185735.0, 428309.0, 40.0));
    }

    #[test]
    fn ecef_roots_get_an_up_that_points_up() {
        // A box around a point on the ground in Nijmegen.
        let c = geodetic_to_ecef(5.86f64.to_radians(), 51.84f64.to_radians(), 50.0);
        let frame = enu_frame(c);
        let local = frame.transform_point3(c);
        assert!(local.x.abs() < 1e-6 && local.y.abs() < 1e-6, "{local}");
        assert!((local.z - 50.0).abs() < 1e-3, "{local}");
        let up = frame.transform_point3(geodetic_to_ecef(5.86f64.to_radians(), 51.84f64.to_radians(), 150.0));
        assert!((up.z - 150.0).abs() < 1e-3 && up.x.abs() < 1e-3);
    }

    /// Against a real tileset: `HAIVIZ_TILESET=/path/tileset.json cargo test
    /// --release real_tileset -- --ignored --nocapture`. Decodes one tile per
    /// level and checks its vertices land inside the tile's box.
    #[test]
    #[ignore]
    fn real_tileset_content_lands_inside_its_tiles() {
        let Ok(p) = std::env::var("HAIVIZ_TILESET") else { return };
        let t0 = std::time::Instant::now();
        let desc = Tileset::open(Path::new(&p)).unwrap();
        println!("{} nodes parsed in {:?}, root box {:?}", desc.node_count(), t0.elapsed(), desc.world_bounds());
        let (lo, _) = desc.world_bounds();
        let origin = lo.to_array();
        let mut seen_depths = std::collections::HashSet::new();
        for n in &desc.nodes {
            if n.uris.is_empty() || !seen_depths.insert(n.depth) {
                continue;
            }
            let t = std::time::Instant::now();
            let m = gltf_load::load_content(&n.uris[0], &n.transform, origin).unwrap();
            let (blo, bhi) = (n.world_lo - DVec3::from(origin), n.world_hi - DVec3::from(origin));
            let mut outside = 0;
            let mut count = 0;
            for p in m.primitives.iter().flat_map(|p| p.positions.iter()) {
                count += 1;
                let v = DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                if v.cmplt(blo - 0.01).any() || v.cmpgt(bhi + 0.01).any() {
                    outside += 1;
                }
            }
            println!(
                "depth {} {}: {} vertices, {} triangles, {} images, {} outside the box, decoded in {:?}",
                n.depth,
                n.uris[0].display(),
                count,
                m.triangle_count(),
                m.images.len(),
                outside,
                t.elapsed()
            );
            assert_eq!(outside, 0);
        }
    }

    #[test]
    fn implicit_tiling_is_refused_with_a_reason() {
        let d = scratch("implicit");
        let p = d.join("tileset.json");
        std::fs::write(
            &p,
            r#"{"asset":{"version":"1.1"},"geometricError":10,"root":{"boundingVolume":{"box":[0,0,0,1,0,0,0,1,0,0,0,1]},
               "geometricError":8,"implicitTiling":{"subdivisionScheme":"QUADTREE"}}}"#,
        )
        .unwrap();
        let e = Tileset::open(&p).err().unwrap();
        assert!(e.contains("implicit"), "{e}");
    }
}
