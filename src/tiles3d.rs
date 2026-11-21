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
//! A tileset that carries `ESRI_crs` — a projected CRS, with a second
//! transform and bounding volume per tile in it, as `haires splat-tiles`
//! writes beside the ECEF ones — is placed by those instead, so it lands on
//! RD clouds exactly rather than in an approximate local frame.
//!
//! Gaussian-splat content goes into a [`SplatAtlas`] per tileset and is drawn
//! with one sort across all the tiles on screen.
//!
//! Point-cloud content (`pnts`, or glTF `POINTS`) is drawn in the cloud's own
//! point pass rather than with the meshes, so it gets the point size and EDL
//! the clouds have. Its geometric error is roughly its point spacing, and a
//! tile that holds points is refined against its own, finer threshold.
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
use crate::splats::{SplatAtlas, SplatTexels};

/// What a tile holds. Each kind has its own detail threshold, since a
/// geometric error means a different thing for each.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Not loaded yet, and nothing in the URI says. Treated as its parent's.
    Unknown,
    Mesh,
    Points,
    Splats,
}

/// Screen-space error thresholds, in physical pixels, per kind of content.
#[derive(Clone, Copy, Debug)]
pub struct Detail {
    pub mesh: f64,
    pub points: f64,
    pub splats: f64,
}

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
    /// Known once loaded; `.pnts` says points before that.
    kind: Kind,
}

struct GpuPrim {
    bindings: Bindings,
    num_indices: i32,
    base_color: [f32; 4],
    unlit: bool,
    has_normals: bool,
    alpha_cutoff: f32,
}

/// Points draw through the tileset's shared index buffer (0, 1, 2, …), so
/// they carry no indices of their own.
struct GpuPoints {
    pos: BufferId,
    col: BufferId,
    count: i32,
}

struct GpuTile {
    prims: Vec<GpuPrim>,
    points: Vec<GpuPoints>,
    textures: Vec<TextureId>,
    /// The tile's splats and their first slot in the tileset's atlas.
    splats: Option<(Arc<SplatTexels>, usize)>,
    bytes: usize,
    triangles: usize,
    point_count: usize,
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
        let point_count = mesh.point_count();
        let points = mesh
            .points
            .iter()
            .map(|p| {
                bytes += p.positions.len() * (12 + 4);
                GpuPoints {
                    pos: ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&p.positions)),
                    col: ctx.new_buffer(BufferType::VertexBuffer, BufferUsage::Immutable, BufferSource::slice(&p.colors)),
                    count: p.positions.len() as i32,
                }
            })
            .collect();
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
        GpuTile { prims, points, textures, splats: None, bytes, triangles, point_count }
    }

    /// Free the buffers and textures. The atlas slots are the caller's to
    /// give back.
    fn free(self, ctx: &mut dyn RenderingBackend) {
        for p in self.prims {
            for vb in p.bindings.vertex_buffers {
                ctx.delete_buffer(vb);
            }
            ctx.delete_buffer(p.bindings.index_buffer);
        }
        for p in self.points {
            ctx.delete_buffer(p.pos);
            ctx.delete_buffer(p.col);
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
                            Ok(m) => mesh.append(m),
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
    /// Read each tile's `ESRI_crs` transform and volume where it has them.
    esri: bool,
}

impl Parser<'_> {
    /// Append `tile` and its subtree; returns its index.
    fn tile(&mut self, tile: &Value, parent_transform: &DMat4, parent_refine: Refine, base_dir: &Path, depth: u32) -> Result<usize, String> {
        if tile.get("implicitTiling").is_some() {
            return Err("implicit tiling is not supported yet".into());
        }
        // The projected twin of the tile's placement, where there is one.
        let esri = tile.get("extensions").and_then(|e| e.get("ESRI_crs")).filter(|_| self.esri);
        let placed = |k: &str| esri.and_then(|e| e.get(k)).or_else(|| tile.get(k));
        let local = f64_array(placed("transform"))
            .filter(|m| m.len() == 16)
            .map(|m| DMat4::from_cols_slice(&m))
            .unwrap_or(DMat4::IDENTITY);
        let transform = *parent_transform * local;
        let (world_lo, world_hi) = placed("boundingVolume")
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
        let pnts = !uris.is_empty() && uris.iter().all(|u| u.extension().is_some_and(|e| e.eq_ignore_ascii_case("pnts")));
        let kind = if pnts { Kind::Points } else { Kind::Unknown };
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
            kind,
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
    pub drawn_points: usize,
    pub drawn_splats: usize,
    /// Loading has stopped because the tiles on screen fill the budget.
    pub budget_full: bool,
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
    /// Placed by its `ESRI_crs` twin: the CRS it names, e.g. "EPSG:28992+5709".
    pub esri_crs: Option<String>,
    /// ECEF to the local frame, or identity. Region volumes in external
    /// tilesets need it as much as the root's do.
    frame: DMat4,
    nodes: Vec<Node>,
    origin: [f64; 3],
    pipeline: Pipeline,
    point_pipeline: Pipeline,
    /// The point pipeline with colour writes off, for EDL's depth replay.
    point_depth_pipeline: Pipeline,
    /// 0, 1, 2, … for drawing points; grown to the largest tile seen.
    iota: Option<(BufferId, usize)>,
    /// Made when the first splat tile arrives.
    atlas: Option<SplatAtlas>,
    /// A splat tile found no room even after growing and evicting; loading
    /// waits until a tile is released.
    atlas_full: bool,
    white: TextureId,
    req_tx: Option<mpsc::Sender<LoadReq>>,
    res_rx: mpsc::Receiver<LoadResult>,
    ready: VecDeque<LoadResult>,
    workers: usize,
    in_flight: usize,
    render: Vec<usize>,
    wishes: Vec<(u32, f32, usize)>,
    pub stats: TilesetStats,
    /// What the tileset holds, as far as is known: the panel offers the
    /// settings that apply.
    pub has_points: bool,
    pub has_mesh: bool,
    pub has_splats: bool,
}

/// Per-frame inputs to the traversal.
struct View {
    planes: [[f32; 4]; 6],
    view_proj: glam::Mat4,
    mpp: f64,
    detail: Detail,
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
        let esri_crs = esri_projected_crs(&json);
        let ecef = esri_crs.is_none() && (is_region || centre.length() > 6.0e6);
        let frame = if ecef { enu_frame(centre) } else { DMat4::IDENTITY };

        let mut nodes = Vec::new();
        Parser { nodes: &mut nodes, frame, esri: esri_crs.is_some() }.tile(root, &frame, Refine::Replace, &base_dir, 0)?;
        Ok(TilesetDesc { path: path.to_path_buf(), nodes, ecef, esri_crs, frame })
    }

    pub fn attach(ctx: &mut dyn RenderingBackend, desc: TilesetDesc, origin: [f64; 3]) -> Tileset {
        let TilesetDesc { path, mut nodes, ecef, esri_crs, frame } = desc;
        for n in &mut nodes {
            set_viewer_box(n, origin);
        }
        let (req_tx, req_rx) = mpsc::channel();
        let (res_tx, res_rx) = mpsc::channel();
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(2, 8);
        spawn_loader_pool(req_rx, res_tx, workers);
        let white = ctx.new_texture_from_rgba8(1, 1, &[255, 255, 255, 255]);
        info_tiles(&path, &nodes, ecef, esri_crs.as_deref());
        let (has_points, has_mesh, has_splats) = content_kinds(&nodes);
        Tileset {
            path,
            visible: true,
            ecef,
            esri_crs,
            frame,
            nodes,
            origin,
            pipeline: mesh_pipeline(ctx),
            point_pipeline: point_pipeline(ctx, false),
            point_depth_pipeline: point_pipeline(ctx, true),
            iota: None,
            atlas: None,
            atlas_full: false,
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
                drawn_points: 0,
                drawn_splats: 0,
                budget_full: false,
                first_error: None,
                warnings: 0,
                first_warning: None,
            },
            has_points,
            has_mesh,
            has_splats,
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
    /// pixel — the unit `detail` is in; `front` orders the splats.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        ctx: &mut dyn RenderingBackend,
        view_proj: &glam::Mat4,
        front: glam::Vec3,
        mpp: f64,
        detail: Detail,
        budget_bytes: usize,
        frame: u64,
    ) {
        while let Ok(r) = self.res_rx.try_recv() {
            self.in_flight = self.in_flight.saturating_sub(1);
            self.ready.push_back(r);
        }
        // Uploads are capped per frame so a burst of arrivals is spread over
        // a few frames instead of stalling one.
        let mut uploaded = 0usize;
        let mut kinds_changed = false;
        while uploaded < 96 << 20 {
            let Some(r) = self.ready.pop_front() else { break };
            match r.result {
                Ok(mut mesh) => {
                    if let Some(w) = mesh.warnings.first() {
                        self.stats.warnings += mesh.warnings.len();
                        if self.stats.first_warning.is_none() {
                            ::log::warn!("{}: {w}", self.path.display());
                            self.stats.first_warning = Some(w.clone());
                        }
                    }
                    let (tris, pts, spl) = (!mesh.primitives.is_empty(), !mesh.points.is_empty(), mesh.splat_count() > 0);
                    let kind = match (tris, pts, spl) {
                        (false, true, false) => Kind::Points,
                        (false, false, true) => Kind::Splats,
                        (false, false, false) => self.nodes[r.node].kind,
                        _ => Kind::Mesh,
                    };
                    kinds_changed |= self.nodes[r.node].kind != kind;
                    self.nodes[r.node].kind = kind;
                    let splats = std::mem::take(&mut mesh.splats).into_iter().reduce(|mut a, b| {
                        a.append(b);
                        a
                    });
                    let largest = mesh.points.iter().map(|p| p.positions.len()).max().unwrap_or(0);
                    let mut tile = GpuTile::upload(ctx, mesh, self.white);
                    if let Some(sp) = splats.filter(|s| !s.is_empty()) {
                        match self.place_splats(ctx, &sp, frame) {
                            Some(start) => {
                                tile.bytes += sp.len() * 48;
                                tile.splats = Some((Arc::new(sp), start));
                            }
                            None => {
                                // No room: back to unloaded, and no more
                                // loading until something is released.
                                tile.free(ctx);
                                self.nodes[r.node].state = State::Unloaded;
                                self.atlas_full = true;
                                continue;
                            }
                        }
                    }
                    if largest > self.iota.map_or(0, |(_, n)| n) {
                        // Grown in steps so a run of ever so slightly larger
                        // tiles is not a rebuild each.
                        let n = largest.next_power_of_two();
                        if let Some((old, _)) = self.iota.take() {
                            ctx.delete_buffer(old);
                        }
                        let idx: Vec<u32> = (0..n as u32).collect();
                        self.iota = Some((ctx.new_buffer(BufferType::IndexBuffer, BufferUsage::Immutable, BufferSource::slice(&idx)), n));
                    }
                    uploaded += tile.bytes;
                    self.stats.gpu_bytes += tile.bytes;
                    self.stats.loaded += 1;
                    self.nodes[r.node].state = State::Loaded(tile);
                }
                Err(e) => {
                    self.stats.failed += 1;
                    if self.stats.first_error.is_none() {
                        ::log::error!("{}: {e}", self.path.display());
                        self.stats.first_error = Some(e);
                    }
                    self.nodes[r.node].state = State::Failed;
                }
            }
        }
        if kinds_changed {
            (self.has_points, self.has_mesh, self.has_splats) = content_kinds(&self.nodes);
        }

        self.render.clear();
        self.wishes.clear();
        if self.visible {
            let view = View { planes: extract_frustum_planes(view_proj), view_proj: *view_proj, mpp: mpp.max(1e-9), detail, frame };
            self.visit(0, &view, Kind::Unknown);
        }

        // The splats among the tiles selected, sorted together.
        if let Some(atlas) = &mut self.atlas {
            let sel: Vec<(usize, usize, Arc<SplatTexels>)> = self
                .render
                .iter()
                .filter_map(|&i| match &self.nodes[i].state {
                    State::Loaded(GpuTile { splats: Some((t, start)), .. }) => Some((i, *start, t.clone())),
                    _ => None,
                })
                .collect();
            atlas.update(ctx, front, &sel);
        }

        // Evict the least recently used tiles not drawn this frame.
        if self.stats.gpu_bytes > budget_bytes {
            let mut cands: Vec<(u64, usize)> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, n)| matches!(n.state, State::Loaded(_)) && n.last_used != frame && !self.atlas.as_ref().is_some_and(|a| a.protects(*i)))
                .map(|(i, n)| (n.last_used, i))
                .collect();
            cands.sort_unstable();
            for (_, i) in cands {
                if self.stats.gpu_bytes <= budget_bytes {
                    break;
                }
                self.release(ctx, i);
            }
        }

        // Then load, unless what is on screen already fills the budget: a
        // detail setting the budget cannot hold stops at the coarser levels,
        // which are asked for first, instead of running the GPU out of memory.
        self.stats.budget_full = self.stats.gpu_bytes > budget_bytes || self.atlas_full;
        // Coarse tiles first — they cover the most screen per byte — then
        // the ones nearest the middle of the view.
        self.wishes.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let cap = self.workers * 2;
        if let Some(tx) = &self.req_tx
            && !self.stats.budget_full
        {
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

        self.stats.drawn_tiles = self.render.len();
        let (mut tris, mut pts) = (0, 0);
        for &i in &self.render {
            if let State::Loaded(t) = &self.nodes[i].state {
                tris += t.triangles;
                pts += t.point_count;
            }
        }
        self.stats.drawn_triangles = tris;
        self.stats.drawn_points = pts;
        self.stats.drawn_splats = self.atlas.as_ref().map_or(0, SplatAtlas::shown);
    }

    /// Unload tile `i`, giving its atlas slots back.
    fn release(&mut self, ctx: &mut dyn RenderingBackend, i: usize) {
        if let State::Loaded(t) = std::mem::replace(&mut self.nodes[i].state, State::Unloaded) {
            self.stats.gpu_bytes -= t.bytes;
            self.stats.loaded -= 1;
            if let (Some((s, start)), Some(atlas)) = (&t.splats, &mut self.atlas) {
                atlas.free(*start, s.len());
                self.atlas_full = false;
            }
            t.free(ctx);
        }
    }

    /// Find atlas slots for a tile's splats and upload them: first fit, then
    /// growing the atlas, then releasing the least recently used splat tiles
    /// nothing on screen refers to.
    fn place_splats(&mut self, ctx: &mut dyn RenderingBackend, sp: &SplatTexels, frame: u64) -> Option<usize> {
        let n = sp.len();
        let atlas = self.atlas.get_or_insert_with(|| SplatAtlas::new(ctx));
        let mut start = atlas.alloc(n);
        if start.is_none() {
            let resident: Vec<(usize, Arc<SplatTexels>)> = self
                .nodes
                .iter()
                .filter_map(|nd| match &nd.state {
                    State::Loaded(GpuTile { splats: Some((t, s)), .. }) => Some((*s, t.clone())),
                    _ => None,
                })
                .collect();
            let atlas = self.atlas.as_mut().unwrap();
            if atlas.grow(ctx, n, &resident) {
                start = atlas.alloc(n);
            }
        }
        if start.is_none() {
            let mut cands: Vec<(u64, usize)> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, nd)| {
                    matches!(nd.state, State::Loaded(GpuTile { splats: Some(_), .. }))
                        && nd.last_used + 1 < frame
                        && !self.atlas.as_ref().unwrap().protects(*i)
                })
                .map(|(i, nd)| (nd.last_used, i))
                .collect();
            cands.sort_unstable();
            for (_, i) in cands {
                self.release(ctx, i);
                start = self.atlas.as_mut().unwrap().alloc(n);
                if start.is_some() {
                    break;
                }
            }
        }
        let start = start?;
        self.atlas.as_ref().unwrap().upload(ctx, start, sp);
        Some(start)
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
    fn visit(&mut self, i: usize, view: &View, inherited: Kind) -> bool {
        if aabb_outside_frustum(&view.planes, &self.nodes[i].lo, &self.nodes[i].hi) {
            return true;
        }
        if !self.nodes[i].external.is_empty() {
            self.graft_external(i);
        }
        let n = &self.nodes[i];
        let has_content = !matches!(n.state, State::Empty);
        let has_children = !n.children.is_empty();
        // A tile not loaded yet is taken to hold what its parent holds.
        let kind = if n.kind == Kind::Unknown { inherited } else { n.kind };
        let max_sse = match kind {
            Kind::Points => view.detail.points,
            Kind::Splats => view.detail.splats,
            Kind::Mesh | Kind::Unknown => view.detail.mesh,
        };
        let refine = has_children && (!has_content || n.geometric_error / view.mpp > max_sse);
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
                complete &= self.visit(c, view, kind);
            }
            return complete;
        }

        let mark = self.render.len();
        let mut complete = true;
        for c in children {
            complete &= self.visit(c, view, kind);
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
                let esri = self.esri_crs.is_some();
                Parser { nodes: &mut self.nodes, frame, esri }.tile(&json["root"], &parent_transform, refine, &base_dir, depth)
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

    /// Draw the point content selected this frame, inside the cloud's point
    /// pass. `depth_only` is EDL's depth replay, as for the clouds.
    pub fn draw_points(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, point_size: f32, z_clip: ZClip, depth_only: bool) {
        let Some((iota, _)) = self.iota else { return };
        if !self.visible || self.render.is_empty() {
            return;
        }
        // EDL reads height as depth. Measured from the tileset's own bottom,
        // over its own span, as each cloud does over its own.
        let root = &self.nodes[0];
        let span = root.hi[2] - root.lo[2];
        let (depth_z0, depth_range) = if span.is_finite() && span > 0.0 && span < 1.0e6 { (root.lo[2], span) } else { (0.0, 0.0) };
        ctx.apply_pipeline(if depth_only { &self.point_depth_pipeline } else { &self.point_pipeline });
        ctx.apply_uniforms(UniformsSource::table(&point_shader::Uniforms {
            mvp: *view_proj,
            point_size,
            depth_z0,
            depth_range,
            z_clip_min: z_clip.min,
            z_clip_max: z_clip.max,
        }));
        for &i in &self.render {
            let State::Loaded(tile) = &self.nodes[i].state else { continue };
            for p in &tile.points {
                ctx.apply_bindings(&Bindings { vertex_buffers: vec![p.pos, p.col], index_buffer: iota, images: vec![] });
                ctx.draw(0, p.count, 1);
            }
        }
    }

    /// Draw the splats of the tiles on screen, blended over everything drawn
    /// so far. `viewport` is the view's size in physical pixels.
    pub fn draw_splats(&self, ctx: &mut dyn RenderingBackend, view_proj: &glam::Mat4, viewport: (f32, f32), z_clip: ZClip) {
        if let Some(atlas) = &self.atlas
            && self.visible
        {
            atlas.draw(ctx, view_proj, viewport, z_clip);
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
        if let Some((b, _)) = self.iota {
            ctx.delete_buffer(b);
        }
        if let Some(a) = self.atlas {
            a.unload(ctx);
        }
        ctx.delete_texture(self.white);
        ctx.delete_pipeline(self.pipeline);
        ctx.delete_pipeline(self.point_pipeline);
        ctx.delete_pipeline(self.point_depth_pipeline);
    }
}

/// A parsed tileset, waiting for the viewer origin.
pub struct TilesetDesc {
    pub path: PathBuf,
    nodes: Vec<Node>,
    pub ecef: bool,
    esri_crs: Option<String>,
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

/// Whether any tile is known to hold points, a mesh, splats.
fn content_kinds(nodes: &[Node]) -> (bool, bool, bool) {
    let any = |k: Kind| nodes.iter().any(|n| n.kind == k);
    (any(Kind::Points), any(Kind::Mesh), any(Kind::Splats))
}

/// The projected CRS a tileset's `ESRI_crs` names, as "EPSG:28992+5709".
/// Geographic and geocentric ones are left to the ECEF path: their numbers
/// are not metres.
fn esri_projected_crs(json: &Value) -> Option<String> {
    let e = json.get("extensions")?.get("ESRI_crs")?;
    let wkid = e.get("wkid").or_else(|| e.get("latestWkid")).and_then(Value::as_u64)?;
    if matches!(wkid, 4326 | 4258 | 4979 | 4978 | 4937 | 4936 | 4269 | 4230) {
        return None;
    }
    Some(match e.get("vcsWkid").or_else(|| e.get("latestVcsWkid")).and_then(Value::as_u64) {
        Some(v) => format!("EPSG:{wkid}+{v}"),
        None => format!("EPSG:{wkid}"),
    })
}

fn info_tiles(path: &Path, nodes: &[Node], ecef: bool, esri: Option<&str>) {
    let with_content = nodes.iter().filter(|n| !n.uris.is_empty()).count();
    let external = nodes.iter().filter(|n| !n.external.is_empty()).count();
    ::log::info!(
        "3D Tiles {}: {} tiles ({} with content, {} external tilesets){}",
        path.display(),
        nodes.len(),
        with_content,
        external,
        match (ecef, esri) {
            (true, _) => ", ECEF — shown in a local east-north-up frame".to_string(),
            (false, Some(c)) => format!(", placed in {c} by ESRI_crs"),
            _ => String::new(),
        }
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

/// The cloud's point pipeline, for tile content: same depth rules, same
/// point size, colour as bytes rather than floats.
fn point_pipeline(ctx: &mut dyn RenderingBackend, depth_only: bool) -> Pipeline {
    // gl_PointSize is ignored on desktop GL without GL_PROGRAM_POINT_SIZE.
    // The cloud pipeline enables it too, but a tileset may be all there is.
    unsafe { miniquad::gl::glEnable(0x8642) };
    let shader = ctx
        .new_shader(ShaderSource::Glsl { vertex: point_shader::VERTEX, fragment: point_shader::FRAGMENT }, point_shader::meta())
        .expect("tile point shader");
    ctx.new_pipeline(
        &[BufferLayout::default(), BufferLayout::default()],
        &[
            VertexAttribute::with_buffer("pos", VertexFormat::Float3, 0),
            VertexAttribute::with_buffer("color", VertexFormat::Byte4, 1),
        ],
        shader,
        PipelineParams {
            primitive_type: PrimitiveType::Points,
            depth_test: Comparison::LessOrEqual,
            depth_write: true,
            color_write: if depth_only { (false, false, false, false) } else { (true, true, true, true) },
            ..Default::default()
        },
    )
}

/// `raw_miniquad::shader` with byte colours and a depth origin. The alpha it
/// writes is the height EDL shades by, as the cloud's does.
mod point_shader {
    use miniquad::*;

    pub const VERTEX: &str = r#"#version 100
        attribute vec3 pos;
        attribute vec4 color;

        uniform mat4 mvp;
        uniform float point_size;
        uniform float depth_z0;
        uniform float depth_range;
        uniform float z_clip_min;
        uniform float z_clip_max;

        varying lowp vec3 v_color;
        varying mediump float v_depth;

        void main() {
            if (pos.z < z_clip_min || pos.z > z_clip_max) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                gl_PointSize = 0.0;
                v_color = vec3(0.0);
                v_depth = 1.0;
                return;
            }
            gl_Position = mvp * vec4(pos, 1.0);
            gl_PointSize = point_size;
            // Byte attributes arrive unnormalised.
            v_color = color.rgb / 255.0;
            mediump float t = (depth_range > 0.0) ? clamp((pos.z - depth_z0) / depth_range, 0.0, 1.0) : 0.5;
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
                    UniformDesc::new("depth_z0", UniformType::Float1),
                    UniformDesc::new("depth_range", UniformType::Float1),
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
        pub depth_z0: f32,
        pub depth_range: f32,
        pub z_clip_min: f32,
        pub z_clip_max: f32,
    }
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
            let pts = m.points.iter().flat_map(|p| p.positions.iter()).copied();
            let spl = m.splats.iter().flat_map(|s| s.t0.iter()).map(|t| [t[0], t[1], t[2]]);
            for p in m.primitives.iter().flat_map(|p| p.positions.iter()).copied().chain(pts).chain(spl) {
                count += 1;
                let v = DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                if v.cmplt(blo - 0.01).any() || v.cmpgt(bhi + 0.01).any() {
                    outside += 1;
                }
            }
            println!(
                "depth {} {}: {} vertices, {} triangles, {} points, {} splats, {} images, {} outside the box, decoded in {:?}",
                n.depth,
                n.uris[0].display(),
                count,
                m.triangle_count(),
                m.point_count(),
                m.splat_count(),
                m.images.len(),
                outside,
                t.elapsed()
            );
            assert_eq!(outside, 0);
        }
    }

    #[test]
    fn esri_crs_twins_place_an_ecef_tileset_in_rd() {
        let d = scratch("esri");
        let p = d.join("tileset.json");
        std::fs::write(
            &p,
            r#"{"asset":{"version":"1.1"},"geometricError":10,"extensions":{"ESRI_crs":{"wkid":28992,"vcsWkid":5709}},
               "root":{"boundingVolume":{"box":[3916197,405557,5001182,60,0,0,0,60,0,0,0,50]},"geometricError":8,
                 "extensions":{"ESRI_crs":{"boundingVolume":{"box":[191086,443150,22,55,0,0,0,64,0,0,0,19]}}},
                 "children":[{"boundingVolume":{"box":[0,0,0,10,0,0,0,10,0,0,0,10]},"geometricError":0,
                   "transform":[0.1,0.99,0,0,-0.78,-0.07,0.61,0,0.61,0.06,0.78,0,3916197,405557,5001182,1],
                   "extensions":{"ESRI_crs":{"boundingVolume":{"box":[0,0,0,10,0,0,0,10,0,0,0,10]},
                     "transform":[1,0,0,0,0,1,0,0,0,0,1,0,191086,443150,22,1]}},"content":{"uri":"a.glb"}}]}}"#,
        )
        .unwrap();
        let desc = Tileset::open(&p).unwrap();
        assert!(!desc.ecef);
        assert_eq!(desc.esri_crs.as_deref(), Some("EPSG:28992+5709"));
        assert_eq!(desc.world_bounds(), (DVec3::new(191031.0, 443086.0, 3.0), DVec3::new(191141.0, 443214.0, 41.0)));
        assert_eq!(desc.nodes[1].world_lo, DVec3::new(191076.0, 443140.0, 12.0));
        assert_eq!(desc.nodes[1].transform.w_axis.truncate(), DVec3::new(191086.0, 443150.0, 22.0));
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
