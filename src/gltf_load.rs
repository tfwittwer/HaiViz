//! Decoding 3D Tiles content into triangle meshes, CPU side.
//!
//! Content is glTF 2.0 — binary (GLB) or JSON — as 3D Tiles 1.1 has it, or
//! one of the 1.0 wrappers around it that tilesets in the wild still carry:
//! `b3dm` (a GLB behind a feature table that may hold `RTC_CENTER`) and
//! `cmpt` (several tiles back to back). Instanced (`i3dm`) and point-cloud
//! (`pnts`) content are refused with a reason rather than drawn wrong.
//!
//! Everything a worker thread can do happens here: the vertices leave in
//! viewer space (world minus the session's viewer origin) as `f32`, already
//! through the whole transform chain, and textures leave decoded to RGBA8, so
//! all the main thread has left is the upload.
//!
//! The decoder is hand-rolled rather than the `gltf` crate: that crate
//! refuses a file whose `extensionsRequired` it does not know, which is most
//! tilesets (`KHR_mesh_quantization`, `CESIUM_RTC`, metadata extensions), and
//! reading accessors generically — any component type, normalised or not,
//! strided, sparse — is what makes quantised meshes work for free.

use std::path::{Path, PathBuf};

use glam::{DMat3, DMat4, DQuat, DVec3};
use serde_json::Value;

/// The fixed rotation 3D Tiles puts between glTF's y-up and the tileset's
/// z-up: glTF (x, y, z) lands at (x, −z, y).
pub const Y_UP_TO_Z_UP: DMat4 = DMat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
]);

pub struct CpuImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    /// Whether the sampler asks for repeat rather than clamp. Photogrammetry
    /// atlases clamp; a tiled façade texture repeats.
    pub repeat: bool,
}

pub struct CpuPrimitive {
    /// Viewer space.
    pub positions: Vec<[f32; 3]>,
    /// World space, unit length. Zero-filled when the primitive has none.
    pub normals: Vec<[f32; 3]>,
    /// sRGB. The material's factor is kept separately, in `base_color`.
    pub colors: Vec<[u8; 4]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    /// sRGB-encoded base colour factor, alpha linear.
    pub base_color: [f32; 4],
    /// Index into [`CpuMesh::images`].
    pub texture: Option<usize>,
    pub unlit: bool,
    pub has_normals: bool,
    /// `alphaMode: MASK` cut-off; `None` draws opaque.
    pub alpha_cutoff: Option<f32>,
}

#[derive(Default)]
pub struct CpuMesh {
    pub primitives: Vec<CpuPrimitive>,
    pub images: Vec<CpuImage>,
    /// Things that were dropped rather than drawn wrong — an undecodable
    /// texture, a line primitive. The tile still shows.
    pub warnings: Vec<String>,
}

impl CpuMesh {
    pub fn triangle_count(&self) -> usize {
        self.primitives.iter().map(|p| p.indices.len() / 3).sum()
    }

    fn append(&mut self, mut other: CpuMesh) {
        let base = self.images.len();
        for p in &mut other.primitives {
            if let Some(t) = p.texture.as_mut() {
                *t += base;
            }
        }
        self.images.append(&mut other.images);
        self.primitives.append(&mut other.primitives);
        self.warnings.append(&mut other.warnings);
    }
}

/// glTF colours are linear light; the framebuffer is not. Applied once, on the
/// CPU, so the shader can multiply plain sRGB values.
pub fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Read one content file and turn it into meshes in viewer space.
///
/// `tile_to_world` is the tile's accumulated transform (including any ENU
/// frame the tileset was put in); `origin` the session's viewer origin.
pub fn load_content(path: &Path, tile_to_world: &DMat4, origin: [f64; 3]) -> Result<CpuMesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let base_dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    decode_content(&bytes, &base_dir, tile_to_world, origin)
}

/// As [`load_content`], from bytes already read. `base_dir` resolves external
/// buffers and images of a JSON glTF.
pub fn decode_content(bytes: &[u8], base_dir: &Path, tile_to_world: &DMat4, origin: [f64; 3]) -> Result<CpuMesh, String> {
    if bytes.len() < 4 {
        return Err("content is empty".into());
    }
    match &bytes[0..4] {
        b"glTF" => decode_glb(bytes, base_dir, tile_to_world, DVec3::ZERO, origin),
        b"b3dm" => decode_b3dm(bytes, base_dir, tile_to_world, origin),
        b"cmpt" => decode_cmpt(bytes, base_dir, tile_to_world, origin),
        b"i3dm" => Err("instanced content (i3dm) is not supported".into()),
        b"pnts" => Err("point-cloud content (pnts) is not supported".into()),
        _ => {
            let first = bytes.iter().find(|b| !b.is_ascii_whitespace()).copied();
            if first == Some(b'{') {
                let json: Value = serde_json::from_slice(bytes).map_err(|e| format!("glTF JSON: {e}"))?;
                Gltf::new(json, None, base_dir)?.to_mesh(tile_to_world, DVec3::ZERO, origin)
            } else {
                Err(format!("unknown content type (magic {:?})", String::from_utf8_lossy(&bytes[0..4])))
            }
        }
    }
}

fn u32_at(b: &[u8], o: usize) -> Result<u32, String> {
    b.get(o..o + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "truncated header".to_string())
}

fn decode_glb(bytes: &[u8], base_dir: &Path, tile_to_world: &DMat4, rtc: DVec3, origin: [f64; 3]) -> Result<CpuMesh, String> {
    let version = u32_at(bytes, 4)?;
    if version != 2 {
        return Err(format!("glTF {version} is not supported, only 2.0"));
    }
    let total = (u32_at(bytes, 8)? as usize).min(bytes.len());
    let mut json: Option<Value> = None;
    let mut bin: Option<&[u8]> = None;
    let mut off = 12;
    while off + 8 <= total {
        let len = u32_at(bytes, off)? as usize;
        let kind = u32_at(bytes, off + 4)?;
        let data = bytes.get(off + 8..off + 8 + len).ok_or("GLB chunk runs past the end of the file")?;
        match kind {
            0x4E4F534A => json = Some(serde_json::from_slice(data).map_err(|e| format!("GLB JSON: {e}"))?),
            0x004E4942 => bin = Some(data),
            _ => {}
        }
        off += 8 + len;
    }
    let json = json.ok_or("GLB has no JSON chunk")?;
    Gltf::new(json, bin, base_dir)?.to_mesh(tile_to_world, rtc, origin)
}

fn decode_b3dm(bytes: &[u8], base_dir: &Path, tile_to_world: &DMat4, origin: [f64; 3]) -> Result<CpuMesh, String> {
    let ft_json = u32_at(bytes, 12)? as usize;
    let ft_bin = u32_at(bytes, 16)? as usize;
    let bt_json = u32_at(bytes, 20)? as usize;
    let bt_bin = u32_at(bytes, 24)? as usize;
    // The two legacy headers put batchLength where the table lengths are now.
    // Their tell is a "length" that is really the first bytes of JSON.
    if bt_json >= 570_425_344 || bt_bin >= 570_425_344 {
        return Err("legacy b3dm header is not supported".into());
    }
    let mut rtc = DVec3::ZERO;
    if ft_json > 0
        && let Some(s) = bytes.get(28..28 + ft_json)
        && let Ok(ft) = serde_json::from_slice::<Value>(s)
        && let Some(c) = ft.get("RTC_CENTER").and_then(Value::as_array)
        && c.len() == 3
    {
        rtc = DVec3::new(
            c[0].as_f64().unwrap_or(0.0),
            c[1].as_f64().unwrap_or(0.0),
            c[2].as_f64().unwrap_or(0.0),
        );
    }
    let glb_start = 28 + ft_json + ft_bin + bt_json + bt_bin;
    let glb = bytes.get(glb_start..).ok_or("b3dm has no glTF")?;
    if glb.len() < 4 || &glb[0..4] != b"glTF" {
        return Err("b3dm does not embed a binary glTF".into());
    }
    decode_glb(glb, base_dir, tile_to_world, rtc, origin)
}

fn decode_cmpt(bytes: &[u8], base_dir: &Path, tile_to_world: &DMat4, origin: [f64; 3]) -> Result<CpuMesh, String> {
    let n = u32_at(bytes, 12)? as usize;
    let mut out = CpuMesh::default();
    let mut off = 16;
    for _ in 0..n {
        let len = u32_at(bytes, off + 8)? as usize;
        let inner = bytes.get(off..off + len).ok_or("cmpt inner tile runs past the end")?;
        match decode_content(inner, base_dir, tile_to_world, origin) {
            Ok(m) => out.append(m),
            Err(e) => out.warnings.push(format!("cmpt inner tile skipped: {e}")),
        }
        off += len;
    }
    Ok(out)
}

/// Required extensions that change geometry in ways this decoder cannot
/// follow. Anything else required is either supported or only adds data.
const UNSUPPORTED_REQUIRED: &[&str] = &[
    "KHR_draco_mesh_compression",
    "EXT_meshopt_compression",
    "KHR_meshopt_compression",
    "EXT_mesh_gpu_instancing",
];

struct Gltf<'a> {
    json: Value,
    buffers: Vec<std::borrow::Cow<'a, [u8]>>,
    base_dir: PathBuf,
}

fn arr<'v>(v: &'v Value, key: &str) -> &'v [Value] {
    v.get(key).and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[])
}

fn usize_of(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}

fn f64s<const N: usize>(v: Option<&Value>) -> Option<[f64; N]> {
    let a = v?.as_array()?;
    if a.len() != N {
        return None;
    }
    let mut out = [0.0; N];
    for (o, x) in out.iter_mut().zip(a) {
        *o = x.as_f64()?;
    }
    Some(out)
}

fn decode_base64(s: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a') as u32 + 26),
            b'0'..=b'9' => Some((c - b'0') as u32 + 52),
            b'+' | b'-' => Some(62),
            b'/' | b'_' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0;
    for &c in s.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        acc = (acc << 6) | val(c).ok_or("bad base64 in data URI")?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Bytes behind a `uri`: an embedded `data:` URI or a file next to the glTF.
fn read_uri(uri: &str, base_dir: &Path) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let comma = rest.find(',').ok_or("malformed data URI")?;
        if !rest[..comma].ends_with(";base64") {
            return Err("only base64 data URIs are supported".into());
        }
        return decode_base64(&rest[comma + 1..]);
    }
    let p = base_dir.join(percent_decode(uri));
    std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
}

/// One accessor, read out as `f32`s, `ncomp` per element.
struct Attr {
    data: Vec<f32>,
    ncomp: usize,
}

impl Attr {
    fn len(&self) -> usize {
        self.data.len() / self.ncomp.max(1)
    }
    fn get<const N: usize>(&self, i: usize, fill: [f32; N]) -> [f32; N] {
        let mut out = fill;
        let base = i * self.ncomp;
        for (k, o) in out.iter_mut().enumerate().take(self.ncomp) {
            *o = self.data[base + k];
        }
        out
    }
}

fn component_size(ct: u64) -> Result<usize, String> {
    match ct {
        5120 | 5121 => Ok(1),
        5122 | 5123 => Ok(2),
        5125 | 5126 => Ok(4),
        _ => Err(format!("unknown accessor componentType {ct}")),
    }
}

fn read_component(b: &[u8], ct: u64, normalized: bool) -> f32 {
    match ct {
        5120 => {
            let v = b[0] as i8 as f32;
            if normalized { (v / 127.0).max(-1.0) } else { v }
        }
        5121 => {
            let v = b[0] as f32;
            if normalized { v / 255.0 } else { v }
        }
        5122 => {
            let v = i16::from_le_bytes([b[0], b[1]]) as f32;
            if normalized { (v / 32767.0).max(-1.0) } else { v }
        }
        5123 => {
            let v = u16::from_le_bytes([b[0], b[1]]) as f32;
            if normalized { v / 65535.0 } else { v }
        }
        5125 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
    }
}

fn type_components(t: &str) -> Result<usize, String> {
    match t {
        "SCALAR" => Ok(1),
        "VEC2" => Ok(2),
        "VEC3" => Ok(3),
        "VEC4" => Ok(4),
        "MAT2" => Ok(4),
        "MAT3" => Ok(9),
        "MAT4" => Ok(16),
        _ => Err(format!("unknown accessor type {t}")),
    }
}

impl<'a> Gltf<'a> {
    fn new(json: Value, bin: Option<&'a [u8]>, base_dir: &Path) -> Result<Gltf<'a>, String> {
        for ext in arr(&json, "extensionsRequired") {
            if let Some(name) = ext.as_str()
                && UNSUPPORTED_REQUIRED.contains(&name)
            {
                return Err(format!("{name} is not supported"));
            }
        }
        let mut buffers = Vec::new();
        for (i, b) in arr(&json, "buffers").iter().enumerate() {
            match b.get("uri").and_then(Value::as_str) {
                Some(uri) => buffers.push(std::borrow::Cow::Owned(read_uri(uri, base_dir)?)),
                None if i == 0 => buffers.push(std::borrow::Cow::Borrowed(bin.ok_or("GLB buffer without a BIN chunk")?)),
                None => return Err(format!("buffer {i} has no uri")),
            }
        }
        Ok(Gltf { json, buffers, base_dir: base_dir.to_path_buf() })
    }

    fn view_bytes(&self, view_idx: usize) -> Result<(&[u8], Option<usize>), String> {
        let view = arr(&self.json, "bufferViews").get(view_idx).ok_or("bufferView out of range")?;
        if view.get("extensions").and_then(|e| e.get("EXT_meshopt_compression")).is_some()
            || view.get("extensions").and_then(|e| e.get("KHR_meshopt_compression")).is_some()
        {
            return Err("EXT_meshopt_compression is not supported".into());
        }
        let buf = usize_of(view, "buffer").ok_or("bufferView without buffer")?;
        let buf = self.buffers.get(buf).ok_or("buffer out of range")?;
        let off = usize_of(view, "byteOffset").unwrap_or(0);
        let len = usize_of(view, "byteLength").ok_or("bufferView without byteLength")?;
        let bytes = buf.get(off..off + len).ok_or("bufferView runs past its buffer")?;
        Ok((bytes, usize_of(view, "byteStride")))
    }

    fn accessor(&self, idx: usize) -> Result<Attr, String> {
        let acc = arr(&self.json, "accessors").get(idx).ok_or("accessor out of range")?;
        let ct = acc.get("componentType").and_then(Value::as_u64).ok_or("accessor without componentType")?;
        let csize = component_size(ct)?;
        let ncomp = type_components(acc.get("type").and_then(Value::as_str).unwrap_or(""))?;
        let count = usize_of(acc, "count").ok_or("accessor without count")?;
        let normalized = acc.get("normalized").and_then(Value::as_bool).unwrap_or(false);
        let mut data = vec![0.0f32; count * ncomp];
        if let Some(view_idx) = usize_of(acc, "bufferView") {
            let (bytes, stride) = self.view_bytes(view_idx)?;
            let off = usize_of(acc, "byteOffset").unwrap_or(0);
            let elem = csize * ncomp;
            let stride = stride.unwrap_or(elem).max(elem);
            if count > 0 && off + (count - 1) * stride + elem > bytes.len() {
                return Err("accessor runs past its bufferView".into());
            }
            for i in 0..count {
                let e = off + i * stride;
                for k in 0..ncomp {
                    data[i * ncomp + k] = read_component(&bytes[e + k * csize..], ct, normalized);
                }
            }
        }
        if let Some(sparse) = acc.get("sparse") {
            let n = usize_of(sparse, "count").ok_or("sparse accessor without count")?;
            let ind = sparse.get("indices").ok_or("sparse accessor without indices")?;
            let val = sparse.get("values").ok_or("sparse accessor without values")?;
            let ict = ind.get("componentType").and_then(Value::as_u64).ok_or("sparse indices without componentType")?;
            let isize = component_size(ict)?;
            let (ib, _) = self.view_bytes(usize_of(ind, "bufferView").ok_or("sparse indices without bufferView")?)?;
            let (vb, _) = self.view_bytes(usize_of(val, "bufferView").ok_or("sparse values without bufferView")?)?;
            let ioff = usize_of(ind, "byteOffset").unwrap_or(0);
            let voff = usize_of(val, "byteOffset").unwrap_or(0);
            if ioff + n * isize > ib.len() || voff + n * ncomp * csize > vb.len() {
                return Err("sparse accessor runs past its bufferView".into());
            }
            for s in 0..n {
                let target = read_component(&ib[ioff + s * isize..], ict, false) as usize;
                if target >= count {
                    return Err("sparse index out of range".into());
                }
                for k in 0..ncomp {
                    data[target * ncomp + k] = read_component(&vb[voff + (s * ncomp + k) * csize..], ct, normalized);
                }
            }
        }
        Ok(Attr { data, ncomp })
    }

    /// Indices read as integers. Through `accessor` they would pass through
    /// `f32`, which is exact only below 2²⁴.
    fn indices(&self, idx: usize) -> Result<Vec<u32>, String> {
        let acc = arr(&self.json, "accessors").get(idx).ok_or("index accessor out of range")?;
        let ct = acc.get("componentType").and_then(Value::as_u64).ok_or("accessor without componentType")?;
        let count = usize_of(acc, "count").ok_or("accessor without count")?;
        if acc.get("sparse").is_some() || usize_of(acc, "bufferView").is_none() {
            // Rare enough for the float path; sparse indices are unheard of.
            return Ok(self.accessor(idx)?.data.iter().map(|&v| v as u32).collect());
        }
        let csize = component_size(ct)?;
        let (bytes, stride) = self.view_bytes(usize_of(acc, "bufferView").unwrap())?;
        let off = usize_of(acc, "byteOffset").unwrap_or(0);
        let stride = stride.unwrap_or(csize).max(csize);
        if count > 0 && off + (count - 1) * stride + csize > bytes.len() {
            return Err("index accessor runs past its bufferView".into());
        }
        Ok((0..count)
            .map(|i| {
                let b = &bytes[off + i * stride..];
                match ct {
                    5121 => b[0] as u32,
                    5123 => u16::from_le_bytes([b[0], b[1]]) as u32,
                    _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                }
            })
            .collect())
    }

    fn node_local(node: &Value) -> DMat4 {
        if let Some(m) = f64s::<16>(node.get("matrix")) {
            return DMat4::from_cols_array(&m);
        }
        let t = f64s::<3>(node.get("translation")).unwrap_or([0.0; 3]);
        let r = f64s::<4>(node.get("rotation")).unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let s = f64s::<3>(node.get("scale")).unwrap_or([1.0; 3]);
        DMat4::from_scale_rotation_translation(
            DVec3::from(s),
            DQuat::from_xyzw(r[0], r[1], r[2], r[3]).normalize(),
            DVec3::from(t),
        )
    }

    /// `(mesh index, node-to-glTF-root matrix)` for every mesh instance in the
    /// default scene.
    fn mesh_instances(&self) -> Vec<(usize, DMat4)> {
        let nodes = arr(&self.json, "nodes");
        let scenes = arr(&self.json, "scenes");
        let scene = usize_of(&self.json, "scene").unwrap_or(0);
        let roots: Vec<usize> = match scenes.get(scene) {
            Some(s) => arr(s, "nodes").iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect(),
            None => {
                // No scene: every node that is nobody's child.
                let mut is_child = vec![false; nodes.len()];
                for n in nodes {
                    for c in arr(n, "children") {
                        if let Some(c) = c.as_u64().and_then(|c| is_child.get_mut(c as usize)) {
                            *c = true;
                        }
                    }
                }
                (0..nodes.len()).filter(|&i| !is_child[i]).collect()
            }
        };
        let mut out = Vec::new();
        let mut stack: Vec<(usize, DMat4, usize)> = roots.into_iter().map(|r| (r, DMat4::IDENTITY, 0)).collect();
        while let Some((i, parent, depth)) = stack.pop() {
            // A cycle is invalid glTF; the depth cap keeps it from hanging a worker.
            let Some(node) = nodes.get(i) else { continue };
            if depth > 64 {
                continue;
            }
            let m = parent * Self::node_local(node);
            if let Some(mesh) = usize_of(node, "mesh") {
                out.push((mesh, m));
            }
            for c in arr(node, "children") {
                if let Some(c) = c.as_u64() {
                    stack.push((c as usize, m, depth + 1));
                }
            }
        }
        out
    }

    fn decode_image(&self, image_idx: usize) -> Result<image::RgbaImage, String> {
        let img = arr(&self.json, "images").get(image_idx).ok_or("image out of range")?;
        let mime = img.get("mimeType").and_then(Value::as_str).unwrap_or("");
        if mime == "image/ktx2" {
            return Err("KTX2 / Basis textures are not supported".into());
        }
        let owned;
        let bytes: &[u8] = if let Some(v) = usize_of(img, "bufferView") {
            self.view_bytes(v)?.0
        } else if let Some(uri) = img.get("uri").and_then(Value::as_str) {
            owned = read_uri(uri, &self.base_dir)?;
            &owned
        } else {
            return Err("image has neither bufferView nor uri".into());
        };
        if bytes.starts_with(b"\xABKTX 20") {
            return Err("KTX2 / Basis textures are not supported".into());
        }
        image::load_from_memory(bytes).map(|i| i.to_rgba8()).map_err(|e| format!("texture: {e}"))
    }

    /// The image a texture draws from: plain `source`, or a WebP alternative
    /// the decoder can read when the plain one is missing.
    fn texture_source(tex: &Value) -> Option<usize> {
        usize_of(tex, "source").or_else(|| {
            tex.get("extensions")
                .and_then(|e| e.get("EXT_texture_webp"))
                .and_then(|w| usize_of(w, "source"))
        })
    }

    fn to_mesh(&self, tile_to_world: &DMat4, rtc: DVec3, origin: [f64; 3]) -> Result<CpuMesh, String> {
        let mut out = CpuMesh::default();
        // CESIUM_RTC is the glTF-side spelling of b3dm's RTC_CENTER.
        let rtc = match f64s::<3>(self.json.get("extensions").and_then(|e| e.get("CESIUM_RTC")).and_then(|r| r.get("center"))) {
            Some(c) => rtc + DVec3::from(c),
            None => rtc,
        };
        let to_tile = *tile_to_world * DMat4::from_translation(rtc) * Y_UP_TO_Z_UP;
        let origin = DVec3::from(origin);

        let materials = arr(&self.json, "materials");
        let textures = arr(&self.json, "textures");
        let samplers = arr(&self.json, "samplers");
        let meshes = arr(&self.json, "meshes");
        // glTF image index -> index in out.images, decoded on first use.
        let mut image_slot: std::collections::HashMap<usize, Option<usize>> = Default::default();

        for (mesh_idx, node_m) in self.mesh_instances() {
            let Some(mesh) = meshes.get(mesh_idx) else { continue };
            let full = to_tile * node_m;
            let lin = DMat3::from_mat4(full);
            let normal_m = lin.inverse().transpose();

            for prim in arr(mesh, "primitives") {
                if prim.get("extensions").and_then(|e| e.get("KHR_draco_mesh_compression")).is_some() {
                    return Err("KHR_draco_mesh_compression is not supported".into());
                }
                let mode = usize_of(prim, "mode").unwrap_or(4);
                if !(4..=6).contains(&mode) {
                    out.warnings.push(format!("primitive mode {mode} (points/lines) skipped"));
                    continue;
                }
                let attrs = prim.get("attributes").ok_or("primitive without attributes")?;
                let Some(pos_idx) = usize_of(attrs, "POSITION") else { continue };
                let pos = self.accessor(pos_idx)?;
                let n = pos.len();
                if n == 0 {
                    continue;
                }

                let material = usize_of(prim, "material").and_then(|m| materials.get(m));
                let pbr = material.and_then(|m| m.get("pbrMetallicRoughness"));
                let factor = f64s::<4>(pbr.and_then(|p| p.get("baseColorFactor"))).unwrap_or([1.0; 4]);
                let base_color = [
                    linear_to_srgb(factor[0] as f32),
                    linear_to_srgb(factor[1] as f32),
                    linear_to_srgb(factor[2] as f32),
                    factor[3] as f32,
                ];
                let unlit = material
                    .and_then(|m| m.get("extensions"))
                    .and_then(|e| e.get("KHR_materials_unlit"))
                    .is_some();
                let alpha_cutoff = match material.and_then(|m| m.get("alphaMode")).and_then(Value::as_str) {
                    Some("MASK") => Some(material.and_then(|m| m.get("alphaCutoff")).and_then(Value::as_f64).unwrap_or(0.5) as f32),
                    _ => None,
                };

                // Base colour texture, with its texCoord set and transform.
                let tex_info = pbr.and_then(|p| p.get("baseColorTexture"));
                let mut texture = None;
                let mut uv_set = 0;
                let mut uv_xform: Option<glam::DMat3> = None;
                if let Some(info) = tex_info {
                    uv_set = usize_of(info, "texCoord").unwrap_or(0);
                    if let Some(tt) = info.get("extensions").and_then(|e| e.get("KHR_texture_transform")) {
                        let o = f64s::<2>(tt.get("offset")).unwrap_or([0.0; 2]);
                        let s = f64s::<2>(tt.get("scale")).unwrap_or([1.0; 2]);
                        let r = tt.get("rotation").and_then(Value::as_f64).unwrap_or(0.0);
                        if let Some(t) = usize_of(tt, "texCoord") {
                            uv_set = t;
                        }
                        let (sr, cr) = r.sin_cos();
                        // T · R · S, the order the extension specifies.
                        let t_m = glam::DMat3::from_cols(DVec3::X, DVec3::Y, DVec3::new(o[0], o[1], 1.0));
                        let r_m = glam::DMat3::from_cols(DVec3::new(cr, -sr, 0.0), DVec3::new(sr, cr, 0.0), DVec3::Z);
                        let s_m = glam::DMat3::from_diagonal(DVec3::new(s[0], s[1], 1.0));
                        uv_xform = Some(t_m * r_m * s_m);
                    }
                    if let Some(tex) = usize_of(info, "index").and_then(|t| textures.get(t))
                        && let Some(src) = Self::texture_source(tex)
                    {
                        let slot = match image_slot.get(&src) {
                            Some(s) => *s,
                            None => {
                                let s = match self.decode_image(src) {
                                    Ok(img) => {
                                        let sampler = usize_of(tex, "sampler").and_then(|s| samplers.get(s));
                                        let wrap = |k: &str| sampler.and_then(|s| s.get(k)).and_then(Value::as_u64).unwrap_or(10497);
                                        let repeat = wrap("wrapS") != 33071 || wrap("wrapT") != 33071;
                                        out.images.push(CpuImage { width: img.width(), height: img.height(), rgba: img.into_raw(), repeat });
                                        Some(out.images.len() - 1)
                                    }
                                    Err(e) => {
                                        out.warnings.push(e);
                                        None
                                    }
                                };
                                image_slot.insert(src, s);
                                s
                            }
                        };
                        texture = slot;
                    }
                }

                let mut positions = Vec::with_capacity(n);
                for i in 0..n {
                    let p = pos.get::<3>(i, [0.0; 3]);
                    let w = full.transform_point3(DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)) - origin;
                    positions.push([w.x as f32, w.y as f32, w.z as f32]);
                }

                let normals_attr = usize_of(attrs, "NORMAL").map(|i| self.accessor(i)).transpose()?;
                let has_normals = normals_attr.as_ref().is_some_and(|a| a.len() == n);
                let normals = match normals_attr.filter(|a| a.len() == n) {
                    Some(a) => (0..n)
                        .map(|i| {
                            let v = a.get::<3>(i, [0.0, 0.0, 1.0]);
                            let w = (normal_m * DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).normalize_or_zero();
                            [w.x as f32, w.y as f32, w.z as f32]
                        })
                        .collect(),
                    None => vec![[0.0; 3]; n],
                };

                let colors = match usize_of(attrs, "COLOR_0").map(|i| self.accessor(i)).transpose()? {
                    Some(a) if a.len() == n => (0..n)
                        .map(|i| {
                            let c = a.get::<4>(i, [1.0; 4]);
                            [
                                (linear_to_srgb(c[0]) * 255.0 + 0.5) as u8,
                                (linear_to_srgb(c[1]) * 255.0 + 0.5) as u8,
                                (linear_to_srgb(c[2]) * 255.0 + 0.5) as u8,
                                (c[3].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                            ]
                        })
                        .collect(),
                    _ => vec![[255; 4]; n],
                };

                let uvs = match (texture, usize_of(attrs, &format!("TEXCOORD_{uv_set}"))) {
                    (Some(_), Some(i)) => {
                        let a = self.accessor(i)?;
                        if a.len() == n {
                            (0..n)
                                .map(|i| {
                                    let t = a.get::<2>(i, [0.0; 2]);
                                    match uv_xform {
                                        Some(m) => {
                                            let v = m * DVec3::new(t[0] as f64, t[1] as f64, 1.0);
                                            [v.x as f32, v.y as f32]
                                        }
                                        None => t,
                                    }
                                })
                                .collect()
                        } else {
                            vec![[0.0; 2]; n]
                        }
                    }
                    _ => vec![[0.0; 2]; n],
                };
                // A texture with no coordinates to read it by is dropped
                // rather than smeared from one corner texel.
                let texture = texture.filter(|_| usize_of(attrs, &format!("TEXCOORD_{uv_set}")).is_some());

                let raw = match usize_of(prim, "indices") {
                    Some(i) => self.indices(i)?,
                    None => (0..n as u32).collect(),
                };
                if raw.iter().any(|&i| i as usize >= n) {
                    return Err("index out of range of the vertices".into());
                }
                let indices = match mode {
                    4 => {
                        let mut v = raw;
                        v.truncate(v.len() / 3 * 3);
                        v
                    }
                    5 => {
                        let mut v = Vec::with_capacity(raw.len().saturating_sub(2) * 3);
                        for k in 2..raw.len() {
                            if k % 2 == 0 {
                                v.extend_from_slice(&[raw[k - 2], raw[k - 1], raw[k]]);
                            } else {
                                v.extend_from_slice(&[raw[k - 1], raw[k - 2], raw[k]]);
                            }
                        }
                        v
                    }
                    _ => {
                        let mut v = Vec::with_capacity(raw.len().saturating_sub(2) * 3);
                        for k in 2..raw.len() {
                            v.extend_from_slice(&[raw[0], raw[k - 1], raw[k]]);
                        }
                        v
                    }
                };
                if indices.is_empty() {
                    continue;
                }

                out.primitives.push(CpuPrimitive {
                    positions,
                    normals,
                    colors,
                    uvs,
                    indices,
                    base_color,
                    texture,
                    unlit,
                    has_normals,
                    alpha_cutoff,
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-triangle GLB, laid out by hand: positions f32, colours
    /// normalised u16 (as `haires amesh` writes them), u16 indices.
    fn triangle_glb(extra_json: &str) -> Vec<u8> {
        let mut bin = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, -3.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for _ in 0..3 {
            for c in [65535u16, 0, 0, 65535] {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2, 0] {
            bin.extend_from_slice(&i.to_le_bytes()); // padded to 4
        }
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],
            "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"COLOR_0":1}},"indices":2,"material":0}}]}}],
            "materials":[{{"extensions":{{"KHR_materials_unlit":{{}}}}}}],
            "accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}},
                         {{"bufferView":1,"componentType":5123,"normalized":true,"count":3,"type":"VEC4"}},
                         {{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}}],
            "bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}},{{"buffer":0,"byteOffset":36,"byteLength":24}},
                           {{"buffer":0,"byteOffset":60,"byteLength":6}}],
            "buffers":[{{"byteLength":{}}}]{extra_json}}}"#,
            bin.len()
        );
        let mut json = json.into_bytes();
        while json.len() % 4 != 0 {
            json.push(b' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"glTF");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&((12 + 8 + json.len() + 8 + bin.len()) as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E4942u32.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }

    #[test]
    fn glb_turns_y_up_into_z_up_and_applies_transform_and_origin() {
        let glb = triangle_glb("");
        let t = DMat4::from_translation(DVec3::new(1000.0, 2000.0, 10.0));
        let m = decode_content(&glb, Path::new("."), &t, [1000.0, 2000.0, 0.0]).unwrap();
        assert_eq!(m.primitives.len(), 1);
        let p = &m.primitives[0];
        // glTF (0, 2, -3) is (0, 3, 2) z-up, then +10 in z.
        assert_eq!(p.positions[2], [0.0, 3.0, 12.0]);
        assert_eq!(p.positions[1], [1.0, 0.0, 10.0]);
        assert_eq!(p.indices, vec![0, 1, 2]);
        assert!(p.unlit);
        assert_eq!(p.colors[0], [255, 0, 0, 255]);
    }

    #[test]
    fn b3dm_rtc_center_is_added_in_the_tile_frame() {
        let glb = triangle_glb("");
        let ft = br#"{"BATCH_LENGTH":0,"RTC_CENTER":[5.0,6.0,7.0]}"#;
        let mut ftj = ft.to_vec();
        while (28 + ftj.len()) % 8 != 0 {
            ftj.push(b' ');
        }
        let mut b = Vec::new();
        b.extend_from_slice(b"b3dm");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&((28 + ftj.len() + glb.len()) as u32).to_le_bytes());
        b.extend_from_slice(&(ftj.len() as u32).to_le_bytes());
        b.extend_from_slice(&[0u8; 12]);
        b.extend_from_slice(&ftj);
        b.extend_from_slice(&glb);
        let m = decode_content(&b, Path::new("."), &DMat4::IDENTITY, [0.0; 3]).unwrap();
        assert_eq!(m.primitives[0].positions[0], [5.0, 6.0, 7.0]);
    }

    #[test]
    fn draco_is_refused_with_its_name() {
        let glb = triangle_glb(r#","extensionsRequired":["KHR_draco_mesh_compression"]"#);
        let e = decode_content(&glb, Path::new("."), &DMat4::IDENTITY, [0.0; 3]).err().unwrap();
        assert!(e.contains("draco"), "{e}");
    }

    #[test]
    fn linear_colours_are_encoded_for_display() {
        assert_eq!((linear_to_srgb(0.2159) * 255.0).round() as u8, 128);
        assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn base64_round_trip() {
        assert_eq!(decode_base64("aGFpdml6").unwrap(), b"haiviz");
        assert_eq!(decode_base64("aGFpdg==").unwrap(), b"haiv");
    }
}
