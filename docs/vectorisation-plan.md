# Manual vectorisation in HaiViz — implementation plan

Status: **phases 0–4 implemented and driven on real data; phase 6 implemented
but not yet driven; phase 5 is still plan only.** Written 2026-09-10, last
updated 2026-09-11.

See [§15](#15-where-this-stands) for what is done, what was learned from using
it, and what a next session would pick up.

Paths below are relative to the workspace root (`haicloudrs/`).

## 1. Scope

Draw 3D polylines directly on the point cloud, by hand, with the machine doing
the precision work:

- **Digitising mode** — click to place vertices, rubber band to the cursor,
  undo, finish, cancel. Works in the 3D view from any camera angle, and later
  in the section view.
- **Snapping** — each vertex is pulled onto real data: nearest point, a low or
  high percentile of Z within a radius, the previous vertex's height, or an
  existing polyline.
- **Edge following** — between two clicks the tool traces a real feature
  (kerb, breakline, wall foot, painted road marking) rather than a straight
  line.
- **Class visibility** — classes can be switched off in the Classes window, and
  everything the operator does then ignores them. This is not a side feature:
  it is the primary control over *what you snap to*.
- **File I/O** — read and write GeoPackage, keep the existing Salome Hydro
  ASCII format.

## 2. Decisions already taken

| # | Decision | Rationale |
|---|---|---|
| 1 | Percentile snapping, not raw min/max | Raw extrema on LiDAR are usually noise. Default 5th / 95th percentile, slider-exposed; 0 / 100 gives the raw behaviour back. |
| 2 | Snap radius on **modifier+wheel** | Plain wheel keeps zooming. Diverges from `SectionAdjustDepth`, deliberately — this is a tool you sit in for minutes. |
| 3 | Both COPC mitigations | Refuse the snap when LOD spacing exceeds the radius, *and* pin the octree nodes under the cursor. |
| 4 | GeoPackage read + write, hand-rolled, pure Rust | No new dependencies; see §7.2. |
| 5 | Salome ASCII stays | Read and write, alongside GeoPackage. |
| 6 | Digitising works from oblique views | Ray-snapped vertices come from real points and are unambiguous from any angle. Explicit exception to the `TOP_VIEW_HINT` rule. |
| 7 | `haiviz/src/gpkg.rs`, not a shared crate | Duplicates a little of `hqmatcher/src/gpkg.rs`; accepted. |
| 8 | Edge following in the 3D/oblique view only | Not offered in the section view. Section-view digitising still gets plain placement and snapping. |
| 9 | Digitised layers **auto-save** | Granularity is the completed line, not the vertex; see §8.2. |
| 10 | Class visibility is **not** persisted in `classes.txt` | A view setting, like the height filter. The documented `id;r;g;b;label` format stays valid. |
| 11 | Snap radius is absolute, default **0.2 m** | Predictable — a metre value means the same thing at every zoom level. |

## 3. What already exists

| Piece | Where | Role here |
|---|---|---|
| `Polyline` / `PolylineFile`, 3D + section drawing, screen-space pick | `haiviz/src/polylines.rs` | Data model and rendering. Load-only today: no writer, no mutation, no vertex-level pick. |
| Plan-view lasso interaction | `haiviz/src/main.rs:2891`, `:462` | The click / rubber-band / Backspace / Enter / Esc skeleton, already shared across all three loading modes. |
| `ground_tool_controls` | `haiviz/src/main.rs:1124` | The pattern for one tool panel called from three per-mode egui panels. |
| `find_nearest_point` ×3 | `haiviz/src/raw_miniquad.rs:673`, `:1101`, `:2246` | Screen picking — full linear scans, see §5. |
| `ZClip` | `haiviz/src/raw_miniquad.rs:27` | The doctrine class visibility copies: hidden means unpickable, not merely invisible. |
| `ClassOp` / `ClassEdits` | `haiviz/src/classify.rs` | Op model and sidecar. Extended in §6.3. |
| `apply_z_clip` | `haiviz/src/main.rs:612` | The pattern for pushing a view setting onto all three backends. |
| Minimal GPKG writer | `hqmatcher/src/gpkg.rs` | Reference for the GeoPackage table layout and WKB encoding. Not reused directly (C dependency, 2D only). |
| Counting-sort voxel grid | `haicloudrs/src/voxelgrid.rs` | In-house model to copy for the spatial index. haiviz does not depend on haicloudrs; port, don't link. |

## 4. Two facts that shape everything

**There is no spatial index.** Every existing query is a linear scan over all
resident points. That is fine for a click and fatal for snapping, which needs a
radius query every frame while the mouse moves. Work package A exists solely to
fix this and everything else depends on it.

**Everything is written three times.** `main.rs` has three per-mode blocks
(single-file, COPC, tile) that each render, draw overlays and build an egui
panel. Shared input handling already lives above them (`main.rs:2891`), and
`draw_3d_overlays` is called identically from all three. New code must route
through shared functions or it will be triplicated by default.

---

## 5. Work package A — spatial index

**New file: `haiviz/src/spatial.rs`.**

A 2D XY grid built by counting sort, same shape as `Voxelgrid::compute`:

- `origin`, `cell`, `nx`, `ny`
- `starts: Vec<u32>`, `idx: Vec<u32>` — CSR-style bucket layout
- per-cell `z_min`, `z_max`, and the point indices achieving them

Roughly 12 bytes per point, O(n) build.

Query API:

- `nearest_ray(origin, dir, max_screen_dist) -> Option<usize>`
- `percentile_in_radius(x, y, r, pct) -> Option<(usize, f64)>`
- `points_in_radius(x, y, r, out: &mut Vec<u32>)`
- `cells_in_box(...)` for the edge-following raster

Every query takes a filter closure so `ZClip` **and class visibility** are
applied inside the grid walk rather than by the caller.

### Attaching it to the three backends

- **`Stage`** — all points resident, order immutable. Build once in
  `Stage::new`, never invalidate.
- **`TileStageManager`** — one grid per `LoadedTile`, built on load, dropped on
  unload.
- **`CopcStage`** — the hard case. `loaded_nodes` churns continuously as the
  camera moves. Build **one grid per node at ingest** and query only the nodes
  whose AABB meets the query circle. Nothing to invalidate, no rebuild storm.

### The COPC LOD problem

Zoomed out, the resident set is a thinned sample, so "5th percentile of Z within
0.5 m" would be computed over whatever sparse subset happened to be loaded and
be silently wrong by decimetres. Both mitigations:

1. **Spacing guard.** Compare the current LOD spacing against the snap radius.
   When spacing > radius, the snap marker renders in a distinct *coarse* state
   and clicking is refused with a status line explaining why.
2. **Neighbourhood pinning.** While digitising, pin the nodes intersecting a box
   around the cursor (default 20 m, or 4× the snap radius), reusing
   `CopcStage::pinned` and the existing force-load pump. Guard with a point
   budget analogous to `CORRIDOR_POINT_LIMIT` but far smaller — 5 M.

Deliverables: `spatial.rs` with unit tests, a criterion bench on a synthetic
10 M-point cloud, and the three backends wired up. **No user-visible change.**

### Status: built

`haiviz/src/spatial.rs` + `haiviz/benches/spatial.rs`. All three backends hold a
grid: `Stage::index` (built in `Stage::new`), `LoadedTile::index` (built in
`load_tile`), `CopcLoadedNode::index` (built in `ingest_node`). Each backend
exposes `snap_nearest_ray` and `snap_percentile_z` wrappers that pass a
`ZClip` filter into the grid walk; `CopcStage` also has `lod_spacing_near`
(the spacing guard) and `pin_neighbourhood` (the pinning mitigation, guarded by
`SNAP_PIN_POINT_LIMIT`). Nothing calls them yet, so every new item carries an
`#[allow(dead_code)]` that phase 4 removes.

Three things came out differently from the sketch above:

- **Coordinates are viewer-space f32, not world f64.** That is the space all
  three `positions` arrays, the camera ray and the existing pick code already
  use, so nothing converts on the way in or out. A caller wanting world
  coordinates adds the `viewer_origin` it already owns. This is the opposite
  choice from `Polyline` in §7.1, and for the opposite reason: an index is
  transient and cloud-local, a stored coordinate is neither.
- **`nearest_ray` returns the frontmost hit within the tolerance**, not the one
  closest to the ray axis, and carries `t` so hits from several tiles or COPC
  nodes can be merged. Within the tolerance the operator means the surface
  facing them, not what sits marginally closer to the axis behind a wall.
- **Percentiles are nearest-rank**, so the answer is always a real point rather
  than an interpolated Z nothing in the cloud sits at. `0` and `100` give the
  raw extrema back, as decision 1 requires.

Multi-source percentiles pool their candidates before ranking
(`spatial::percentile_by_z`): a percentile over a union is not the union of
percentiles, and a snap radius straddling a tile edge is the normal case.

### Measured

`cargo bench -p haiviz --bench spatial`, on a synthetic 10 M-point cloud over
200 m (≈250 pts/m², 0.089 m auto cell, 2237×2237 cells). The linear-scan rows
are what the existing `find_nearest_point` does today, and what a snap on every
mouse move would have cost.

| Query | Indexed | Linear scan |
|---|---|---|
| radius r=0.2 m | 85 ns | 3.83 ms |
| radius r=0.5 m | 384 ns | — |
| radius r=2 m | 6.97 µs | — |
| p5 percentile r=0.2 m (sort included) | 413 ns | — |
| nearest ray, top-down | 87 ns | 21.4 ms |
| nearest ray, oblique 40° | 19.3 µs | 36.9 ms |
| 40 m raster window (phase 5, per anchor) | 263 µs | — |

Build is 443 ms for 10 M points and 1.35 ms for a 100 k-point COPC node. The
first is a one-off beside a multi-second LAZ read; the second sits on the ingest
path next to `build_chunks` and the colour pass, and is the same order as both.
Memory works out at ~14 B/point at the default occupancy of 2 points per cell —
100 MB of per-cell arrays plus 40 MB of indices at 10 M — against ~120 MB for
the `positions` the grid indexes.

The oblique ray is 200× the top-down one because its corridor crosses hundreds
of columns rather than one; the Z clip against the grid extent is what keeps it
to 19 µs, which is still a tenth of a percent of a frame.

**This does not close risk 1.** The bench measures one contiguous grid. The COPC
worry is fragmentation — many small per-node grids resident at once, each
queried separately — and only real COPC data will show that.

---

## 6. Work package B — class visibility

### 6.1 State

`ClassDefs` is `HashMap<u8, (String, [f32; 3])>` — a tuple that is awkward to
extend, and used in many places. So visibility lives **beside** it, not in it:

```
pub struct ClassVis([bool; 256]);   // haiviz/src/raw_miniquad.rs, next to ZClip
```

Held per stage / COPC stage / tile manager exactly as `z_clip` is, and pushed
from the UI by a helper modelled on `apply_z_clip` (`main.rs:612`).

`classes.txt` is **not** extended (decision 10). Visibility is a view setting,
like the height filter, and the documented `id;r;g;b;label` format stays valid.

### 6.2 Rendering

The vertex shader is GLSL ES 100 (`raw_miniquad.rs:2495`), which has no bit
operations. Two viable options:

- **Sentinel colour (recommended).** A hidden point gets its red channel set to
  `-1.0` during the colour rebuild — impossible for a real colour. The vertex
  shader culls it with the same "push past the far plane" trick `ZClip` already
  uses. Zero memory cost, no vertex-format change, and it reuses
  `rebuild_colors` as the single update path for both colour changes and
  visibility toggles. Overloads a channel, so it needs the same kind of
  explanatory comment the `ZClip` hack already carries.
- **vec4 colour.** Cleaner, but +4 bytes per point on the GPU (~17 % of total
  vertex memory) and it touches `Vertex`, `build_chunks`, `upload_split_verts`
  and section extraction.

Toggling a class therefore costs one colour-buffer rebuild — the same cost as
changing colour mode, which is already accepted as interactive.

### 6.3 Everything else must honour it

The `ZClip` doctrine is that a clipped point is not merely invisible: it cannot
be measured, inspected or lassoed either. Class visibility extends exactly the
same rule, which means every one of these sites gains the check:

- vertex shader cull (above)
- `find_nearest_point` ×3
- `extract_section_verts` and its COPC / tile equivalents
- `pick_section_point` (`main.rs:750`)
- `collect_points_in_bbox` ×3 — the ground-plane fit
- all new snap queries and the edge-following raster

The ground fit is a real bonus: hide everything but class 2 and the plane is
fitted to actual ground instead of to whatever minima the grid found.

### Status: built

`ClassVis` is a type alias for `ClassMask`, not a separate `[bool; 256]`. The
plan kept the visible set and an op's `from` restriction apart, but §6.4 then
intersects them — so one representation (`[u64; 4]`, 32 bytes, cheap to copy
into a per-point filter closure) makes that intersection a single `&`.

Filter sites carrying the check: the vertex shader (via the sentinel), the
three `find_nearest_point`s, `pick_section_point`, the three
`collect_points_in_bbox`es, `Stage`/`TileStageManager`/`CopcStage` section
extraction, and the four phase-0 snap closures.

The colour rebuild is triggered **once**, right after `apply_class_vis`, rather
than in each of the three per-mode blocks — see §4. It is unconditional rather
than gated on Classification mode, because the sentinel rides in the colour
buffer whatever the colour mode is. `apply_class_vis` reports "did any backend
change" rather than "did the operator toggle", which is also how a file loaded
while a filter is already on gets repainted.

### 6.4 Ops must freeze the visible set

Ops are replayed on COPC nodes as they stream in and again on every node at
export, so *"the exported file agrees with the display by construction"*. If a
lasso's meaning depended on the live visibility state, an export an hour later
with different classes toggled would reclassify different points and break that
guarantee. So the visible set is **frozen into the op** when it is made, exactly
as the ground plane and the height band already are.

The clean way to do this is to generalise the field that already exists.
`ClassOp::from_class: Option<u8>` becomes a 256-bit mask:

- "Only from: 5" is the singleton mask
- no restriction is the all-ones mask
- the visible set is a mask
- what gets stored is the **intersection**, and `apply_point` does one test

Serialisation stays positional-safe because the `from` token sits at index 2,
ahead of the variable-length polygon. It gains a third form:

```
op;seq;*;...          unchanged — no restriction
op;seq;5;...          unchanged — single class
op;seq;2|5|6;...      new — an explicit set
```

**Downgrade behaviour:** an older haiviz cannot parse `2|5|6` and would skip the
op with a warning (`ClassEdits::load` skips malformed lines). Mitigation: emit
the set form **only when a visibility filter was actually in force**. Ordinary
sessions keep writing byte-identical old-format sidecars. Bump the header
comment to `v2` and document the new token in the comment block
`ClassEdits::save` already writes.

**As built, there is a fourth form:** `!3|7`, everything except those. A visible
set is usually "all but the two I switched off", and writing that out as 254 ids
would put a kilobyte on every op line. `format_mask` picks whichever of the set
and its complement is shorter. Downgrade behaviour is identical — an older build
skips either new form — and the v1 guarantee is unchanged and now tested
(`a_session_without_a_visibility_filter_writes_v1_op_lines`). An empty mask
serialises as `-`; the UI refuses to make one, but a file carrying one
round-trips rather than failing to parse.

### 6.5 UI

In the Classes window (`main.rs:1030`): a checkbox per row, plus **All**,
**None**, **Invert** and **Isolate** (show only this one).

Two things that must be loud, because both are foot-guns:

- **Visibility does not affect export.** Same as the height filter. Use *Omit
  classes* for that. Say so in the panel, not only in the README.
- **A hidden class is invisible in more ways than one.** When any class is
  hidden, show a persistent indicator near the tool panels ("3 classes hidden"),
  or the first "why did my lasso catch nothing?" report is inevitable.

Also: if *Only from: 5* is set while class 5 is hidden, the intersection is
empty and the op would do nothing. Warn at that point rather than applying a
no-op.

As built, All / None / Invert / Isolate act on the **listed** ids only. Isolate
via `ClassMask::single` would also switch off the 200-odd ids the window never
shows, and "N classes hidden" would stop being a number the operator could
reconcile with what is on screen.

---

## 7. Work package C — polyline model and file I/O

### 7.1 Model changes

`polylines.rs` currently stores `points: Vec<[f32; 3]>` in viewer space and can
only load. It needs:

- **f64 world coordinates as the master representation**, with a cached f32
  viewer-space copy for drawing. f32-relative-to-origin is fine for a
  cloud-local overlay (sub-mm within a few km of the origin) but round-tripping
  an external GeoPackage through it would quantise coordinates. Cheap to do now,
  painful later.
- **A layer level.** GeoPackage is a container of named feature tables; Salome
  is one flat file. `PolylineFile` gains `format` and a layer grouping so a
  GeoPackage round-trips its table name, and the panel becomes a layer tree
  rather than a flat list.
- **Mutability**: append vertex, pop vertex, insert, delete, close ring, plus a
  `dirty` flag and an undo stack.
- **A destination from the moment the layer is created**, because layers
  auto-save (§8.2). A new layer defaults to a sidecar beside the cloud —
  `<cloud>.haiviz-vectors.gpkg`, mirroring the existing
  `<cloud>.haiviz-class.txt` convention — with *Save As* to relocate it. No
  file dialog on layer creation.
- **Atomic writes.** We always rewrite a whole file rather than editing in
  place, so every save goes to a temporary file and is renamed over the target.
  A crash mid-write must not be able to destroy the previous version.

The undo stack should be a plain `Vec` of edits — deliberately *not* modelled on
`ClassEdits`. That design stores geometric ops in a sidecar because
classification has to be replayable against streaming nodes and at export. A
polyline is just data.

### Status: built (2a)

`Polyline` holds world f64 `points` plus a private viewer-space `view` cache;
mutation goes through `PolylineFile` rather than through the line, which is what
keeps the undo entry and the cache refresh from being forgettable.
`PolylineFile::refresh_view(origin)` is called once a frame and is a single
float comparison unless the origin actually moved.

The undo stack stores **inverses**, not actions — `Truncate`, `Reinsert`,
`Uninsert`, `Restore`, `Unadd`, `Readd`, `Unclose` — so undo is a straight apply
with nothing re-derived. `add_line` goes on the same stack, so a full unwind
removes the line too.

Also landed here, ahead of their users: `detect_format` by magic bytes (§7.3),
`changed_on_disk` for the auto-save mtime check (§8.2), `salome_losses` for the
§7.3 downgrade warning, and `pick_vertex` for §11. The atomic write is
temp-file + `sync_all` + rename within the same directory; without the sync the
rename can order ahead of the data and leave the new name over empty content,
which is worse than either version.

A new layer does **not** yet get a default destination — that belongs with the
digitising flow in phase 3, which is when a layer is first created from inside
haiviz. `PolylineFile::save` refuses a GeoPackage until 2b lands.

### 7.2 GeoPackage, pure Rust

Every mainstream SQLite crate (`rusqlite`, `sqlx`, `diesel`) wraps the C
library. `turso` is a genuine pure-Rust SQLite, but it is pre-1.0 and is a full
MVCC engine — a lot of dependency for writing a table of blobs.

**Decision: hand-roll a GeoPackage-specific SQLite codec in
`haiviz/src/gpkg.rs`.** This is not "implement SQLite". Two constraints we
control shrink it drastically:

- We **only ever write a fresh file**, never update in place. No freelists, no
  page reuse, no incremental vacuum, no journalling.
- We **only ever read feature tables**. No SQL engine, no query planner, no
  indexes, no triggers, no virtual tables.

What that leaves: varints, the record serial-type encoding, table b-tree leaf
and interior traversal, overflow pages (needed — a long linestring blob exceeds
the page payload), and reading `sqlite_schema` to find the feature tables.
Budget ~900 lines on top of the GeoPackage-specific work.

GeoPackage specifics, following `hqmatcher/src/gpkg.rs` but going 3D:

- `gpkg_spatial_ref_sys`, `gpkg_contents`, `gpkg_geometry_columns` as there.
- **`LineStringZ`** — WKB type 1002, and `z = 1` in `gpkg_geometry_columns`.
  A 2D export would throw away the elevation all the snapping worked to get
  right. Read `LineString`, `LineStringZ`, `MultiLineString(Z)` and
  `Polygon(Z)` rings.
- `application_id` (0x47504B47) and `user_version` (10300) live at fixed header
  offsets 68 and 60, so we write them directly and never need `PRAGMA`.
- Attributes: write `fid, geom, name, code, n_vertices, length_3d, created`. On
  read, pull non-geometry columns into a per-feature string map and write them
  back verbatim on save, so a file from QGIS round-trips without losing columns.
  Typed attribute *editing* is out of scope for v1.

Risks:

- **WAL sidecars.** A `.gpkg-wal` means committed data lives outside the main
  file. Detect it and refuse with a clear message rather than silently reading
  stale rows. Implementing WAL reads is a separate chunk of work; not up front.
- **QGIS files carry rtree virtual tables.** The reader must skip
  `sqlite_schema` entries it does not understand rather than failing on them.
- **Validation is the real cost, not the code.** Needs a differential test
  suite: write, verify with the `sqlite3` CLI, open in QGIS; and read files QGIS
  produced. "It parses on my machine" means very little for a format other
  people's tools must consume.

Fallback if reading real-world files proves hairier than expected:
`sqlite-parser-nom` (pure Rust, read-only b-tree/record parser) for the read
half, hand-rolled writer for the other. Adds one small dependency.

### Status: built (2b)

`haiviz/src/gpkg.rs`, ~1450 lines including tests, no new dependencies. The
fallback was not needed.

One thing the plan did not account for: the GeoPackage metadata tables declare
TEXT primary keys and unique constraints, and SQLite expects a matching
`sqlite_autoindex_*` b-tree for each. Dropping the constraints would have
avoided writing index b-trees but would have produced files that do not match
the spec, so **index b-trees are implemented too** — leaf and interior, with
dividers promoted out of the leaves rather than copied up, which is where index
b-trees differ from table ones.

Other details that only showed up against real files:

- The text-encoding field at offset 56 is a 4-byte big-endian integer, not a
  byte. Reading it as a byte rejects every file GDAL writes.
- A closed ring in a `LINESTRING` layer is written with a repeated first vertex
  and read back as `closed` with the repeat dropped. Nothing in the format
  distinguishes that from an open line returning to its start, so there is no
  information to lose — and it is what makes a ring survive a round trip.
- `fid` is an `INTEGER PRIMARY KEY`, so it is a rowid alias: the record stores
  NULL and the value is the cell key. Reading it out of the record gives NULL
  for every feature in the file.
- Feature ids are renumbered from 1 on write rather than carried over. A
  feature read from one file has no claim on its id in another, and a duplicate
  would silently drop a row.

**Validation was indeed the real cost.** Five differential tests, skipped
rather than failed when the tools are absent:

| Test | Checks |
|---|---|
| `sqlite3_agrees_that_the_file_is_sound` | `integrity_check` + `foreign_key_check` on 300 features × 500 vertices, then indexed lookups |
| `sqlite3_agrees_when_the_indexes_outgrow_one_page` | 300 layers, so the index b-tree splits; asserts the root really is an interior page |
| `ogr_reads_the_layer_as_a_3d_line_string` | GDAL sees 3D Line String, the right feature count, and the EPSG resolves |
| `a_file_written_by_gdal_reads_back_correctly` | reads a real GDAL file, rtree tables and triggers and all |
| `a_gdal_file_round_trips_back_through_gdal` | GDAL → us → GDAL, with sub-millimetre coordinates intact |

### 7.3 Format handling

- **Detect on load by magic bytes** (`SQLite format 3\0`), not extension —
  `.gpkg` is not guaranteed, and the existing Salome files use
  `.txt` / `.xyz` / `.poly` / no extension.
- **Save** writes back in the format the file came from; **Save As** offers
  both.
- Salome carries no attributes, so GeoPackage → Salome silently drops
  `name` / `code` / columns. Warn; do not just do it.

---

## 8. Work package D — digitising mode

**New file: `haiviz/src/vectorise.rs`.** `main.rs` is already 4437 lines and
should hold only the mode wiring.

- New `AppMode::Digitise`, alongside `GroundLasso`.
- Shared input handling next to the existing lasso block (`main.rs:2891`):
  left-click appends, Backspace pops, Enter finishes the line, Esc cancels.
- Rubber band and committed vertices drawn through `draw_3d_overlays`, which is
  already called identically from all three per-mode blocks.
- Target layer selection, new-layer creation, line naming.
- One shared panel function, called from the three egui blocks, in the shape of
  `ground_tool_controls`.

Proposed keys — `P` start/cancel digitising, `E` toggle edge following, `N`
cycle snap mode, modifier+wheel for radius, Shift-click for an unsnapped vertex.
(`O`, `G`, `V`, `C`, `T/F/B/L/R`, `WASD` are taken.)

### 8.2 Auto-save

Layers save themselves, following the classification sidecar's habit of writing
out immediately rather than waiting for a Save.

**Granularity is the completed line, not the vertex.** `ClassEdits` saves after
each lasso, not after each lasso point, and the same reasoning applies here: a
whole-file rewrite per vertex would be wasteful, and a half-drawn line is not a
meaningful thing to persist. So a save fires on Enter (line finished), on a
committed edit (vertex moved or deleted, line deleted), and on exit.

Consequences:

### Status: built (3)

`haiviz/src/vectorise.rs` plus the mode wiring in `main.rs`: `AppMode::Digitise`,
`P` beside the `G`/`V` lasso keys, click/Backspace/Enter/Esc in the shared input
block, one `vectorise::controls` panel called from all three per-mode blocks,
and the overlay drawn through `draw_3d_overlays`.

A vertex resolves through `pick_world_point`, a new shared screen-space pick
across all three backends — the same rule Measure and Point Info use, so a
vertex lands where a pick would. Shift-click places freely on a horizontal
plane through the previous vertex. Phase 4 replaces `pick_world_point` with the
ray and percentile snapping from phase 0 without touching anything else here.

The panel returns a `Request` rather than taking `AppMode`, so `vectorise.rs`
never learns about the mode enum; `main.rs` acts on it in one place, next to the
`start_ground_lasso` handler.

**A phase-0 bug surfaced while testing this.** `cell = sqrt(area · occupancy /
n)` is self-scaling — the cell count comes out as `n / occupancy` whatever the
footprint — but the `MAX_CELL` clamp broke that in the expensive direction: a
cell forced *smaller* than the formula asked for means *more* cells than there
are points. A COPC root node is a thinned sample of the whole file, so 50 k
points over a 30 km extent asked for a 130 m cell, got clamped to 5 m, and
allocated 36 M cells — **720 MB** — on the loader thread while the camera
moves. `MAX_CELLS_PER_POINT` is now a floor on the cell size that overrides
`MAX_CELL`, taking that case to 4 MB, and it is enforced in `build_with_cell`
too so an explicit cell size cannot bypass it.

- A crash loses at most the line currently being drawn. That is an acceptable
  loss and much simpler than journalling in-progress geometry.
- Whole-file rewrite cost is negligible at realistic sizes — a GeoPackage of
  10 k linestrings is a few MB and writes in milliseconds. Worth a note in the
  code so nobody is surprised if a project reaches six figures of lines.
- **External edits can be clobbered.** If the file changed on disk since we
  loaded it — someone editing the same GeoPackage in QGIS — an auto-save would
  overwrite their work without asking. Record the mtime at load, compare before
  writing, and stop and ask rather than silently winning.

## 9. Work package E — snapping

Modes, all within radius `R` of the cursor, all honouring `ZClip` **and class
visibility**:

- **Nearest point** — true 3D, ray-based, works from any camera angle
- **Low percentile Z** (default 5th) — gutters, ditch invert, toe of slope
- **High percentile Z** (default 95th) — kerb top, crest, cable, roof ridge
- **Same Z as previous vertex** — contour and breakline digitising
- **Existing polyline** vertex or segment — closing loops, tying into
  neighbouring lines

Radius is an absolute metre value, default 0.2 m, adjusted with modifier+wheel.
Absolute rather than zoom-relative because a metre means the same thing at every
zoom level, which is what makes a snap reproducible.

Percentiles are computed over the in-radius set after filtering; a few hundred
points, so sorting is free.

Visual feedback:

- Snap candidate marker (cross, like `measure_a` / `measure_b`) and the radius
  circle drawn at the snapped Z
- A visibly distinct *coarse* / *snap failed* state, never a silent fallback
- Live readout: segment length, slope, ΔZ, total length, vertex count — the
  format the measure tool already uses

Class visibility is what makes this precise: hide everything but ground and
"lowest Z in radius" means the ground. Hide ground and show class 6 and it means
the building.

### Status: built (4)

`SnapMode`, `SnapQuality` and `SnapHit` in `vectorise.rs`; `resolve_snap` and
`cursor_ray` in `main.rs`, where the backends are. `N` cycles the mode,
Ctrl + wheel sets the radius, and the marker is a cross with the radius circle
drawn at the *snapped* Z so the operator can see what a percentile covers.

The percentile modes need an XY to work from, and it comes from the **ray hit**
rather than from a fixed plane. That is what makes "5th percentile within
0.2 m" mean the same thing from an oblique view as from directly above.

Both COPC mitigations are live: `lod_spacing_near` refuses the aggregate modes
when the resident spacing exceeds the radius — but not `Nearest`, since a real
point is a real point at any LOD — and `pin_neighbourhood` force-loads a 20 m
box around the cursor, released on leaving the mode and skipped while a section
corridor owns the pin set.

**One phase-0 API turned out to be the wrong shape.** The per-backend
`snap_percentile_z` wrappers each ranked their own points, which is exactly the
"a percentile over a union is not the union of percentiles" mistake their own
doc comment warned about — a snap radius straddling two tiles, or a stage and a
COPC file loaded together, would have been quietly wrong. They are replaced by
`snap_z_candidates`, which appends raw candidates so `resolve_snap` pools them
and ranks once. `pick_world_point` from phase 3 is gone, superseded as planned.

## 10. Work package F — edge following

Livewire / intelligent scissors, adapted to point clouds. **3D / oblique view
only** — the section view gets placement and snapping, but not tracing.

1. On each new anchor (the last committed vertex), rasterise resident points in
   a window around it into a world-XY grid — per cell: count, z_min, z_max, mean
   intensity, dominant class. ~40 m window, capped at 512×512 cells.
2. Derive an **edge-strength field** as a weighted sum of normalised channels:
   - gradient of z_min → kerbs, breaklines, ditch edges, embankment tops
   - z_max − z_min per cell → walls, building footprints, vertical faces
   - **intensity gradient → road markings and painted lines**, probably the
     highest-value channel for road work
   - classification boundary → veg/ground, building/ground (much sharper once
     hidden classes are excluded from the raster entirely)
   - occupancy gradient → cloud boundary, roof outline
3. Dijkstra once per anchor over the grid (8-connected, cost ≈ `1/(edge+ε)` plus
   a small length term), caching cost and parent. Then **every frame** the ghost
   path is just a parent-pointer walk from the cursor cell back to the anchor —
   this is what makes it feel live. 512² ≈ 262 k nodes, a few ms per anchor,
   nothing per frame.
4. Click commits the traced path through Douglas–Peucker, so the result is a
   handful of vertices rather than one per cell.
5. Optionally constrain the search to a corridor around the straight
   anchor→cursor line, so the path cannot wander around the far side of a
   building.
6. Vertex Z comes from the raster channel matching the active snap mode, then
   optionally re-snapped per vertex at full resolution.

Fallback when the cursor is unreachable or outside the window: plain straight
rubber band. The tool must never block.

**Edge-strength heatmap overlay, toggleable.** Not a nicety — without seeing
what the tracer sees, tuning channel weights is guesswork.

## 11. Work package G — editing and section view

- Vertex-level pick (`polylines::pick` only resolves to a whole polyline), then
  move / insert / delete vertex.
- Delete line, close ring, re-drape (re-snap every vertex after changing snap
  mode).
- Digitising in the section view, in (along, Z) space — the natural place to
  pick a rail profile or a kerb face precisely. Placement and snapping only; no
  edge following (decision 8).

### Status: built (6)

`haiviz/src/vector_edit.rs` holds the editor state, the edit operations, the
re-drape, the panel and the handles; `main.rs` holds `AppMode::EditLines`,
`AppMode::SectionDigitise` and the wiring. Section-view snapping lives beside
the 3D snap in `vectorise.rs` and `main.rs::resolve_section_snap`.

**Editing is its own mode (`M`), not part of digitising.** Near an existing
vertex a click means "grab it" to an editor and "tie into it" to a digitiser;
one mode cannot mean both. `P` and `M` switch between the two, but `M` refuses
mid-line rather than drop a half-drawn line on a stray key.

**Drag, not click.** A press on a handle only moves anything once the pointer
has travelled `DRAG_THRESHOLD_PX`; below that the release is a click and only
selects. Otherwise selecting a vertex to read its coordinates would re-snap it
to whatever sits a pixel away. Nothing reaches the model until release, so a
drag is one undo entry and one auto-save. A release over a refused snap — or
over the panel — leaves the line alone and says why; no stale position from
earlier in the drag is committed.

A dragged vertex snaps through the same `resolve_snap` as a new one, with two
additions: `prev` (the vertex's own starting height, for same-Z) and
`SnapExclude`, which hides the dragged vertex and its two segments from the
existing-polyline snap. Without it that snap finds the vertex under the cursor
on every frame and the drag never goes anywhere.

**The selected vertex carries its line.** `Editor::vertex` is
`(PolyRef, usize)`, not an index into `selected_poly`: selecting another line in
the Polylines panel would otherwise silently move the selection to vertex *i* of
a different line.

**Undo is ordered across files.** Each `PolylineFile` keeps its own inverse
stack (§7.1), so `Editor` records which file each committed change went to and
`Ctrl+Z` walks that order. A finished digitised line goes on the same order, so
the first undo after finishing takes the line back.

**Re-drape keeps plan position.** Each vertex keeps its XY and takes a new Z,
through the same backends, filters and COPC spacing guard as a snap. Letting it
also move sideways onto whichever point realised a percentile would undo the
placement the operator did by hand. `Nearest` means closest in **3D** among the
in-radius candidates, so a vertex on a bridge deck stays on the deck. Same-Z and
existing-polyline have no height of their own and are not offered. The whole
drape is one `Edit::RestorePoints`, so one undo reverses it. Vertices that
cannot be answered keep their height and are counted in the status line.

**Section digitising lands on the plane.** A vertex takes the (along, height) it
snapped to, on the section plane, even when the point it snapped to sits up to
the corridor depth off it: the section view shows nothing of that offset, so a
vertex that kept it would be a coordinate the operator never saw. Tying into an
existing line is the exception (`polylines::nearest_in_section`), because
sharing a coordinate is the whole point. Low/High Z take the percentile over a
vertical strip of ±radius along the section — the slice's equivalent of the 3D
radius column. The queries run over `SectionData`'s own points, which already
carry the height band and class visibility; the index over them is a
`OnceCell`, built on first use, because a section is rebuilt on every colour or
ramp change and almost none of those are ever digitised on. On COPC, starting
the tool force-loads the corridor through the lasso's path (`CorridorThen`
replaces the old `corridor_load_to_lasso` flag), so there is no LOD to guard
against.

Not in the plan, added because they cost little: **Shift+Enter finishes a line
as a ring**, and the selected line draws twice as wide.

Found and fixed while wiring this:

- Removing a polyline file left `digitise.target` and `selected_poly` pointing
  at whatever file slid into its index. Removal is now deferred to the top of
  the frame like cloud unloads, and every file index is fixed up in one place.
- **Ctrl + wheel zoomed as well as changing the snap radius** — the zoom never
  checked the modifier.
- In tile mode `Enter` and `Delete` loaded and unloaded the selected tile while
  digitising — so finishing a line also loaded a tile.
- Three status messages had lost their `\` line continuations and showed long
  runs of spaces in the panel.

Not done: vertex editing in the section view (§11 never asked for it), and a
re-drape on COPC does not pin nodes along the line — vertices where the loaded
data is too coarse keep their height and are reported, and zooming in along the
line then re-draping again fills them in.

---

## 12. Order of work

| Phase | Content | Rough size | Ships something? |
|---|---|---|---|
| **0** | A — spatial index, all three backends, tests + bench | ~400 | no — **done** |
| **1** | B — class visibility: state, shader cull, all filter sites, `ClassOp` mask, Classes window UI | ~500 | **yes, standalone — done** |
| **2a** | C — polyline model: f64, layers, mutability, undo, Salome writer | ~250 | marginally — **done** |
| **2b** | C — GeoPackage read + write, EPSG detection, differential tests | ~1400 | **yes — overlay any GPKG — done** |
| **3** | D — digitising mode, free vertex placement, panel wiring | ~450 | yes — **done** |
| **4** | E — snapping, radius control, feedback, COPC guard + pinning | ~450 | yes, the core tool — **done** |
| **5** | F — edge following, cost channels, livewire, heatmap | ~600 | yes |
| **6** | G — vertex editing, section-view digitising, re-drape | ~400 | yes — **done** |

Phase 1 is worth doing early and independently: it is useful on its own, it
improves the ground-plane fit immediately, and every later phase depends on the
filter being in place.

Phase 0 is the risk concentration — nothing above it works without a fast radius
query, and it is the only part whose performance cannot be predicted by reading
the code. Build it with the bench first.

## 13. Risks

1. **Spatial index performance on COPC.** Per-node grids may fragment queries
   badly when many small nodes are resident. Measure before building on it.
2. **GeoPackage interoperability.** The format is easy to write plausibly and
   hard to write correctly. Budget as much time for the differential tests as
   for the codec.
3. **Sidecar downgrade.** Sidecars containing a visibility mask are dropped
   silently by older builds. Mitigated by only emitting the new token when a
   filter was in force, but it should be in the release notes.
4. **Hidden-class confusion.** A view setting that silently changes what a
   destructive tool does is dangerous. The persistent indicator in §6.5 is a
   requirement, not a nicety.
5. **Scope.** Phases 0–4 are a complete, useful tool. Phase 5 is the speculative
   one — its value cannot be judged until the heatmap can be seen on real data.
6. **Auto-save over a whole-file rewrite.** Combining "save on every completed
   line" with "always write the file fresh" means the write path is exercised
   constantly and a bug there destroys work rather than merely failing. Atomic
   temp-file-plus-rename and the mtime check (§7.1, §8.2) are load-bearing, not
   polish.

## 14. Documentation to update on completion

`haiviz/README.md` needs: a Vectorisation section (modes, keys, snapping, edge
following, auto-save and where a new layer lands), class visibility under Manual
classification — including that it does not affect export — the GeoPackage
format in the Polylines section, the controls cheat-sheet, and an explicit note
on why digitising works from oblique views when the plan-view lassos do not.

All of that is written **except the edge-following part of the Vectorisation
section**, which waits on phase 5. The README also gained two sections the plan
did not anticipate: *The panel* and *Class visibility*. Phase 6 added *Editing
lines* and *Digitising in a section* under Vectorisation, with their keys in the
cheat-sheet.

---

## 15. Where this stands

Branch `vectorisation`, twelve commits off `master`. `cargo test -p haiviz` is
green (120 tests) and `cargo clippy --all-targets` adds no warnings beyond the
two that predate this work.

### Done

| Phase | Commit subject |
|---|---|
| 0 | spatial index for snap queries |
| 1 | class visibility |
| 2a | polyline model, layers, mutability, undo |
| 2b | GeoPackage read and write |
| 3 | digitising mode |
| 4 | snapping |
| 6 | vertex editing, section-view digitising, re-drape |

Plus four follow-ups after the tool was actually driven on a cloud: a ray-based
fix to the existing-polyline snap, and three rounds of UI work — one docked
panel instead of five floating windows, depth-tested lines with real width, a
view that renders beside the panel rather than under it, a cursor readout, and
fixes to pointer and keyboard capture.

### Not done, by decision

**Phase 5 — edge following.** §13 risk 5 calls it the speculative one, and its
value cannot be judged until the heatmap can be seen on real data. Deferred
deliberately so phases 0–4 could be used first, not because it ran out of road.

### Built, not yet driven

**Phase 6.** Everything in §11, plus the fixes listed under its status. The
model and editor logic are tested; the interaction has only been launched, not
used. What most likely needs a second round, in the order worth trying:

- the drag threshold and pick radii (`DRAG_THRESHOLD_PX`, `VERTEX_PICK_PX`,
  `SEGMENT_PICK_PX`) — chosen, not measured, and never tried on a display whose
  DPI scale is not 1;
- left-drag does not orbit in edit mode, exactly as in digitising, because it
  grabs handles; whether orbiting on empty space is missed;
- section digitising on COPC, which goes through the corridor force-load;
- re-drape of a long COPC line, where vertices away from the loaded
  neighbourhood are kept and counted rather than draped.

### Things worth knowing before touching this again

- **Eye-dome lighting breaks overlay depth testing.** EDL composites through an
  offscreen target whose depth the default framebuffer does not share, so with
  EDL on the polylines draw over the cloud. Pre-existing, documented in the
  README, not fixed. The fixes are to blit the EDL depth back, or to render the
  overlays inside the EDL pass.
- **macroquad's screen and mouse coordinates are logical pixels;
  `apply_viewport` takes physical ones.** `ViewRect::gl` is the only place that
  crosses that boundary, and it cost a session to find. Any new viewport work
  goes through it.
- **egui's `is_pointer_over_area` is only true over a `SidePanel` after the
  panel has been laid out**, because a panel is `Order::Background` and the
  check falls back to `unused_rect`. Read the capture flags at the *end* of the
  ui closure, never the start.
- **The panel width lags one frame.** It is measured from egui after layout,
  which happens after the view renders. Only visible while dragging the panel
  edge.
- The phase-0 bench numbers in §5 were re-measured after the
  `MAX_CELLS_PER_POINT` fix and still hold.
