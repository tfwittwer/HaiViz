# HaiViz

A point-cloud viewer for LAS / LAZ data, built with
[macroquad](https://github.com/not-fl3/macroquad) and
[egui](https://github.com/emilk/egui). It renders large clouds with an
orthographic camera, extracts vertical cross sections, overlays polylines, and
lets you edit classification labels and colours live.

![haiviz screenshot](docs/haiviz.png)

<!-- Placeholder — drop a real screenshot at docs/screenshot.png to replace this. -->

## Building

```sh
cargo build --release
```

## Usage

```sh
cargo run --release -- [PATH]
```

or, if the binary is on your PATH

```sh
haiviz [PATH]
```

`PATH` is optional and decides which mode haiviz starts in (see below). Release
mode is strongly recommended — point loading and colouring are much faster.

## Loading modes

The kind of `PATH` you pass selects one of three modes automatically:

### 1. Single file / manual loading

```sh
cargo run --release -- cloud.laz
```

Pass a single `.las` / `.laz` file and the whole cloud is loaded into memory at
once. You can keep loading more individual files after launch:

- **Open file…** button in the control panel, or
- **drag & drop** one or more `.laz` / `.las` files onto the window
  (Windows / X11; on Wayland use the *Open file…* button).

Each file becomes its own entry in the **Loaded files** list, where you can
**Unload** it again. This is the mode to use for browsing a handful of tiles or
comparing several clouds side by side.

Launching with no `PATH` starts in this mode with an empty window, ready for
drag & drop or *Open file…*.

### 2. Tiles

```sh
cargo run --release -- /path/to/tile_directory/
```

Pass a **directory** and haiviz enters tile mode: it indexes every LAS/LAZ tile
in the folder and shows their footprints. Tiles are streamed on demand rather
than all held in memory —

- **click** a tile footprint to select it,
- **Enter** loads the selected tile, **Delete** unloads it,
- **Zoom on Object** frames the current selection.

If the directory also contains COPC files, they are loaded alongside the tiles
under a shared coordinate origin.

### 3. COPC

```sh
cargo run --release -- cloud.copc.laz
```

If the file you pass is a [COPC](https://copc.io/) (Cloud Optimized Point Cloud)
LAZ — detected from its COPC info VLR — haiviz streams it from its octree
hierarchy, loading finer detail near the camera and releasing distant nodes.
This is the mode for very large clouds that don't fit in memory. Multiple COPC
files can be loaded together, each listed under **Loaded COPC files** with its
own **Unload** control.

You can create COPC files out of conventional LAZ files with [copc-converter](https://github.com/360-geo/copc-converter).

## Navigation

The 3D view uses an orthographic camera shared across all modes:

| Input | Action |
|-------|--------|
| `W` `A` `S` `D` / arrow keys | Move the camera |
| Left-drag | Orbit around the screen centre |
| Right-drag | Pan |
| Mouse wheel | Zoom (orthographic scale) |
| `Tab` | Toggle mouse-look (cursor grab) |
| `Esc` | Cancel the current pick / measure / section action |
| **Zoom All** button | Frame all loaded data |

## Colouring

The **Color** dropdown switches how points are shaded:
Height, Intensity, Classification, RGB, PointSourceId, or Time. In **Height**
mode you can set the min/max range used for the colour ramp. The
**Background** colour and **point size** are adjustable, and an EDL
(eye-dome-lighting) **Strength** slider adds depth shading.

## Cross sections

Cross sections cut a vertical slice through the cloud along a line you draw:

1. Click **Cross Section** in the control panel.
2. Click the **start** point, then the **end** point of the section line.
3. **Scroll** to adjust the section **depth** (the corridor half-width around the
   line), then click to confirm.

You then drop into the 2D **section view**, looking straight at the slice.
There you can:

- **scroll** to zoom the section,
- **Measure** distances between two clicked points (with slope readout),
- click **Point Info** and then a point to inspect its attributes,
- see any polylines that cross the corridor drawn into the slice.

Click **Back to 3D** to return.

## Polylines

Open the **Polylines** window and **Load Polyline File…** to overlay vector
lines on the cloud. The format is Salome Hydro polyline ASCII: one `x y z` per
line, with blank lines separating individual polylines. Coordinates are
interpreted in the same CRS as the cloud (the viewer subtracts the data origin
automatically).

Each loaded file is listed with its polylines, where you can:

- select a polyline,
- change its **colour**,
- adjust its **radius** (line thickness, in metres — this sets how wide it draws
  in the section view), and
- **Remove file** to unload it.

Polylines are drawn both in the 3D view and, where they intersect the corridor,
in cross-section view.

## Class editing

Open the **Classes** window to edit classification labels and colours live.
It lists every classification id present in the data (plus any extras defined in
your file), and for each id you can edit its **colour** and **label**. Changes
apply immediately when colouring by **Classification**.

Definitions are stored in a `classes.txt` file in the working directory, using
the format:

```
# id;r;g;b;label
2;139;69;19;Ground
5;0;128;0;High vegetation
```

- **Save classes.txt** writes the current definitions to disk.
- **Reload** re-reads `classes.txt`, discarding unsaved edits.

If no `classes.txt` is present, built-in default labels and colours are used.

## Controls cheat-sheet

**3D view (all modes)**

| Input | Action |
|-------|--------|
| `W` `A` `S` `D` / arrows | Move camera |
| Left-drag | Orbit around screen centre |
| Right-drag | Pan |
| Mouse wheel | Zoom |
| `Tab` | Toggle mouse-look (cursor grab) |
| `Esc` | Cancel current pick / measure / section action |

**Building a cross section** (start from the **Cross Section** button)

| Step | Input |
|------|-------|
| Pick start point | Left-click |
| Pick end point | Left-click |
| Adjust corridor depth | Mouse wheel |
| Confirm | Left-click |

**Section (2D) view**

| Input | Action |
|-------|--------|
| Mouse wheel | Zoom the slice |
| **Measure** → click ×2 | Distance + slope between two points |
| **Point Info** → click | Inspect a point's attributes |
| **Back to 3D** | Leave the section view |

**Tile mode**

| Input | Action |
|-------|--------|
| Left-click footprint | Select tile |
| `Enter` | Load selected tile |
| `Delete` | Unload selected tile |
| **Zoom on Object** | Frame the selection |

**Measure & point info (3D)**

| Input | Action |
|-------|--------|
| **Measure** → click A, click B | Distance + slope |
| **Point Info** → click | Inspect a point's attributes |

## License

Licensed under the [MIT License](LICENSE).
