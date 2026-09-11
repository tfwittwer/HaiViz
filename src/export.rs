//! Writing edited point clouds back out.
//!
//! Output is always plain LAS/LAZ, never COPC: the `las` crate reads the COPC
//! octree but cannot write one, and a COPC chunk cannot be patched in place —
//! rewriting a point's classification changes the compressed chunk's length,
//! which shifts every subsequent chunk offset and invalidates the hierarchy.
//! Re-COPC-ing the exported file (copc-converter, PDAL) is a separate step.
//!
//! The export walks the *whole* source, not the resident subset, applying the
//! same `ClassEdits` the viewer applies at ingest. That is what keeps the file
//! and the screen in agreement: both run `ClassEdits::apply_to_node`.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender, channel};

use ::log::{error, info};
use las::copc::CopcEntryReader;
use las::{Reader, Writer};

use crate::classify::{ClassEdits, ExportFilter};
use crate::raw_miniquad::read_full_hierarchy;

/// Progress messages from the export thread to the UI.
pub enum ExportProgress {
    /// Total units of work (COPC nodes, or point chunks for a plain file).
    Started { total: usize },
    Advanced { done: usize, written: u64, dropped: u64 },
    Done { path: PathBuf, written: u64, dropped: u64 },
    Failed(String),
}

/// A running export. Polled once per frame from the UI thread.
pub struct ExportJob {
    pub rx: Receiver<ExportProgress>,
    pub label: String,
    pub total: usize,
    pub done: usize,
    pub written: u64,
    pub dropped: u64,
    pub finished: Option<Result<(PathBuf, u64, u64), String>>,
}

impl ExportJob {
    /// Drain everything the worker has sent since the last frame.
    pub fn poll(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                ExportProgress::Started { total } => self.total = total,
                ExportProgress::Advanced { done, written, dropped } => {
                    self.done = done;
                    self.written = written;
                    self.dropped = dropped;
                }
                ExportProgress::Done { path, written, dropped } => {
                    self.written = written;
                    self.dropped = dropped;
                    self.done = self.total;
                    self.finished = Some(Ok((path, written, dropped)));
                }
                ExportProgress::Failed(e) => self.finished = Some(Err(e)),
            }
        }
    }

    pub fn fraction(&self) -> f32 {
        if self.total == 0 { return 0.0; }
        (self.done as f32 / self.total as f32).clamp(0.0, 1.0)
    }
}

/// Start an export on a background thread. Large clouds take minutes; the
/// viewer stays interactive and reports progress.
pub fn spawn(
    source: String,
    out_path: PathBuf,
    edits: ClassEdits,
    filter: ExportFilter,
    is_copc: bool,
) -> ExportJob {
    let (tx, rx) = channel();
    let label = out_path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("export")
        .to_string();

    let out_for_thread = out_path.clone();
    let tx_err = tx.clone();
    let spawned = std::thread::Builder::new()
        .name("haiviz-export".into())
        .spawn(move || {
            let result = if is_copc {
                export_copc(&source, &out_for_thread, &edits, &filter, &tx)
            } else {
                export_plain(&source, &out_for_thread, &edits, &filter, &tx)
            };
            match result {
                Ok((written, dropped)) => {
                    info!(
                        "Export finished: {} — {} points written, {} dropped",
                        out_for_thread.display(), written, dropped,
                    );
                    let _ = tx.send(ExportProgress::Done { path: out_for_thread, written, dropped });
                }
                Err(e) => {
                    error!("Export failed: {e}");
                    let _ = tx.send(ExportProgress::Failed(e));
                }
            }
        });

    if let Err(e) = spawned {
        let _ = tx_err.send(ExportProgress::Failed(format!("cannot spawn export thread: {e}")));
    }

    ExportJob {
        rx, label,
        total: 0, done: 0, written: 0, dropped: 0,
        finished: None,
    }
}

/// Build the output header from the source, stripping anything that would
/// misdescribe the result: the COPC VLRs (the output has no octree) and the
/// source's laszip VLR (the writer adds its own).
fn output_header(src: &las::Header, compressed: bool) -> Result<las::Header, String> {
    let mut builder = las::Builder::from(src.clone());
    builder.vlrs.retain(|v| {
        !v.is_copc_info() && !las::laz::is_laszip_vlr(v)
    });
    builder.evlrs.retain(|v| !v.is_copc_hierarchy());
    builder.point_format.is_compressed = compressed;
    builder.generating_software = "haiviz".to_string();
    builder.into_header().map_err(|e| format!("header: {e}"))
}

/// How many nodes are decompressed at once. Export is decompression-bound on
/// anything large, and each node is independent.
const EXPORT_THREADS: usize = 4;

/// Read one COPC node. A fresh reader per node is required, not merely tidy:
/// `CopcEntryReader` wraps a layered LAZ decompressor that carries per-chunk
/// state, and `read_entry_points` seeks the file without resetting it — so a
/// reader reused across entries returns a short read on the second one.
fn read_node(source: &str, entry: &crate::raw_miniquad::RawCopcEntry) -> Result<Vec<las::Point>, String> {
    let file = BufReader::new(File::open(source).map_err(|e| format!("open {source}: {e}"))?);
    let mut reader = CopcEntryReader::new(file).map_err(|e| format!("copc reader: {e}"))?;
    let las_entry = las::copc::Entry {
        key: entry.key,
        offset: entry.offset,
        byte_size: entry.byte_size,
        point_count: entry.point_count,
    };
    let mut points = Vec::with_capacity(entry.point_count.max(0) as usize);
    reader.read_entry_points(&las_entry, &mut points)
        .map_err(|e| format!("read node {:?}: {e}", entry.key))?;
    Ok(points)
}

/// Apply the op list to one node's points, using the node's own extent as the
/// rejection box so ops nowhere near it cost nothing.
fn apply_edits_to(points: &mut [las::Point], edits: &ClassEdits) {
    if edits.is_empty() || points.is_empty() { return; }
    let mut nmin = [f64::MAX; 3];
    let mut nmax = [f64::MIN; 3];
    for p in points.iter() {
        for (a, v) in [p.x, p.y, p.z].iter().enumerate() {
            nmin[a] = nmin[a].min(*v);
            nmax[a] = nmax[a].max(*v);
        }
    }
    edits.apply_to_node(points, &nmin, &nmax);
}

fn export_copc(
    source: &str,
    out_path: &PathBuf,
    edits: &ClassEdits,
    filter: &ExportFilter,
    tx: &Sender<ExportProgress>,
) -> Result<(u64, u64), String> {
    let entries = {
        let file = BufReader::new(File::open(source).map_err(|e| format!("open {source}: {e}"))?);
        let reader = CopcEntryReader::new(file).map_err(|e| format!("copc reader: {e}"))?;
        read_full_hierarchy(&reader)
    };
    if entries.is_empty() {
        return Err("no COPC hierarchy entries found".to_string());
    }
    let _ = tx.send(ExportProgress::Started { total: entries.len() });

    let header = {
        let file = BufReader::new(File::open(source).map_err(|e| format!("open {source}: {e}"))?);
        let reader = CopcEntryReader::new(file).map_err(|e| format!("copc reader: {e}"))?;
        let compressed = out_path.extension().is_some_and(|e| e.eq_ignore_ascii_case("laz"));
        output_header(reader.header(), compressed)?
    };

    let out = BufWriter::new(File::create(out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?);
    let mut writer = Writer::new(out, header).map_err(|e| format!("writer: {e}"))?;

    // Decompress on a small pool and write on this thread. The bounded channel
    // is what caps memory: at most a few nodes are in flight at once, however
    // fast the workers run. Points therefore leave in node-completion order
    // rather than source order — LAS imposes no ordering, and a subsequent
    // re-COPC pass rebuilds spatial coherence anyway.
    let entries = std::sync::Arc::new(entries);
    let next = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let (node_tx, node_rx) = std::sync::mpsc::sync_channel::<Result<Vec<las::Point>, String>>(EXPORT_THREADS * 2);

    let mut workers = Vec::new();
    for _ in 0..EXPORT_THREADS {
        let entries = entries.clone();
        let next = next.clone();
        let node_tx = node_tx.clone();
        let source = source.to_string();
        let edits = edits.clone();
        let handle = std::thread::Builder::new()
            .name("haiviz-export-node".into())
            .spawn(move || {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(entry) = entries.get(i) else { break };
                    let msg = read_node(&source, entry).map(|mut points| {
                        apply_edits_to(&mut points, &edits);
                        points
                    });
                    let failed = msg.is_err();
                    if node_tx.send(msg).is_err() || failed { break; }
                }
            })
            .map_err(|e| format!("cannot spawn export worker: {e}"))?;
        workers.push(handle);
    }
    drop(node_tx);

    let (mut written, mut dropped) = (0u64, 0u64);
    let mut done = 0usize;
    let mut error = None;

    for msg in node_rx {
        match msg {
            Ok(points) => {
                for p in points {
                    if filter.keeps(&p) {
                        if let Err(e) = writer.write_point(p) {
                            error = Some(format!("write: {e}"));
                            break;
                        }
                        written += 1;
                    } else {
                        dropped += 1;
                    }
                }
            }
            Err(e) => { error = Some(e); }
        }
        if error.is_some() { break; }
        done += 1;
        // Reporting every node would flood the channel on a file with tens of
        // thousands of them; the UI only needs a moving bar.
        if done.is_multiple_of(32) || done == entries.len() {
            let _ = tx.send(ExportProgress::Advanced { done, written, dropped });
        }
    }

    // `node_rx` was moved into the loop above and is dropped by now, so any
    // worker blocked on a full channel sees a disconnected send and exits.
    for w in workers { let _ = w.join(); }

    if let Some(e) = error { return Err(e); }
    if done != entries.len() {
        return Err(format!("only {done} of {} nodes were written", entries.len()));
    }

    writer.close().map_err(|e| format!("close: {e}"))?;
    Ok((written, dropped))
}

fn export_plain(
    source: &str,
    out_path: &PathBuf,
    edits: &ClassEdits,
    filter: &ExportFilter,
    tx: &Sender<ExportProgress>,
) -> Result<(u64, u64), String> {
    let mut reader = Reader::from_path(source).map_err(|e| format!("open {source}: {e}"))?;
    let total_points = reader.header().number_of_points();

    let compressed = out_path.extension().is_some_and(|e| e.eq_ignore_ascii_case("laz"));
    let header = output_header(reader.header(), compressed)?;

    const BATCH: usize = 100_000;
    let total_batches = (total_points as usize).div_ceil(BATCH).max(1);
    let _ = tx.send(ExportProgress::Started { total: total_batches });

    let out = BufWriter::new(File::create(out_path).map_err(|e| format!("create {}: {e}", out_path.display()))?);
    let mut writer = Writer::new(out, header).map_err(|e| format!("writer: {e}"))?;

    let (mut written, mut dropped) = (0u64, 0u64);
    let mut batch: Vec<las::Point> = Vec::with_capacity(BATCH);
    let mut batch_idx = 0usize;

    let flush = |batch: &mut Vec<las::Point>,
                     writer: &mut Writer<BufWriter<File>>,
                     written: &mut u64,
                     dropped: &mut u64| -> Result<(), String> {
        edits.apply_all(batch);
        for p in batch.drain(..) {
            if filter.keeps(&p) {
                writer.write_point(p).map_err(|e| format!("write: {e}"))?;
                *written += 1;
            } else {
                *dropped += 1;
            }
        }
        Ok(())
    };

    loop {
        let slab = reader.read_points(BATCH as u64).map_err(|e| format!("read: {e}"))?;
        let mut any = false;
        for point in slab.points() {
            batch.push(point.map_err(|e| format!("read: {e}"))?);
            any = true;
        }
        if !any { break; }
        flush(&mut batch, &mut writer, &mut written, &mut dropped)?;
        batch_idx += 1;
        let _ = tx.send(ExportProgress::Advanced { done: batch_idx, written, dropped });
    }
    flush(&mut batch, &mut writer, &mut written, &mut dropped)?;

    writer.close().map_err(|e| format!("close: {e}"))?;
    Ok((written, dropped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::{ClassOp, effective_class};
    use std::sync::mpsc::channel;

    /// A COPC file to exercise the real read → edit → write path against.
    /// Set `HAIVIZ_TEST_COPC` to point at one; otherwise the `las` crate's
    /// bundled autzen sample is used if the registry copy is present.
    pub fn test_copc() -> Option<String> {
        if let Ok(p) = std::env::var("HAIVIZ_TEST_COPC")
            && std::path::Path::new(&p).exists() {
                return Some(p);
        }
        let base = format!("{}/.cargo/registry/src", std::env::var("HOME").ok()?);
        let mut stack = vec![std::path::PathBuf::from(base)];
        let mut depth = 0;
        while let Some(dir) = stack.pop() {
            depth += 1;
            if depth > 5000 { break; }
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            for e in entries.flatten() {
                let path = e.path();
                if path.is_dir() {
                    // Only descend where the sample could live.
                    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if name.starts_with("las-") || name.starts_with("index.") || name == "tests" || name == "data" || name == "src" {
                        stack.push(path);
                    }
                } else if path.file_name().and_then(|s| s.to_str()) == Some("autzen.copc.laz") {
                    return Some(path.to_string_lossy().into_owned());
                }
            }
        }
        None
    }

    fn bounds(path: &str) -> las::Bounds {
        las::Reader::from_path(path).unwrap().header().bounds()
    }

    fn read_all(path: &str) -> Vec<las::Point> {
        let mut r = las::Reader::from_path(path).unwrap();
        let data = r.read_all().unwrap();
        data.points().map(|p| p.unwrap()).collect()
    }

    /// An op covering the middle half of the file's X extent, full height,
    /// with a corridor wide enough to swallow the whole Y extent.
    fn full_width_op(b: &las::Bounds, target: u8, from: Option<u8>) -> ClassOp {
        let y_mid = (b.min.y + b.max.y) * 0.5;
        let len = b.max.x - b.min.x;
        let z_lo = b.min.z - 1.0;
        let z_hi = b.max.z + 1.0;
        let (a0, a1) = (len * 0.25, len * 0.75);
        ClassOp::corridor(
            0,
            (b.min.x, y_mid),
            (b.max.x, y_mid),
            (b.max.y - b.min.y) + 1.0,
            vec![(a0, z_lo), (a1, z_lo), (a1, z_hi), (a0, z_hi)],
            target,
            from.into(),
        )
    }

    #[test]
    fn copc_export_applies_ops_and_drops_classes() {
        let Some(src) = test_copc() else {
            eprintln!("skipping: no COPC test file (set HAIVIZ_TEST_COPC)");
            return;
        };
        let b = bounds(&src);
        let src_points = read_all(&src);
        let total = src_points.len();
        assert!(total > 0);

        // How many points the op should hit, computed independently of the
        // export by testing every point in the source.
        let op = full_width_op(&b, 7, None);
        let expected_hit = src_points.iter()
            .filter(|p| op.contains(p.x, p.y, p.z) && effective_class(p) != 7)
            .count();
        assert!(expected_hit > 0, "the test op must actually select points");
        assert!(expected_hit < total, "the test op must not select everything");
        eprintln!("source {src}: {total} points, op selects {expected_hit}");

        let mut edits = ClassEdits::default();
        edits.push(op.clone());

        let dir = std::env::temp_dir().join("haiviz-export-test");
        std::fs::create_dir_all(&dir).unwrap();

        // ---- Pass 1: reclassify only, nothing dropped ----
        let out = dir.join("reclass.laz");
        let (tx, _rx) = channel();
        let (written, dropped) =
            export_copc(&src, &out, &edits, &ExportFilter::default(), &tx).unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(written as usize, total, "no points may be lost by a reclassify");

        let out_points = read_all(out.to_str().unwrap());
        assert_eq!(out_points.len(), total);
        let got_class7 = out_points.iter().filter(|p| effective_class(p) == 7).count();
        let already_7 = src_points.iter().filter(|p| effective_class(p) == 7).count();
        assert_eq!(got_class7, expected_hit + already_7);
        // Every point inside the prism carries the new class.
        for p in &out_points {
            if op.contains(p.x, p.y, p.z) {
                assert_eq!(effective_class(p), 7, "point inside the op was missed");
            }
        }

        // ---- Pass 2: same ops, and now drop what they marked ----
        let out2 = dir.join("filtered.laz");
        let mut filter = ExportFilter::default();
        filter.drop_classes.insert(7);
        let (tx2, _rx2) = channel();
        let (written2, dropped2) = export_copc(&src, &out2, &edits, &filter, &tx2).unwrap();
        assert_eq!(dropped2 as usize, expected_hit + already_7);
        assert_eq!((written2 + dropped2) as usize, total);

        let out2_points = read_all(out2.to_str().unwrap());
        assert_eq!(out2_points.len(), written2 as usize);
        assert!(out2_points.iter().all(|p| effective_class(p) != 7));

        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&out2);

    }

    /// The plan-view ground tool end to end on real data: fit a plane to the
    /// points in a footprint, freeze it into an op, and check the export marks
    /// exactly what stands above it.
    #[test]
    fn above_ground_op_exports_what_stands_above_the_fitted_plane() {
        let Some(src) = test_copc() else {
            eprintln!("skipping: no COPC test file (set HAIVIZ_TEST_COPC)");
            return;
        };
        let b = bounds(&src);
        let src_points = read_all(&src);
        // The `las` crate's bundled autzen sample is a handful of points — far
        // too few for a ground plane to mean anything.
        if src_points.len() < 100_000 {
            eprintln!("skipping: {src} has only {} points, too few to fit ground \
                       (set HAIVIZ_TEST_COPC to a real file)", src_points.len());
            return;
        }

        // Centre the footprint on the point centroid, not on the middle of the
        // header bounds: survey data is often a strip hugging one edge, and a
        // footprint over the empty part would fit a plane to scattered noise
        // and still "pass" by agreeing with itself.
        let n = src_points.len() as f64;
        let cx = src_points.iter().map(|p| p.x).sum::<f64>() / n;
        let cy = src_points.iter().map(|p| p.y).sum::<f64>() / n;
        let half = ((b.max.x - b.min.x).min(b.max.y - b.min.y)) * 0.25;
        let (x0, x1) = (cx - half, cx + half);
        let (y0, y1) = (cy - half, cy + half);
        let polygon = vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)];
        let (z_lo, z_hi) = (b.min.z - 1.0, b.max.z + 1.0);

        let samples: Vec<(f64, f64, f64)> =
            src_points.iter().map(|p| (p.x, p.y, p.z)).collect();
        let fit = crate::classify::fit_ground_plane(&samples, &polygon, z_lo, z_hi)
            .expect("real ground should fit");
        eprintln!(
            "fit: {} cells from {} points, slope {:.2}deg, rms {:.3}m",
            fit.cells_used, fit.points_considered, fit.slope_degrees(), fit.rms,
        );
        assert!(fit.cells_used >= 8);
        // Over real ground the fit should be tight and near-horizontal. A loose
        // bound here would let a fit through scattered noise pass unnoticed.
        assert!(fit.rms < 0.5, "loose ground fit: rms {:.3}m", fit.rms);
        assert!(fit.slope_degrees() < 15.0, "implausible ground slope: {:.1}deg", fit.slope_degrees());
        assert!(fit.points_considered > 10_000, "only {} points in the footprint", fit.points_considered);

        let offset = 1.0;
        let op = ClassOp::new(
            0,
            crate::classify::ClassRegion::AboveGround {
                polygon: polygon.clone(),
                view: None,
                origin: fit.origin,
                plane: fit.plane,
                offset,
                z_min: z_lo,
                z_max: z_hi,
            },
            7,
            crate::classify::ClassMask::ALL,
        );

        // What the op should select, computed straight from the definition.
        let expected: usize = src_points.iter()
            .filter(|p| {
                p.z >= fit.height_at(p.x, p.y) + offset
                    && p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1
                    && effective_class(p) != 7
            })
            .count();
        assert!(expected > 0, "the footprint must contain something standing up");
        assert!(expected < src_points.len(), "it must not select the whole file");
        eprintln!("expecting {expected} of {} points", src_points.len());

        let mut edits = ClassEdits::default();
        edits.push(op.clone());

        let dir = std::env::temp_dir().join("haiviz-export-test");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("aboveground.laz");
        let (tx, _rx) = channel();
        let (written, dropped) =
            export_copc(&src, &out, &edits, &ExportFilter::default(), &tx).unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(written as usize, src_points.len());

        let out_points = read_all(out.to_str().unwrap());
        let got = out_points.iter().filter(|p| effective_class(p) == 7).count();
        assert_eq!(got, expected, "export marked a different set than the op defines");

        // And nothing at or below the fitted ground was touched.
        for p in &out_points {
            if effective_class(p) == 7 {
                assert!(
                    p.z >= fit.height_at(p.x, p.y) + offset,
                    "a point below the ground plane was reclassified",
                );
            }
        }
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn exported_file_is_not_advertised_as_copc() {
        let Some(src) = test_copc() else {
            eprintln!("skipping: no COPC test file (set HAIVIZ_TEST_COPC)");
            return;
        };
        let dir = std::env::temp_dir().join("haiviz-export-test");
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("novlr.laz");
        let (tx, _rx) = channel();
        export_copc(&src, &out, &ClassEdits::default(), &ExportFilter::default(), &tx).unwrap();

        let header = las::Reader::from_path(&out).unwrap().header().clone();
        assert!(header.copc_info_vlr().is_none(), "output must not claim to be COPC");
        assert!(!header.evlrs().iter().any(|v| v.is_copc_hierarchy()));
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn plain_laz_export_round_trips() {
        let Some(src) = test_copc() else { return };
        // Make a plain LAZ first, then export that through the non-COPC path.
        let dir = std::env::temp_dir().join("haiviz-export-test");
        std::fs::create_dir_all(&dir).unwrap();
        let plain = dir.join("plain.laz");
        let (tx0, _r0) = channel();
        export_copc(&src, &plain, &ClassEdits::default(), &ExportFilter::default(), &tx0).unwrap();

        let plain_str = plain.to_str().unwrap().to_string();
        let b = bounds(&plain_str);
        let src_points = read_all(&plain_str);
        let op = full_width_op(&b, 18, None);
        let expected_hit = src_points.iter()
            .filter(|p| op.contains(p.x, p.y, p.z) && effective_class(p) != 18)
            .count();
        assert!(expected_hit > 0);

        let mut edits = ClassEdits::default();
        edits.push(op.clone());
        let out = dir.join("plain_out.laz");
        let (tx, _rx) = channel();
        let (written, dropped) =
            export_plain(&plain_str, &out, &edits, &ExportFilter::default(), &tx).unwrap();
        assert_eq!(dropped, 0);
        assert_eq!(written as usize, src_points.len());

        let out_points = read_all(out.to_str().unwrap());
        for p in &out_points {
            if op.contains(p.x, p.y, p.z) {
                assert_eq!(effective_class(p), 18);
            }
        }
        let _ = std::fs::remove_file(&plain);
        let _ = std::fs::remove_file(&out);
    }
}

/// Tests for the force-load selection: the pinning that guarantees a section
/// corridor holds every point in it, not just the resident LOD sample.
#[cfg(test)]
mod corridor_tests {
    use super::tests::test_copc;
    use crate::raw_miniquad::{corridor_meets_aabb_2d, read_full_hierarchy, voxel_bounds};
    use las::copc::CopcEntryReader;
    use std::fs::File;
    use std::io::BufReader;

    /// Whether a fixture has enough octree structure for corridor selection to
    /// mean anything: several nodes, spread over more than one level.
    fn is_substantial(entries: &[crate::raw_miniquad::RawCopcEntry]) -> bool {
        let levels: std::collections::HashSet<i32> = entries.iter().map(|e| e.key.l).collect();
        entries.len() >= 8 && levels.len() >= 2
    }

    /// The property the whole design rests on: if any point of a node falls
    /// inside the corridor, that node must be pinned. A miss here is silent —
    /// the operator would classify a slice with points quietly absent from it.
    #[test]
    fn pinning_selects_every_node_holding_corridor_points() {
        let Some(src) = test_copc() else {
            eprintln!("skipping: no COPC test file (set HAIVIZ_TEST_COPC)");
            return;
        };
        let file = BufReader::new(File::open(&src).unwrap());
        let reader = CopcEntryReader::new(file).unwrap();
        let info = reader.header().copc_info_vlr().unwrap();
        let b = reader.header().bounds();
        let entries = read_full_hierarchy(&reader);
        assert!(!entries.is_empty());
        drop(reader);

        // The `las` crate's bundled autzen sample is a single tiny node — it
        // exercises nothing about corridor selection. Only run against a
        // fixture with real octree structure.
        if !is_substantial(&entries) {
            eprintln!("skipping: {src} is too small to exercise corridor selection \
                       (set HAIVIZ_TEST_COPC to a real file)");
            return;
        }

        // Corridors are aimed at the point centroid, not at the middle of the
        // header bounds: survey data is routinely a strip hugging one edge of
        // its extent, and lines through the bbox centre can miss it entirely.
        let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u64);
        for entry in &entries {
            for p in super::read_node(&src, entry).unwrap() {
                sx += p.x; sy += p.y; n += 1;
            }
        }
        assert!(n > 0);
        let (cx, cy) = (sx / n as f64, sy / n as f64);
        let span = (b.max.x - b.min.x).max(b.max.y - b.min.y);
        let corridors = [
            ((b.min.x, cy), (b.max.x, cy), 2.0),
            ((cx, b.min.y), (cx, b.max.y), 5.0),
            ((cx - span, cy - span), (cx + span, cy + span), 3.0),
            ((cx - span, cy + span), (cx + span, cy - span), 0.5),
            // Deliberately clear of the data: nothing should be selected.
            ((cx - span, cy + span * 8.0), (cx + span, cy + span * 8.0), 1.0),
        ];

        for (i, (start, end, depth)) in corridors.iter().enumerate() {
            let mut checked_nodes = 0;
            let mut nodes_with_points = 0;
            for entry in &entries {
                let (xmin, ymin, _zmin, xmax, ymax, _zmax) =
                    voxel_bounds(&entry.key, info.center_x, info.center_y, info.center_z, info.halfsize);
                let pinned = corridor_meets_aabb_2d(
                    *start, *end, *depth, (xmin, ymin), (xmax, ymax),
                );

                let points = super::read_node(&src, entry).unwrap();
                // Same in/out test the section extraction uses.
                let (dx, dy) = (end.0 - start.0, end.1 - start.1);
                let len = (dx * dx + dy * dy).sqrt();
                let dir = (dx / len, dy / len);
                let normal = (-dir.1, dir.0);
                let inside = points.iter().filter(|p| {
                    let (rx, ry) = (p.x - start.0, p.y - start.1);
                    let along = rx * dir.0 + ry * dir.1;
                    let perp = rx * normal.0 + ry * normal.1;
                    along >= 0.0 && along <= len && perp.abs() <= *depth
                }).count();

                if inside > 0 {
                    nodes_with_points += 1;
                    assert!(
                        pinned,
                        "corridor {i}: node {:?} holds {inside} corridor points but was not pinned",
                        entry.key,
                    );
                }
                checked_nodes += 1;
            }
            assert_eq!(checked_nodes, entries.len());
            if i < 4 {
                assert!(nodes_with_points > 0, "corridor {i} should cross the data");
            } else {
                assert_eq!(nodes_with_points, 0, "corridor {i} should miss the data");
            }
        }
    }

    /// The selection must also be reasonably tight — a bounding-box stand-in
    /// for a diagonal corridor would pin most of the file and blow the budget.
    #[test]
    fn diagonal_corridor_does_not_pin_the_whole_file() {
        let Some(src) = test_copc() else { return };
        let file = BufReader::new(File::open(&src).unwrap());
        let reader = CopcEntryReader::new(file).unwrap();
        let info = reader.header().copc_info_vlr().unwrap();
        let b = reader.header().bounds();
        let entries = read_full_hierarchy(&reader);
        if !is_substantial(&entries) {
            eprintln!("skipping: {src} is too small to exercise corridor selection");
            return;
        }

        // A narrow diagonal corner to corner: its axis-aligned bounding box is
        // the whole extent, so a bbox test would pin every node.
        let pinned = entries.iter().filter(|e| {
            let (xmin, ymin, _, xmax, ymax, _) =
                voxel_bounds(&e.key, info.center_x, info.center_y, info.center_z, info.halfsize);
            corridor_meets_aabb_2d(
                (b.min.x, b.min.y), (b.max.x, b.max.y), 0.5, (xmin, ymin), (xmax, ymax),
            )
        }).count();

        assert!(pinned > 0);
        assert!(
            pinned < entries.len(),
            "a 1 m-wide diagonal pinned all {} nodes — the test is not discriminating",
            entries.len(),
        );
        eprintln!("diagonal corridor pins {pinned}/{} nodes", entries.len());
    }
}
