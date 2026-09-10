//! GeoPackage read and write, in pure Rust, with no SQLite dependency.
//!
//! Every mainstream SQLite crate wraps the C library, and the one genuine
//! pure-Rust engine is a pre-1.0 MVCC database — a lot of dependency for a
//! table of blobs. So this is a GeoPackage-specific SQLite codec instead.
//!
//! This is emphatically **not** "implement SQLite". Two constraints we control
//! shrink the job drastically:
//!
//! - We only ever **write a fresh file**. No freelists, no page reuse, no
//!   incremental vacuum, no journalling, no updates in place.
//! - We only ever **read feature tables**. No SQL engine, no query planner, no
//!   expression evaluation, no triggers, no virtual tables.
//!
//! What is left is varints, the record serial-type encoding, table and index
//! b-tree traversal, overflow pages — a long linestring blob comfortably
//! exceeds what fits on a page — and reading `sqlite_schema` to find the
//! feature tables.
//!
//! ## What a real-world file contains
//!
//! A GeoPackage written by QGIS or GDAL carries far more than the three
//! required metadata tables: rtree virtual tables and their shadow tables,
//! a dozen triggers, `gpkg_ogr_contents`, tile-matrix tables, `gpkg_extensions`.
//! The reader must **skip what it does not understand** rather than fail on it,
//! so layers are discovered by joining `gpkg_contents` (`data_type='features'`)
//! against `gpkg_geometry_columns`, and everything else in `sqlite_schema` is
//! ignored.
//!
//! ## What is deliberately not supported
//!
//! A `-wal` sidecar means committed rows live outside the main file. Reading
//! WAL is a separate chunk of work, and reading the main file alone would
//! silently return stale data — so a file with a non-empty WAL is refused with
//! an explanation instead.

// The codec lands with phase 2b; `polylines.rs` starts calling it in the same
// phase, but the low-level pieces (Db, Value accessors, build_database) are a
// deliberately complete SQLite surface that only the tests exercise.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const PAGE_SIZE: usize = 4096;
/// Byte offsets fixed by the SQLite file header, so these can be written
/// directly and `PRAGMA` is never needed.
const OFF_USER_VERSION: usize = 60;
const OFF_APPLICATION_ID: usize = 68;
/// "GPKG".
const APPLICATION_ID: u32 = 0x4750_4B47;
/// GeoPackage 1.3.
const USER_VERSION: u32 = 10300;

/// B-tree page types.
const PAGE_INTERIOR_INDEX: u8 = 2;
const PAGE_INTERIOR_TABLE: u8 = 5;
const PAGE_LEAF_INDEX: u8 = 10;
const PAGE_LEAF_TABLE: u8 = 13;

type Res<T> = Result<T, String>;

// =========================================================================
// Values and records
// =========================================================================

/// One SQLite column value. The five storage classes, nothing else.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl Value {
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Real(r) => Some(*r as i64),
            Value::Text(t) => t.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Real(r) => Some(*r),
            Value::Text(t) => t.trim().parse().ok(),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self { Value::Text(t) => Some(t), _ => None }
    }

    pub fn as_blob(&self) -> Option<&[u8]> {
        match self { Value::Blob(b) => Some(b), _ => None }
    }

    pub fn is_null(&self) -> bool { matches!(self, Value::Null) }

    /// Render for the per-feature attribute map. `NULL` has no string form —
    /// callers leave the key out instead, which is what makes a NULL survive a
    /// round-trip rather than becoming an empty string.
    pub fn to_display(&self) -> String {
        match self {
            Value::Null => String::new(),
            Value::Int(i) => i.to_string(),
            Value::Real(r) => format!("{r}"),
            Value::Text(t) => t.clone(),
            Value::Blob(b) => format!("<{} bytes>", b.len()),
        }
    }
}

/// SQLite's variable-length integer: big-endian, 7 bits per byte, high bit
/// continues, except the ninth byte which contributes all 8 of its bits.
fn put_varint(out: &mut Vec<u8>, v: u64) {
    if v <= 0x7f {
        out.push(v as u8);
        return;
    }
    if v > 0x00ff_ffff_ffff_ffff {
        // Nine-byte form: eight 7-bit groups then a full byte.
        let mut buf = [0u8; 9];
        buf[8] = (v & 0xff) as u8;
        let mut rest = v >> 8;
        for i in (0..8).rev() {
            buf[i] = ((rest & 0x7f) as u8) | 0x80;
            rest >>= 7;
        }
        out.extend_from_slice(&buf);
        return;
    }
    let mut tmp = [0u8; 9];
    let mut n = 0;
    let mut rest = v;
    while rest > 0 {
        tmp[n] = (rest & 0x7f) as u8;
        rest >>= 7;
        n += 1;
    }
    for i in (0..n).rev() {
        let last = i == 0;
        out.push(if last { tmp[i] } else { tmp[i] | 0x80 });
    }
}

/// Decode a varint at `pos`. Returns the value and how many bytes it took.
fn get_varint(buf: &[u8], pos: usize) -> Res<(u64, usize)> {
    let mut v: u64 = 0;
    for i in 0..8 {
        let b = *buf.get(pos + i).ok_or("truncated varint")?;
        if i == 7 {
            // The ninth byte would be index 8; at index 7 without a
            // continuation we are done.
            v = (v << 7) | (b & 0x7f) as u64;
            if b & 0x80 == 0 { return Ok((v, 8)); }
            let last = *buf.get(pos + 8).ok_or("truncated varint")?;
            return Ok(((v << 8) | last as u64, 9));
        }
        v = (v << 7) | (b & 0x7f) as u64;
        if b & 0x80 == 0 { return Ok((v, i + 1)); }
    }
    unreachable!()
}

fn serial_type(v: &Value) -> u64 {
    match v {
        Value::Null => 0,
        Value::Int(i) => {
            let i = *i;
            if i == 0 { 8 }
            else if i == 1 { 9 }
            else if (-128..=127).contains(&i) { 1 }
            else if (-32768..=32767).contains(&i) { 2 }
            else if (-8_388_608..=8_388_607).contains(&i) { 3 }
            else if (-2_147_483_648..=2_147_483_647).contains(&i) { 4 }
            else if (-140_737_488_355_328..=140_737_488_355_327).contains(&i) { 5 }
            else { 6 }
        }
        Value::Real(_) => 7,
        Value::Blob(b) => 12 + 2 * b.len() as u64,
        Value::Text(t) => 13 + 2 * t.len() as u64,
    }
}

fn put_body(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Null => {}
        Value::Int(i) => {
            let i = *i;
            match serial_type(v) {
                8 | 9 => {}
                1 => out.push(i as u8),
                2 => out.extend_from_slice(&(i as i16).to_be_bytes()),
                3 => out.extend_from_slice(&(i as i32).to_be_bytes()[1..]),
                4 => out.extend_from_slice(&(i as i32).to_be_bytes()),
                5 => out.extend_from_slice(&i.to_be_bytes()[2..]),
                _ => out.extend_from_slice(&i.to_be_bytes()),
            }
        }
        Value::Real(r) => out.extend_from_slice(&r.to_be_bytes()),
        Value::Text(t) => out.extend_from_slice(t.as_bytes()),
        Value::Blob(b) => out.extend_from_slice(b),
    }
}

/// Encode one row in SQLite's record format: a header of serial types,
/// prefixed by its own length, then the bodies.
fn encode_record(vals: &[Value]) -> Vec<u8> {
    let mut types = Vec::new();
    for v in vals { put_varint(&mut types, serial_type(v)); }
    // The header length includes the varint that states it, which can itself
    // grow when the length crosses 127 — so try the short form and widen once.
    let mut header_len = types.len() + 1;
    if header_len > 127 { header_len = types.len() + 2; }
    if header_len > 16383 { header_len = types.len() + 3; }

    let mut out = Vec::with_capacity(header_len + 32);
    put_varint(&mut out, header_len as u64);
    debug_assert_eq!(out.len() + types.len(), header_len);
    out.extend_from_slice(&types);
    for v in vals { put_body(&mut out, v); }
    out
}

fn decode_record(buf: &[u8]) -> Res<Vec<Value>> {
    let (header_len, n) = get_varint(buf, 0)?;
    let header_len = header_len as usize;
    if header_len > buf.len() { return Err("record header overruns payload".into()); }
    let mut types = Vec::new();
    let mut pos = n;
    while pos < header_len {
        let (t, used) = get_varint(buf, pos)?;
        types.push(t);
        pos += used;
    }

    let mut out = Vec::with_capacity(types.len());
    let mut body = header_len;
    let take = |body: &mut usize, len: usize| -> Res<&[u8]> {
        let s = buf.get(*body..*body + len).ok_or("record body overruns payload")?;
        *body += len;
        Ok(s)
    };
    for t in types {
        let v = match t {
            0 => Value::Null,
            1..=6 => {
                let len = match t { 1 => 1, 2 => 2, 3 => 3, 4 => 4, 5 => 6, _ => 8 };
                let s = take(&mut body, len)?;
                // Sign-extend from the stored width.
                let mut acc: i64 = if s[0] & 0x80 != 0 { -1 } else { 0 };
                for b in s { acc = (acc << 8) | *b as i64; }
                Value::Int(acc)
            }
            7 => {
                let s = take(&mut body, 8)?;
                Value::Real(f64::from_be_bytes(s.try_into().unwrap()))
            }
            8 => Value::Int(0),
            9 => Value::Int(1),
            10 | 11 => return Err("reserved serial type in record".into()),
            t if t % 2 == 0 => {
                let len = ((t - 12) / 2) as usize;
                Value::Blob(take(&mut body, len)?.to_vec())
            }
            t => {
                let len = ((t - 13) / 2) as usize;
                let s = take(&mut body, len)?;
                Value::Text(String::from_utf8_lossy(s).into_owned())
            }
        };
        out.push(v);
    }
    Ok(out)
}

// =========================================================================
// Reader
// =========================================================================

pub struct SchemaRow {
    pub kind: String,
    pub name: String,
    pub tbl_name: String,
    pub rootpage: u32,
    pub sql: String,
}

pub struct Db {
    data: Vec<u8>,
    page_size: usize,
    /// Page size minus the reserved tail. Every payload calculation uses this,
    /// not the page size.
    usable: usize,
    n_pages: u32,
}

impl Db {
    pub fn open(path: &str) -> Res<Db> {
        // A WAL sidecar means committed rows live outside the main file.
        // Reading the main file alone would quietly return the state before
        // whatever the other program last wrote, which is worse than refusing.
        let wal = format!("{path}-wal");
        if fs::metadata(&wal).map(|m| m.len() > 0).unwrap_or(false) {
            return Err(format!(
                "{}: a write-ahead log ({}) is present, so the file on disk is not \
                 the whole story. Close it in the other program, or run \
                 `sqlite3 <file> \"PRAGMA wal_checkpoint(TRUNCATE);\"` first.",
                Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path),
                Path::new(&wal).file_name().and_then(|n| n.to_str()).unwrap_or(&wal),
            ));
        }

        let data = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        if data.len() < 100 || &data[..16] != b"SQLite format 3\0" {
            return Err(format!("{path}: not a SQLite database"));
        }
        let raw = u16::from_be_bytes([data[16], data[17]]) as usize;
        let page_size = if raw == 1 { 65536 } else { raw };
        if page_size < 512 || (page_size & (page_size - 1)) != 0 {
            return Err(format!("{path}: implausible page size {page_size}"));
        }
        // Byte 18 is the write version: 2 means WAL mode. The sidecar check
        // above catches the case that matters, but a WAL-mode file with no
        // sidecar has been checkpointed and is safe to read.
        let reserved = data[20] as usize;
        if reserved >= page_size { return Err(format!("{path}: implausible reserved space")); }
        // Text encoding is a 4-byte big-endian field, not a byte: 1 = UTF-8,
        // 2 and 3 are the UTF-16 variants, which this reader does not decode.
        let encoding = u32::from_be_bytes([data[56], data[57], data[58], data[59]]);
        if encoding != 1 {
            return Err(format!("{path}: only UTF-8 databases are supported (encoding {encoding})"));
        }
        let n_pages = u32::from_be_bytes([data[28], data[29], data[30], data[31]]);
        let n_pages = if n_pages == 0 || (n_pages as usize) * page_size > data.len() {
            (data.len() / page_size) as u32
        } else {
            n_pages
        };
        Ok(Db { data, page_size, usable: page_size - reserved, n_pages })
    }

    fn page(&self, n: u32) -> Res<&[u8]> {
        if n == 0 || n > self.n_pages { return Err(format!("page {n} out of range")); }
        let start = (n as usize - 1) * self.page_size;
        self.data.get(start..start + self.page_size).ok_or_else(|| format!("page {n} truncated"))
    }

    /// `sqlite_schema` always lives at page 1 and always has the same five
    /// columns, so it needs no bootstrapping.
    pub fn schema(&self) -> Res<Vec<SchemaRow>> {
        let mut out = Vec::new();
        for (_, vals) in self.rows(1)? {
            if vals.len() < 5 { continue; }
            out.push(SchemaRow {
                kind: vals[0].as_text().unwrap_or_default().to_string(),
                name: vals[1].as_text().unwrap_or_default().to_string(),
                tbl_name: vals[2].as_text().unwrap_or_default().to_string(),
                rootpage: vals[3].as_i64().unwrap_or(0).max(0) as u32,
                sql: vals[4].as_text().unwrap_or_default().to_string(),
            });
        }
        Ok(out)
    }

    /// Every row of one table, in rowid order.
    pub fn rows(&self, rootpage: u32) -> Res<Vec<(i64, Vec<Value>)>> {
        let mut out = Vec::new();
        let mut seen = 0usize;
        self.walk_table(rootpage, &mut out, &mut seen, 0)?;
        Ok(out)
    }

    fn walk_table(
        &self,
        page_no: u32,
        out: &mut Vec<(i64, Vec<Value>)>,
        seen: &mut usize,
        depth: u32,
    ) -> Res<()> {
        // A corrupt or hostile file can point a child back at an ancestor.
        // Depth alone bounds that without tracking a visited set.
        if depth > 32 { return Err("b-tree deeper than 32 levels".into()); }
        let page = self.page(page_no)?;
        // Page 1 carries the 100-byte database header before its b-tree header.
        let base = if page_no == 1 { 100 } else { 0 };
        let hdr = &page[base..];
        let kind = hdr[0];
        let n_cells = u16::from_be_bytes([hdr[3], hdr[4]]) as usize;
        let hdr_len = if kind == PAGE_INTERIOR_TABLE || kind == PAGE_INTERIOR_INDEX { 12 } else { 8 };

        let cell_ptr = |i: usize| -> Res<usize> {
            let at = base + hdr_len + i * 2;
            let p = page.get(at..at + 2).ok_or("cell pointer array overruns page")?;
            Ok(u16::from_be_bytes([p[0], p[1]]) as usize)
        };

        match kind {
            PAGE_LEAF_TABLE => {
                for i in 0..n_cells {
                    let at = cell_ptr(i)?;
                    let (payload_len, a) = get_varint(page, at)?;
                    let (rowid, b) = get_varint(page, at + a)?;
                    let body = self.read_payload(page, at + a + b, payload_len as usize, true)?;
                    *seen += 1;
                    if *seen > 50_000_000 { return Err("implausible row count".into()); }
                    out.push((rowid as i64, decode_record(&body)?));
                }
            }
            PAGE_INTERIOR_TABLE => {
                for i in 0..n_cells {
                    let at = cell_ptr(i)?;
                    let child = u32::from_be_bytes(
                        page.get(at..at + 4).ok_or("interior cell truncated")?.try_into().unwrap(),
                    );
                    self.walk_table(child, out, seen, depth + 1)?;
                }
                let right = u32::from_be_bytes(
                    page.get(base + 8..base + 12).ok_or("missing right pointer")?.try_into().unwrap(),
                );
                self.walk_table(right, out, seen, depth + 1)?;
            }
            // Index pages never hold table rows. They are reached only if a
            // rootpage was misread, so say so rather than returning nothing.
            PAGE_LEAF_INDEX | PAGE_INTERIOR_INDEX => {
                return Err(format!("page {page_no} is an index, not a table"));
            }
            other => return Err(format!("page {page_no}: unknown b-tree page type {other}")),
        }
        Ok(())
    }

    /// Assemble a cell payload, following the overflow chain when the record
    /// does not fit on its page — which for a linestring of any length is the
    /// normal case, not an edge case.
    fn read_payload(&self, page: &[u8], at: usize, total: usize, leaf_table: bool) -> Res<Vec<u8>> {
        let local = local_payload_size(total, self.usable, leaf_table);
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(page.get(at..at + local).ok_or("cell payload overruns page")?);
        if local == total { return Ok(out); }

        let mut next = u32::from_be_bytes(
            page.get(at + local..at + local + 4).ok_or("missing overflow pointer")?.try_into().unwrap(),
        );
        let mut guard = 0;
        while next != 0 && out.len() < total {
            guard += 1;
            if guard > 1_000_000 { return Err("overflow chain does not terminate".into()); }
            let p = self.page(next)?;
            next = u32::from_be_bytes(p[0..4].try_into().unwrap());
            let want = (total - out.len()).min(self.usable - 4);
            out.extend_from_slice(p.get(4..4 + want).ok_or("overflow page truncated")?);
        }
        if out.len() != total { return Err("overflow chain ended early".into()); }
        Ok(out)
    }
}

/// How much of a payload of `total` bytes lives on the page itself.
///
/// SQLite spreads payloads so that a page never holds a uselessly small tail:
/// below `max_local` everything is local, above it the local part is the
/// remainder that keeps overflow pages full, clamped to `min_local`.
fn local_payload_size(total: usize, usable: usize, leaf_table: bool) -> usize {
    let max_local = if leaf_table { usable - 35 } else { ((usable - 12) * 64 / 255) - 23 };
    let min_local = ((usable - 12) * 32 / 255) - 23;
    if total <= max_local { return total; }
    let k = min_local + (total - min_local) % (usable - 4);
    if k <= max_local { k } else { min_local }
}

/// Column names from a `CREATE TABLE` statement.
///
/// Not a SQL parser: it takes the parenthesised column list, splits on commas
/// at depth zero, drops anything starting with a table-constraint keyword, and
/// takes the first identifier of what is left. That is enough for the tables
/// GeoPackage defines and for the ones QGIS and GDAL write, and anything it
/// misreads costs an attribute column rather than a wrong geometry.
pub fn column_names(sql: &str) -> Vec<String> {
    let Some(open) = sql.find('(') else { return Vec::new() };
    let body = &sql[open + 1..];
    let mut depth = 0i32;
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in body.chars() {
        match ch {
            '(' => { depth += 1; cur.push(ch); }
            ')' if depth == 0 => break,
            ')' => { depth -= 1; cur.push(ch); }
            ',' if depth == 0 => { parts.push(std::mem::take(&mut cur)); }
            _ => cur.push(ch),
        }
    }
    parts.push(cur);

    const CONSTRAINTS: [&str; 7] =
        ["constraint", "primary", "unique", "check", "foreign", "exclude", "key"];
    parts.iter()
        .filter_map(|p| {
            let p = p.trim();
            let first = p.split([' ', '\t', '\n', '(']).next()?;
            let bare = first.trim_matches(|c| c == '"' || c == '`' || c == '[' || c == ']');
            if bare.is_empty() { return None; }
            if CONSTRAINTS.contains(&bare.to_ascii_lowercase().as_str()) { return None; }
            Some(bare.to_string())
        })
        .collect()
}

/// Declared column types, aligned with `column_names`. Used to put an
/// attribute back in the type it came out as.
fn column_types(sql: &str) -> Vec<String> {
    let Some(open) = sql.find('(') else { return Vec::new() };
    let body = &sql[open + 1..];
    let mut depth = 0i32;
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in body.chars() {
        match ch {
            '(' => { depth += 1; cur.push(ch); }
            ')' if depth == 0 => break,
            ')' => { depth -= 1; cur.push(ch); }
            ',' if depth == 0 => { parts.push(std::mem::take(&mut cur)); }
            _ => cur.push(ch),
        }
    }
    parts.push(cur);

    const CONSTRAINTS: [&str; 7] =
        ["constraint", "primary", "unique", "check", "foreign", "exclude", "key"];
    parts.iter()
        .filter_map(|p| {
            let p = p.trim();
            let mut it = p.split_whitespace();
            let first = it.next()?;
            let bare = first.trim_matches(|c| c == '"' || c == '`' || c == '[' || c == ']');
            if bare.is_empty() { return None; }
            if CONSTRAINTS.contains(&bare.to_ascii_lowercase().as_str()) { return None; }
            Some(it.next().unwrap_or("").to_ascii_uppercase())
        })
        .collect()
}

// =========================================================================
// Writer
// =========================================================================

pub struct Table {
    pub name: String,
    pub sql: String,
    /// Rows in rowid order. The writer does not sort them.
    pub rows: Vec<(i64, Vec<Value>)>,
}

pub struct Index {
    pub name: String,
    pub tbl_name: String,
    /// `None` for an automatic index, which `sqlite_schema` records with a
    /// NULL `sql` because the constraint in the table's own DDL is its
    /// definition.
    pub sql: Option<String>,
    /// Key values, with the rowid appended as the final column. Must be sorted.
    pub keys: Vec<Vec<Value>>,
}

struct Builder {
    pages: Vec<Vec<u8>>,
}

impl Builder {
    fn new() -> Builder {
        // Page 1 is reserved for the sqlite_schema root; it is filled last.
        Builder { pages: vec![vec![0u8; PAGE_SIZE]] }
    }

    fn alloc(&mut self) -> u32 {
        self.pages.push(vec![0u8; PAGE_SIZE]);
        self.pages.len() as u32
    }

    fn page_mut(&mut self, n: u32) -> &mut [u8] {
        &mut self.pages[n as usize - 1]
    }

    /// Split a payload into its on-page part and a chain of overflow pages,
    /// returning the bytes to place in the cell after the header.
    fn payload_cell_tail(&mut self, payload: &[u8], leaf_table: bool) -> Vec<u8> {
        let local = local_payload_size(payload.len(), PAGE_SIZE, leaf_table);
        let mut tail = payload[..local].to_vec();
        if local == payload.len() { return tail; }

        let mut rest = &payload[local..];
        let per_page = PAGE_SIZE - 4;
        // Allocate the whole chain first so each page can point at the next.
        let n = rest.len().div_ceil(per_page);
        let chain: Vec<u32> = (0..n).map(|_| self.alloc()).collect();
        for (i, &p) in chain.iter().enumerate() {
            let take = rest.len().min(per_page);
            let next = if i + 1 < chain.len() { chain[i + 1] } else { 0 };
            let page = self.page_mut(p);
            page[0..4].copy_from_slice(&next.to_be_bytes());
            page[4..4 + take].copy_from_slice(&rest[..take]);
            rest = &rest[take..];
        }
        tail.extend_from_slice(&chain[0].to_be_bytes());
        tail
    }
}

/// Write a b-tree page from finished cells.
///
/// Cells are laid out from the end of the page backwards, with the pointer
/// array growing forward from just after the header — the layout SQLite itself
/// uses, and the one its readers assume.
fn write_btree_page(page: &mut [u8], base: usize, kind: u8, cells: &[Vec<u8>], right: Option<u32>) {
    let hdr_len = if right.is_some() { 12 } else { 8 };
    let mut content = PAGE_SIZE;
    let mut ptrs = Vec::with_capacity(cells.len());
    for c in cells {
        content -= c.len();
        page[content..content + c.len()].copy_from_slice(c);
        ptrs.push(content as u16);
    }
    page[base] = kind;
    page[base + 1..base + 3].copy_from_slice(&0u16.to_be_bytes()); // no freeblocks
    page[base + 3..base + 5].copy_from_slice(&(cells.len() as u16).to_be_bytes());
    // A content start of 65536 is stored as 0; only reachable for an empty
    // page at the maximum page size, but the encoding is free to honour.
    let start = if content == 65536 { 0u16 } else { content as u16 };
    page[base + 5..base + 7].copy_from_slice(&start.to_be_bytes());
    page[base + 7] = 0; // no fragmented free bytes
    if let Some(r) = right {
        page[base + 8..base + 12].copy_from_slice(&r.to_be_bytes());
    }
    for (i, p) in ptrs.iter().enumerate() {
        let at = base + hdr_len + i * 2;
        page[at..at + 2].copy_from_slice(&p.to_be_bytes());
    }
}

/// Usable bytes for cells on a page, given its header and pointer array.
fn cell_space(base: usize, hdr_len: usize, n_cells: usize) -> usize {
    PAGE_SIZE - base - hdr_len - n_cells * 2
}

/// Build a table b-tree bottom-up and return its root page.
///
/// `forced_root` places the top node on a page that is already spoken for —
/// page 1 for `sqlite_schema`, whose root the format fixes at page 1 and whose
/// b-tree header therefore starts after the 100-byte database header.
fn build_table_btree(
    b: &mut Builder,
    rows: &[(i64, Vec<Value>)],
    forced_root: Option<u32>,
) -> Res<u32> {
    // One cell per row: payload length, rowid, then the payload — or its local
    // part plus a pointer to the overflow chain.
    let mut cells: Vec<(i64, Vec<u8>)> = Vec::with_capacity(rows.len());
    for (rowid, vals) in rows {
        let payload = encode_record(vals);
        let tail = b.payload_cell_tail(&payload, true);
        let mut cell = Vec::with_capacity(tail.len() + 18);
        put_varint(&mut cell, payload.len() as u64);
        put_varint(&mut cell, *rowid as u64);
        cell.extend_from_slice(&tail);
        cells.push((*rowid, cell));
    }

    let root_base = if forced_root == Some(1) { 100 } else { 0 };

    // Everything on one page is the common case, and for `sqlite_schema` it is
    // what keeps page 1 a leaf rather than an interior node.
    let lens: Vec<usize> = cells.iter().map(|(_, c)| c.len()).collect();
    if fits_on_one_page(&lens, root_base, 8) {
        let root = forced_root.unwrap_or_else(|| b.alloc());
        let owned: Vec<Vec<u8>> = cells.into_iter().map(|(_, c)| c).collect();
        write_btree_page(b.page_mut(root), root_base, PAGE_LEAF_TABLE, &owned, None);
        return Ok(root);
    }

    // Otherwise fill leaves greedily, then stack interior levels over them.
    let mut level: Vec<(u32, i64)> = Vec::new();
    let mut batch: Vec<Vec<u8>> = Vec::new();
    let mut batch_max = i64::MIN;
    for (rowid, cell) in cells {
        if !batch.is_empty() && !fits_with(&batch, &cell, 0, 8) {
            let p = b.alloc();
            write_btree_page(b.page_mut(p), 0, PAGE_LEAF_TABLE, &batch, None);
            level.push((p, batch_max));
            batch.clear();
        }
        batch_max = rowid;
        batch.push(cell);
    }
    if !batch.is_empty() {
        let p = b.alloc();
        write_btree_page(b.page_mut(p), 0, PAGE_LEAF_TABLE, &batch, None);
        level.push((p, batch_max));
    }

    build_table_interiors(b, level, forced_root, root_base)
}

/// One table-interior cell: the child page, then the largest rowid beneath it.
///
/// Unlike an index b-tree, that key is *also* still present in the child — a
/// table interior cell is a signpost, not an entry.
fn table_interior_cell(child: u32, max_key: i64) -> Vec<u8> {
    let mut c = Vec::with_capacity(13);
    c.extend_from_slice(&child.to_be_bytes());
    put_varint(&mut c, max_key as u64);
    c
}

/// Stack interior levels over `level` until one page holds the lot.
fn build_table_interiors(
    b: &mut Builder,
    mut level: Vec<(u32, i64)>,
    forced_root: Option<u32>,
    root_base: usize,
) -> Res<u32> {
    for _ in 0..32 {
        // The last child of any interior page is its right pointer rather than
        // a cell, so a level of N children needs N-1 cells.
        let cells: Vec<Vec<u8>> = level[..level.len() - 1].iter()
            .map(|(c, k)| table_interior_cell(*c, *k))
            .collect();
        let lens: Vec<usize> = cells.iter().map(|c| c.len()).collect();
        if fits_on_one_page(&lens, root_base, 12) {
            let root = forced_root.unwrap_or_else(|| b.alloc());
            let right = level.last().unwrap().0;
            write_btree_page(b.page_mut(root), root_base, PAGE_INTERIOR_TABLE, &cells, Some(right));
            return Ok(root);
        }

        // Too wide: emit a row of interior pages and go round again.
        let mut next: Vec<(u32, i64)> = Vec::new();
        let mut start = 0;
        while start < level.len() {
            let mut cells: Vec<Vec<u8>> = Vec::new();
            let mut end = start;
            while end < level.len() {
                let c = table_interior_cell(level[end].0, level[end].1);
                if !cells.is_empty() && !fits_with(&cells, &c, 0, 12) { break; }
                cells.push(c);
                end += 1;
            }
            // The last child taken becomes this page's right pointer.
            let (right_page, right_key) = level[end - 1];
            cells.pop();
            let p = b.alloc();
            write_btree_page(b.page_mut(p), 0, PAGE_INTERIOR_TABLE, &cells, Some(right_page));
            next.push((p, right_key));
            start = end;
        }
        if next.len() >= level.len() {
            return Err("table b-tree is not converging".into());
        }
        level = next;
    }
    Err("table b-tree would exceed 32 levels".into())
}

/// Build an index b-tree bottom-up and return its root page.
///
/// Unlike a table b-tree, a key placed in an interior page is the divider and
/// does **not** also live in a leaf, so entries are promoted out rather than
/// copied up.
fn build_index_btree(b: &mut Builder, keys: &[Vec<Value>]) -> Res<u32> {
    let mut cells: Vec<Vec<u8>> = Vec::with_capacity(keys.len());
    for vals in keys {
        let payload = encode_record(vals);
        let tail = b.payload_cell_tail(&payload, false);
        let mut cell = Vec::with_capacity(tail.len() + 9);
        put_varint(&mut cell, payload.len() as u64);
        cell.extend_from_slice(&tail);
        cells.push(cell);
    }

    let lens: Vec<usize> = cells.iter().map(|c| c.len()).collect();
    if fits_on_one_page(&lens, 0, 8) {
        let root = b.alloc();
        write_btree_page(b.page_mut(root), 0, PAGE_LEAF_INDEX, &cells, None);
        return Ok(root);
    }

    // Group into leaf-sized runs, then promote the first entry of every run
    // after the first to be the divider ahead of it.
    let mut groups: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut batch: Vec<Vec<u8>> = Vec::new();
    for cell in cells {
        if !batch.is_empty() && !fits_with(&batch, &cell, 0, 8) {
            groups.push(std::mem::take(&mut batch));
        }
        batch.push(cell);
    }
    if !batch.is_empty() { groups.push(batch); }

    let mut leaves: Vec<u32> = Vec::new();
    let mut dividers: Vec<Vec<u8>> = Vec::new();
    for (i, mut g) in groups.into_iter().enumerate() {
        if i > 0 {
            if g.len() < 2 {
                return Err("index b-tree run too small to promote a divider from".into());
            }
            dividers.push(g.remove(0));
        }
        let p = b.alloc();
        write_btree_page(b.page_mut(p), 0, PAGE_LEAF_INDEX, &g, None);
        leaves.push(p);
    }
    if dividers.len() + 1 != leaves.len() {
        return Err("index b-tree divider count does not match its leaves".into());
    }

    let interior: Vec<Vec<u8>> = dividers.into_iter().enumerate().map(|(i, d)| {
        let mut c = Vec::with_capacity(d.len() + 4);
        c.extend_from_slice(&leaves[i].to_be_bytes());
        c.extend_from_slice(&d);
        c
    }).collect();
    let lens: Vec<usize> = interior.iter().map(|c| c.len()).collect();
    if !fits_on_one_page(&lens, 0, 12) {
        // Reaching this needs tens of thousands of layers in one file.
        return Err("index b-tree needs more than two levels".into());
    }
    let root = b.alloc();
    let right = *leaves.last().unwrap();
    write_btree_page(b.page_mut(root), 0, PAGE_INTERIOR_INDEX, &interior, Some(right));
    Ok(root)
}

/// Would `batch` still fit on a page with `extra` added?
fn fits_with(batch: &[Vec<u8>], extra: &[u8], base: usize, hdr_len: usize) -> bool {
    let total: usize = batch.iter().map(|c| c.len()).sum::<usize>() + extra.len();
    let n = batch.len() + 1;
    total + n * 2 <= cell_space(base, hdr_len, 0)
}

fn fits_on_one_page(cell_lens: &[usize], base: usize, hdr_len: usize) -> bool {
    let total: usize = cell_lens.iter().sum();
    total + cell_lens.len() * 2 <= cell_space(base, hdr_len, 0)
}

/// Assemble a complete database file.
///
/// Tables and indexes are laid out first, from page 2 up, so that
/// `sqlite_schema` can record their root pages; then page 1 is written last
/// with the schema itself. `sqlite_schema`'s root is fixed at page 1 by the
/// format, which is why it cannot be allocated like everything else.
pub fn build_database(tables: &[Table], indexes: &[Index]) -> Res<Vec<u8>> {
    let mut b = Builder::new();
    let mut schema_rows: Vec<(i64, Vec<Value>)> = Vec::new();
    let mut rowid = 0i64;

    for t in tables {
        let root = build_table_btree(&mut b, &t.rows, None)?;
        rowid += 1;
        schema_rows.push((rowid, vec![
            Value::Text("table".into()),
            Value::Text(t.name.clone()),
            Value::Text(t.name.clone()),
            Value::Int(root as i64),
            Value::Text(t.sql.clone()),
        ]));
        // An automatic index must follow the table that declares it, because
        // SQLite matches them up in schema order.
        for idx in indexes.iter().filter(|i| i.tbl_name == t.name) {
            let root = build_index_btree(&mut b, &idx.keys)?;
            rowid += 1;
            schema_rows.push((rowid, vec![
                Value::Text("index".into()),
                Value::Text(idx.name.clone()),
                Value::Text(idx.tbl_name.clone()),
                Value::Int(root as i64),
                match &idx.sql { Some(s) => Value::Text(s.clone()), None => Value::Null },
            ]));
        }
    }

    let schema_root = build_table_btree(&mut b, &schema_rows, Some(1))?;
    debug_assert_eq!(schema_root, 1);

    // Header. Everything not named here is zero, which is the correct value
    // for a freshly built file: no freelist, no auto-vacuum, no cache hint.
    let n_pages = b.pages.len() as u32;
    let page1 = b.page_mut(1);
    page1[0..16].copy_from_slice(b"SQLite format 3\0");
    page1[16..18].copy_from_slice(&(PAGE_SIZE as u16).to_be_bytes());
    page1[18] = 1; // write version: legacy rollback journal
    page1[19] = 1; // read version
    page1[20] = 0; // reserved space per page
    page1[21] = 64; // max embedded payload fraction, fixed by the format
    page1[22] = 32; // min embedded payload fraction
    page1[23] = 32; // leaf payload fraction
    page1[24..28].copy_from_slice(&1u32.to_be_bytes()); // change counter
    page1[28..32].copy_from_slice(&n_pages.to_be_bytes());
    page1[40..44].copy_from_slice(&1u32.to_be_bytes()); // schema cookie
    page1[44..48].copy_from_slice(&4u32.to_be_bytes()); // schema format 4
    page1[56..60].copy_from_slice(&1u32.to_be_bytes()); // text encoding: UTF-8
    page1[OFF_USER_VERSION..OFF_USER_VERSION + 4].copy_from_slice(&USER_VERSION.to_be_bytes());
    page1[OFF_APPLICATION_ID..OFF_APPLICATION_ID + 4].copy_from_slice(&APPLICATION_ID.to_be_bytes());
    // "Version valid for" must match the change counter or readers treat the
    // library-version field as stale.
    page1[92..96].copy_from_slice(&1u32.to_be_bytes());
    page1[96..100].copy_from_slice(&3_045_000u32.to_be_bytes());

    let mut out = Vec::with_capacity(b.pages.len() * PAGE_SIZE);
    for p in &b.pages { out.extend_from_slice(p); }
    Ok(out)
}

// =========================================================================
// GeoPackage
// =========================================================================

/// One linestring, with whatever non-geometry columns came with it.
#[derive(Clone, Debug)]
pub struct Feature {
    pub fid: i64,
    pub points: Vec<[f64; 3]>,
    /// True when the geometry was a polygon ring, so a save can put it back as
    /// one rather than as an open line that happens to end where it started.
    pub closed: bool,
    /// Non-geometry columns as text, keyed by column name. A column that was
    /// NULL is **absent** rather than empty, which is what lets a NULL survive
    /// the round trip.
    pub attrs: BTreeMap<String, String>,
}

/// One feature table.
#[derive(Debug)]
pub struct GpkgLayer {
    pub name: String,
    pub geom_column: String,
    pub srs_id: i32,
    /// Columns as declared, geometry and rowid alias excluded, in table order.
    /// Kept so a file from QGIS is written back with the columns it arrived
    /// with rather than with ours.
    pub columns: Vec<(String, String)>,
    pub features: Vec<Feature>,
}

impl GpkgLayer {
    /// A layer haiviz created itself, with the attribute columns from §7.2.
    pub fn new(name: impl Into<String>, srs_id: i32) -> GpkgLayer {
        GpkgLayer {
            name: name.into(),
            geom_column: "geom".to_string(),
            srs_id,
            columns: vec![
                ("name".to_string(), "TEXT".to_string()),
                ("code".to_string(), "TEXT".to_string()),
                ("n_vertices".to_string(), "INTEGER".to_string()),
                ("length_3d".to_string(), "REAL".to_string()),
                ("created".to_string(), "DATETIME".to_string()),
            ],
            features: Vec::new(),
        }
    }

    fn bbox(&self) -> (f64, f64, f64, f64) {
        let mut b = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for f in &self.features {
            for p in &f.points {
                b.0 = b.0.min(p[0]); b.1 = b.1.min(p[1]);
                b.2 = b.2.max(p[0]); b.3 = b.3.max(p[1]);
            }
        }
        if b.0 > b.2 { (0.0, 0.0, 0.0, 0.0) } else { b }
    }
}

// ---- WKB and the GeoPackage binary header ----

const WKB_LINESTRING: u32 = 2;
const WKB_POLYGON: u32 = 3;
const WKB_MULTILINESTRING: u32 = 5;

struct Cursor<'a> { b: &'a [u8], at: usize, le: bool }

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let s = self.b.get(self.at..self.at + n).ok_or("geometry blob truncated")?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Res<u32> {
        let s: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.le { u32::from_le_bytes(s) } else { u32::from_be_bytes(s) })
    }
    fn f64(&mut self) -> Res<f64> {
        let s: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.le { f64::from_le_bytes(s) } else { f64::from_be_bytes(s) })
    }
}

/// Decode a GeoPackage geometry blob into one or more 3D polylines.
///
/// `LineString`, `LineStringZ`, `MultiLineString(Z)` and `Polygon(Z)` rings all
/// reduce to lists of vertices; a 2D geometry gets `z = 0`, which is honest —
/// there was no elevation in the file. Anything else is skipped rather than
/// guessed at.
pub fn decode_geometry(blob: &[u8]) -> Res<Vec<(Vec<[f64; 3]>, bool)>> {
    if blob.len() < 8 || &blob[0..2] != b"GP" {
        return Err("not a GeoPackage geometry blob".into());
    }
    let flags = blob[3];
    let header_le = flags & 0x01 != 0;
    let envelope = match (flags >> 1) & 0x07 {
        0 => 0,
        1 => 32,  // xy
        2 => 48,  // xyz
        3 => 48,  // xym
        4 => 64,  // xyzm
        other => return Err(format!("unknown envelope indicator {other}")),
    };
    // An empty geometry carries no WKB at all.
    if flags & 0x10 != 0 { return Ok(Vec::new()); }
    let _srs = if header_le {
        i32::from_le_bytes(blob[4..8].try_into().unwrap())
    } else {
        i32::from_be_bytes(blob[4..8].try_into().unwrap())
    };

    let wkb_at = 8 + envelope;
    let wkb = blob.get(wkb_at..).ok_or("geometry blob has no WKB body")?;
    let mut out = Vec::new();
    let mut c = Cursor { b: wkb, at: 0, le: true };
    read_wkb(&mut c, &mut out, 0)?;
    Ok(out)
}

fn read_wkb(c: &mut Cursor, out: &mut Vec<(Vec<[f64; 3]>, bool)>, depth: u32) -> Res<()> {
    if depth > 8 { return Err("geometry nested too deeply".into()); }
    c.le = c.take(1)?[0] == 1;
    let raw = c.u32()?;
    // ISO WKB puts dimensionality in the thousands digit: 1000 = Z, 2000 = M,
    // 3000 = ZM. EWKB flags the high bits instead; both spellings appear in
    // files in the wild.
    let has_z = (1000..2000).contains(&raw) || (3000..4000).contains(&raw)
        || raw & 0x8000_0000 != 0;
    let has_m = (2000..3000).contains(&raw) || (3000..4000).contains(&raw)
        || raw & 0x4000_0000 != 0;
    let kind = (raw & 0x0fff_ffff) % 1000;

    let read_points = |c: &mut Cursor| -> Res<Vec<[f64; 3]>> {
        let n = c.u32()? as usize;
        if n > 50_000_000 { return Err("implausible vertex count".into()); }
        let mut pts = Vec::with_capacity(n.min(4096));
        for _ in 0..n {
            let x = c.f64()?;
            let y = c.f64()?;
            let z = if has_z { c.f64()? } else { 0.0 };
            if has_m { c.f64()?; }
            pts.push([x, y, z]);
        }
        Ok(pts)
    };

    match kind {
        WKB_LINESTRING => {
            let mut pts = read_points(c)?;
            // A linestring that ends where it began is a ring. Nothing in the
            // format distinguishes that from an open line that happens to
            // return to its start, so there is no information to lose by
            // reading it as closed — and it is what makes a ring survive a
            // round trip through a layer whose geometry type is LINESTRING.
            let ring = pts.len() >= 4 && pts[0] == pts[pts.len() - 1];
            if ring { pts.pop(); }
            if pts.len() >= 2 { out.push((pts, ring)); }
        }
        WKB_POLYGON => {
            let rings = c.u32()? as usize;
            for _ in 0..rings {
                let mut pts = read_points(c)?;
                // A WKB ring repeats its first vertex; `closed` says the same
                // thing without storing it twice.
                if pts.len() > 2 && pts[0] == pts[pts.len() - 1] { pts.pop(); }
                if pts.len() >= 3 { out.push((pts, true)); }
            }
        }
        WKB_MULTILINESTRING | 6 | 7 => {
            let n = c.u32()? as usize;
            for _ in 0..n { read_wkb(c, out, depth + 1)?; }
        }
        // Points and everything else: not a polyline, and not something to
        // invent one from.
        _ => {}
    }
    Ok(())
}

/// Encode a 3D linestring as a GeoPackage blob: the binary header with an xyz
/// envelope, then little-endian ISO WKB `LineStringZ` (type 1002).
///
/// Always Z, never 2D. A flat export would throw away the elevation that every
/// bit of the snapping exists to get right.
pub fn encode_linestring_z(srs_id: i32, points: &[[f64; 3]], closed: bool) -> Vec<u8> {
    let mut ring: Vec<[f64; 3]>;
    let pts = if closed && points.len() > 2 {
        ring = points.to_vec();
        ring.push(points[0]);
        &ring[..]
    } else {
        points
    };

    let mut env = [f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN];
    for p in pts {
        env[0] = env[0].min(p[0]); env[1] = env[1].max(p[0]);
        env[2] = env[2].min(p[1]); env[3] = env[3].max(p[1]);
        env[4] = env[4].min(p[2]); env[5] = env[5].max(p[2]);
    }
    if pts.is_empty() { env = [0.0; 6]; }

    let mut b = Vec::with_capacity(8 + 48 + 9 + pts.len() * 24);
    b.extend_from_slice(b"GP");
    b.push(0);
    // bit 0: little-endian. bits 1-3: envelope indicator 2 (xyz).
    b.push(0x01 | (2 << 1));
    b.extend_from_slice(&srs_id.to_le_bytes());
    for v in env { b.extend_from_slice(&v.to_le_bytes()); }

    b.push(1);
    b.extend_from_slice(&1002u32.to_le_bytes());
    b.extend_from_slice(&(pts.len() as u32).to_le_bytes());
    for p in pts {
        b.extend_from_slice(&p[0].to_le_bytes());
        b.extend_from_slice(&p[1].to_le_bytes());
        b.extend_from_slice(&p[2].to_le_bytes());
    }
    b
}

// ---- Reading a whole file ----

/// The column a table declares as `INTEGER PRIMARY KEY`, if any.
///
/// Such a column is an alias for the rowid: the record stores NULL in its slot
/// and the real value is the cell's key. Reading it back out of the record
/// without this would give NULL for every `fid` in the file.
fn rowid_alias_column(sql: &str) -> Option<usize> {
    let names = column_names(sql);
    let open = sql.find('(')?;
    let body = &sql[open + 1..];
    let mut depth = 0i32;
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in body.chars() {
        match ch {
            '(' => { depth += 1; cur.push(ch); }
            ')' if depth == 0 => break,
            ')' => { depth -= 1; cur.push(ch); }
            ',' if depth == 0 => { parts.push(std::mem::take(&mut cur)); }
            _ => cur.push(ch),
        }
    }
    parts.push(cur);
    let mut col = 0usize;
    for p in parts {
        let low = p.trim().to_ascii_lowercase();
        let first = low.split_whitespace().next().unwrap_or("");
        if ["constraint", "primary", "unique", "check", "foreign", "exclude", "key"]
            .contains(&first.trim_matches('"')) {
            continue;
        }
        if col < names.len() && low.contains("integer") && low.contains("primary key") {
            return Some(col);
        }
        col += 1;
    }
    None
}

/// Read every feature table from a GeoPackage.
///
/// Layers are found by joining `gpkg_contents` against `gpkg_geometry_columns`,
/// so the rtree tables, triggers, tile-matrix tables and extension registries a
/// real file carries are skipped by construction rather than by a deny-list.
pub fn read(path: &str) -> Res<Vec<GpkgLayer>> {
    let db = Db::open(path)?;
    let schema = db.schema()?;
    let find = |name: &str| schema.iter().find(|r| r.kind == "table" && r.name == name);

    let contents = find("gpkg_contents")
        .ok_or("not a GeoPackage: no gpkg_contents table")?;
    let geom_cols = find("gpkg_geometry_columns")
        .ok_or("not a GeoPackage: no gpkg_geometry_columns table")?;

    let c_names = column_names(&contents.sql);
    let col = |names: &[String], want: &str| names.iter().position(|n| n.eq_ignore_ascii_case(want));

    let c_table = col(&c_names, "table_name").ok_or("gpkg_contents has no table_name")?;
    let c_type = col(&c_names, "data_type").ok_or("gpkg_contents has no data_type")?;

    let g_names = column_names(&geom_cols.sql);
    let g_table = col(&g_names, "table_name").ok_or("gpkg_geometry_columns has no table_name")?;
    let g_col = col(&g_names, "column_name").ok_or("gpkg_geometry_columns has no column_name")?;
    let g_srs = col(&g_names, "srs_id");

    let mut geom_of: BTreeMap<String, (String, i32)> = BTreeMap::new();
    for (_, row) in db.rows(geom_cols.rootpage)? {
        let (Some(t), Some(c)) = (
            row.get(g_table).and_then(|v| v.as_text()),
            row.get(g_col).and_then(|v| v.as_text()),
        ) else { continue };
        let srs = g_srs.and_then(|i| row.get(i)).and_then(|v| v.as_i64()).unwrap_or(0) as i32;
        geom_of.insert(t.to_string(), (c.to_string(), srs));
    }

    let mut layers = Vec::new();
    for (_, row) in db.rows(contents.rootpage)? {
        let Some(table) = row.get(c_table).and_then(|v| v.as_text()) else { continue };
        if row.get(c_type).and_then(|v| v.as_text()) != Some("features") { continue }
        let Some((geom_column, srs_id)) = geom_of.get(table).cloned() else { continue };
        let Some(entry) = find(table) else { continue };

        let names = column_names(&entry.sql);
        let types = column_types(&entry.sql);
        let alias = rowid_alias_column(&entry.sql);
        let Some(gi) = names.iter().position(|n| n.eq_ignore_ascii_case(&geom_column)) else { continue };

        let columns: Vec<(String, String)> = names.iter().enumerate()
            .filter(|(i, _)| *i != gi && Some(*i) != alias)
            .map(|(i, n)| (n.clone(), types.get(i).cloned().unwrap_or_default()))
            .collect();

        let mut features = Vec::new();
        for (rowid, vals) in db.rows(entry.rootpage)? {
            let mut attrs = BTreeMap::new();
            for (i, name) in names.iter().enumerate() {
                if i == gi || Some(i) == alias { continue; }
                match vals.get(i) {
                    None | Some(Value::Null) => {}
                    Some(v) => { attrs.insert(name.clone(), v.to_display()); }
                }
            }
            let Some(blob) = vals.get(gi).and_then(|v| v.as_blob()) else { continue };
            // One row can hold a multi-linestring or a multi-ring polygon.
            // Each part becomes its own line; they share the row's attributes.
            for (points, closed) in decode_geometry(blob)? {
                features.push(Feature { fid: rowid, points, closed, attrs: attrs.clone() });
            }
        }

        layers.push(GpkgLayer {
            name: table.to_string(),
            geom_column,
            srs_id,
            columns,
            features,
        });
    }
    Ok(layers)
}

// ---- Writing a whole file ----

fn quote_ident(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Sort key for an index entry. Every key this writer produces is TEXT, whose
/// default collation is BINARY, so a byte comparison is the whole ordering.
fn index_key_bytes(vals: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in vals {
        match v {
            Value::Text(t) => out.extend_from_slice(t.as_bytes()),
            Value::Blob(b) => out.extend_from_slice(b),
            Value::Int(i) => out.extend_from_slice(&i.to_be_bytes()),
            Value::Real(r) => out.extend_from_slice(&r.to_be_bytes()),
            Value::Null => {}
        }
        out.push(0);
    }
    out
}

const SQL_SRS: &str = "CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL,srs_id INTEGER NOT NULL PRIMARY KEY,organization TEXT NOT NULL,organization_coordsys_id INTEGER NOT NULL,definition TEXT NOT NULL,description TEXT)";
const SQL_CONTENTS: &str = "CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY,data_type TEXT NOT NULL,identifier TEXT UNIQUE,description TEXT DEFAULT '',last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),min_x DOUBLE, min_y DOUBLE,max_x DOUBLE, max_y DOUBLE,srs_id INTEGER,CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id))";
const SQL_GEOM_COLS: &str = "CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL,column_name TEXT NOT NULL,geometry_type_name TEXT NOT NULL,srs_id INTEGER NOT NULL,z TINYINT NOT NULL,m TINYINT NOT NULL,CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),CONSTRAINT uk_gc_table_name UNIQUE (table_name),CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id))";

/// Serialise layers into a complete GeoPackage.
pub fn encode(layers: &[GpkgLayer]) -> Res<Vec<u8>> {
    let now = iso8601_now();

    // gpkg_spatial_ref_sys. The three rows below -1/0/4326 are required by the
    // spec whether or not anything references them. `srs_id` is the rowid
    // alias, so its record slot is NULL and the value rides in the key.
    let mut srs_rows: Vec<(i64, Vec<Value>)> = vec![
        (-1, vec![Value::Text("Undefined cartesian SRS".into()), Value::Null,
                  Value::Text("NONE".into()), Value::Int(-1),
                  Value::Text("undefined".into()),
                  Value::Text("undefined cartesian coordinate reference system".into())]),
        (0, vec![Value::Text("Undefined geographic SRS".into()), Value::Null,
                 Value::Text("NONE".into()), Value::Int(0),
                 Value::Text("undefined".into()),
                 Value::Text("undefined geographic coordinate reference system".into())]),
        (4326, vec![Value::Text("WGS 84 geodetic".into()), Value::Null,
                    Value::Text("EPSG".into()), Value::Int(4326),
                    Value::Text(WGS84_WKT.into()),
                    Value::Text("longitude/latitude coordinates in decimal degrees on WGS 84".into())]),
    ];
    let mut extra: Vec<i32> = layers.iter().map(|l| l.srs_id)
        .filter(|s| ![-1, 0, 4326].contains(s))
        .collect();
    extra.sort_unstable();
    extra.dedup();
    for srs in extra {
        // A bare EPSG reference. GDAL and QGIS resolve the code from
        // `organization_coordsys_id`, so a full WKT string is not needed to
        // make the file usable — and inventing one we cannot verify would be
        // worse than saying where to look it up.
        srs_rows.push((srs as i64, vec![
            Value::Text(format!("EPSG:{srs}")),
            Value::Null,
            Value::Text("EPSG".into()),
            Value::Int(srs as i64),
            Value::Text(format!("EPSG:{srs}")),
            Value::Text(String::new()),
        ]));
    }
    srs_rows.sort_by_key(|(id, _)| *id);

    let mut contents_rows = Vec::new();
    let mut geom_rows = Vec::new();
    let mut feature_tables = Vec::new();

    for (i, layer) in layers.iter().enumerate() {
        let (min_x, min_y, max_x, max_y) = layer.bbox();
        contents_rows.push((i as i64 + 1, vec![
            Value::Text(layer.name.clone()),
            Value::Text("features".into()),
            Value::Text(layer.name.clone()),
            Value::Text(String::new()),
            Value::Text(now.clone()),
            Value::Real(min_x), Value::Real(min_y),
            Value::Real(max_x), Value::Real(max_y),
            Value::Int(layer.srs_id as i64),
        ]));
        geom_rows.push((i as i64 + 1, vec![
            Value::Text(layer.name.clone()),
            Value::Text(layer.geom_column.clone()),
            // The type name stays LINESTRING; `z = 1` is what says 3D. That is
            // how GDAL writes it, and what QGIS reads back as a 3D layer.
            Value::Text("LINESTRING".into()),
            Value::Int(layer.srs_id as i64),
            Value::Int(1),
            Value::Int(0),
        ]));

        let mut cols = format!(
            "CREATE TABLE {} (fid INTEGER PRIMARY KEY, {} LINESTRING",
            quote_ident(&layer.name), quote_ident(&layer.geom_column),
        );
        for (name, ty) in &layer.columns {
            let ty = if ty.is_empty() { "TEXT" } else { ty.as_str() };
            cols.push_str(&format!(", {} {}", quote_ident(name), ty));
        }
        cols.push(')');

        let mut rows = Vec::with_capacity(layer.features.len());
        for (n, f) in layer.features.iter().enumerate() {
            // fid is the rowid alias, so its slot is NULL. Renumbered from 1
            // rather than carried over: a feature read from one file and
            // written into another has no claim on its old id, and a duplicate
            // would silently drop a row.
            let mut vals = vec![
                Value::Null,
                Value::Blob(encode_linestring_z(layer.srs_id, &f.points, f.closed)),
            ];
            for (name, ty) in &layer.columns {
                vals.push(match f.attrs.get(name) {
                    None => Value::Null,
                    Some(s) => text_as(ty, s),
                });
            }
            rows.push((n as i64 + 1, vals));
        }
        feature_tables.push(Table { name: layer.name.clone(), sql: cols, rows });
    }

    let mut tables = vec![
        Table { name: "gpkg_spatial_ref_sys".into(), sql: SQL_SRS.into(), rows: srs_rows },
        Table { name: "gpkg_contents".into(), sql: SQL_CONTENTS.into(), rows: contents_rows.clone() },
        Table { name: "gpkg_geometry_columns".into(), sql: SQL_GEOM_COLS.into(), rows: geom_rows.clone() },
    ];
    tables.extend(feature_tables);

    // The automatic indexes SQLite would create for the TEXT primary keys and
    // unique constraints above. Their names and their order are what SQLite
    // matches against the constraints in the DDL, so both are fixed.
    let mut indexes = vec![
        auto_index("sqlite_autoindex_gpkg_contents_1", "gpkg_contents",
                   &contents_rows, &[0]),
        auto_index("sqlite_autoindex_gpkg_contents_2", "gpkg_contents",
                   &contents_rows, &[2]),
        auto_index("sqlite_autoindex_gpkg_geometry_columns_1", "gpkg_geometry_columns",
                   &geom_rows, &[0, 1]),
        auto_index("sqlite_autoindex_gpkg_geometry_columns_2", "gpkg_geometry_columns",
                   &geom_rows, &[0]),
    ];
    for idx in &mut indexes {
        idx.keys.sort_by_cached_key(|k| index_key_bytes(k));
    }

    build_database(&tables, &indexes)
}

fn auto_index(name: &str, table: &str, rows: &[(i64, Vec<Value>)], cols: &[usize]) -> Index {
    let keys = rows.iter().map(|(rowid, vals)| {
        let mut k: Vec<Value> = cols.iter().map(|&c| vals[c].clone()).collect();
        // An index entry ends with the rowid it points at.
        k.push(Value::Int(*rowid));
        k
    }).collect();
    Index { name: name.into(), tbl_name: table.into(), sql: None, keys }
}

/// Put an attribute string back in the type its column declares. An empty
/// string in a numeric column is a NULL, not a zero.
fn text_as(declared: &str, s: &str) -> Value {
    let d = declared.to_ascii_uppercase();
    if d.contains("INT") {
        if s.trim().is_empty() { return Value::Null; }
        s.trim().parse::<i64>().map(Value::Int).unwrap_or_else(|_| Value::Text(s.to_string()))
    } else if d.contains("REAL") || d.contains("DOUB") || d.contains("FLOA") {
        if s.trim().is_empty() { return Value::Null; }
        s.trim().parse::<f64>().map(Value::Real).unwrap_or_else(|_| Value::Text(s.to_string()))
    } else {
        Value::Text(s.to_string())
    }
}

const WGS84_WKT: &str = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]";

/// `gpkg_contents.last_change` wants an ISO 8601 UTC instant with
/// milliseconds. Computed here rather than pulled in as a dependency; the
/// civil-from-days conversion is Howard Hinnant's, which is exact for every
/// date this will ever see.
pub fn iso8601_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs() as i64;
    let ms = d.subsec_millis();
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { y + 1 } else { y };

    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        tod / 3600, (tod % 3600) / 60, tod % 60,
    )
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn tmp(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("haiviz-gpkg-{}-{}", std::process::id(), name));
        p.to_string_lossy().into_owned()
    }

    /// Run an external tool, or return `None` when it is not installed.
    ///
    /// The differential tests are the point of this module — "it parses on my
    /// machine" means very little for a format other people's tools consume —
    /// but they must not turn into a build-environment requirement.
    fn tool(cmd: &str, args: &[&str]) -> Option<(bool, String)> {
        match Command::new(cmd).args(args).output() {
            Ok(o) => Some((
                o.status.success(),
                format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)),
            )),
            Err(_) => None,
        }
    }

    // ---------------- primitives ----------------

    #[test]
    fn varints_round_trip_across_every_width() {
        let cases = [
            0u64, 1, 0x7f, 0x80, 0x3fff, 0x4000, 0x1f_ffff, 0x20_0000,
            0x0fff_ffff, 0x1000_0000, 0x0007_ffff_ffff, 0x0008_0000_0000,
            0x00ff_ffff_ffff_ffff, 0x0100_0000_0000_0000, u64::MAX,
        ];
        for v in cases {
            let mut b = Vec::new();
            put_varint(&mut b, v);
            assert!(b.len() <= 9, "{v:#x} encoded to {} bytes", b.len());
            let (got, used) = get_varint(&b, 0).expect("decodes");
            assert_eq!(got, v, "{v:#x} round trip");
            assert_eq!(used, b.len(), "{v:#x} length");
        }
        assert!(get_varint(&[0x81], 0).is_err(), "a truncated varint must not panic");
    }

    #[test]
    fn records_round_trip_every_storage_class() {
        let vals = vec![
            Value::Null,
            Value::Int(0), Value::Int(1), Value::Int(-1),
            Value::Int(127), Value::Int(-128), Value::Int(32767), Value::Int(-32768),
            Value::Int(8_388_607), Value::Int(-8_388_608),
            Value::Int(2_147_483_647), Value::Int(-2_147_483_648),
            Value::Int(140_737_488_355_327), Value::Int(-140_737_488_355_328),
            Value::Int(i64::MAX), Value::Int(i64::MIN),
            Value::Real(0.0), Value::Real(-1.5), Value::Real(208_802.001),
            Value::Text(String::new()), Value::Text("kerb".into()),
            Value::Text("Ĳsselmeer — ø".into()),
            Value::Blob(Vec::new()), Value::Blob(vec![0, 1, 255, 128]),
        ];
        let enc = encode_record(&vals);
        assert_eq!(decode_record(&enc).expect("decodes"), vals);
    }

    #[test]
    fn a_record_header_longer_than_127_bytes_still_states_its_own_length() {
        // The header length is a varint that counts itself, so at the 127-byte
        // boundary it widens and the count has to widen with it.
        for n in [100usize, 126, 127, 128, 200, 5000] {
            let vals: Vec<Value> = (0..n).map(|i| Value::Int(i as i64 + 1000)).collect();
            let enc = encode_record(&vals);
            assert_eq!(decode_record(&enc).expect("decodes"), vals, "{n} columns");
        }
    }

    #[test]
    fn the_local_payload_split_matches_sqlites_rule() {
        let u = 4096;
        // Anything up to max_local stays on the page.
        assert_eq!(local_payload_size(10, u, true), 10);
        assert_eq!(local_payload_size(u - 35, u, true), u - 35);
        // Past it, the local part never exceeds max_local and never drops
        // below min_local.
        let min_local = ((u - 12) * 32 / 255) - 23;
        for total in [u - 34, u, 5000, 100_000, 1_000_000] {
            let l = local_payload_size(total, u, true);
            assert!(l >= min_local && l <= u - 35, "total {total} gave local {l}");
            assert!(l < total);
        }
    }

    #[test]
    fn column_names_survive_the_ddl_real_files_carry() {
        let sql = "CREATE TABLE \"roads\" ( \"fid\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, \
                   \"geom\" LINESTRING, \"name\" TEXT, \"code\" TEXT)";
        assert_eq!(column_names(sql), vec!["fid", "geom", "name", "code"]);
        assert_eq!(rowid_alias_column(sql), Some(0));

        // Table constraints and parenthesised defaults must not be mistaken
        // for columns.
        let names = column_names(SQL_CONTENTS);
        assert_eq!(names, vec!["table_name", "data_type", "identifier", "description",
                               "last_change", "min_x", "min_y", "max_x", "max_y", "srs_id"]);
        let names = column_names(SQL_GEOM_COLS);
        assert_eq!(names, vec!["table_name", "column_name", "geometry_type_name",
                               "srs_id", "z", "m"]);
        assert_eq!(rowid_alias_column(SQL_SRS), Some(1));
    }

    // ---------------- geometry ----------------

    #[test]
    fn a_linestring_z_survives_encode_and_decode() {
        let pts = vec![
            [208_802.001, 373_320.002, 12.003],
            [208_812.5, 373_320.0, 12.25],
            [208_820.0, 373_325.0, 13.0],
        ];
        let blob = encode_linestring_z(28992, &pts, false);
        assert_eq!(&blob[0..2], b"GP");
        assert_eq!(blob[3], 0x05, "little-endian with an xyz envelope");
        assert_eq!(i32::from_le_bytes(blob[4..8].try_into().unwrap()), 28992);
        assert_eq!(u32::from_le_bytes(blob[57..61].try_into().unwrap()), 1002,
                   "ISO WKB LineStringZ");

        let got = decode_geometry(&blob).expect("decodes");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, pts, "coordinates must be exact, not merely close");
        assert!(!got[0].1);
    }

    #[test]
    fn a_ring_is_written_closed_and_read_back_without_the_repeat() {
        let pts = vec![[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [10.0, 10.0, 1.0]];
        let blob = encode_linestring_z(28992, &pts, true);
        let got = decode_geometry(&blob).expect("decodes");
        // Written as four vertices — WKB repeats the first — but the ring flag
        // is what a reader should get back, not a duplicate vertex.
        assert_eq!(u32::from_le_bytes(blob[61..65].try_into().unwrap()), 4);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, pts);
        assert!(got[0].1, "the ring flag must survive, not the repeated vertex");

        // An open line is not turned into one by accident.
        let open = vec![[0.0, 0.0, 1.0], [10.0, 0.0, 1.0], [10.0, 10.0, 1.0], [5.0, 5.0, 1.0]];
        let got = decode_geometry(&encode_linestring_z(28992, &open, false)).expect("decodes");
        assert_eq!(got[0].0, open);
        assert!(!got[0].1);
    }

    #[test]
    fn two_dimensional_and_multipart_geometries_are_read_too() {
        // Plain 2D LineString, no envelope: z comes back as 0 rather than
        // being invented.
        let mut b = vec![b'G', b'P', 0, 0x01];
        b.extend_from_slice(&28992i32.to_le_bytes());
        b.push(1);
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        for v in [1.0f64, 2.0, 3.0, 4.0] { b.extend_from_slice(&v.to_le_bytes()); }
        let got = decode_geometry(&b).expect("decodes");
        assert_eq!(got[0].0, vec![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]);

        // MultiLineStringZ of two parts becomes two lines.
        let mut b = vec![b'G', b'P', 0, 0x01];
        b.extend_from_slice(&28992i32.to_le_bytes());
        b.push(1);
        b.extend_from_slice(&1005u32.to_le_bytes());
        b.extend_from_slice(&2u32.to_le_bytes());
        for off in [0.0f64, 100.0] {
            b.push(1);
            b.extend_from_slice(&1002u32.to_le_bytes());
            b.extend_from_slice(&2u32.to_le_bytes());
            for v in [off, off + 1.0, 5.0, off + 2.0, off + 3.0, 6.0] {
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        let got = decode_geometry(&b).expect("decodes");
        assert_eq!(got.len(), 2);
        assert_eq!(got[1].0[0], [100.0, 101.0, 5.0]);

        // A big-endian WKB body, which the byte-order byte is there to allow.
        let mut b = vec![b'G', b'P', 0, 0x01];
        b.extend_from_slice(&28992i32.to_le_bytes());
        b.push(0);
        b.extend_from_slice(&1002u32.to_be_bytes());
        b.extend_from_slice(&2u32.to_be_bytes());
        for v in [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] { b.extend_from_slice(&v.to_be_bytes()); }
        let got = decode_geometry(&b).expect("decodes");
        assert_eq!(got[0].0, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }

    #[test]
    fn a_truncated_blob_is_an_error_not_a_panic() {
        let pts = vec![[0.0; 3], [1.0, 1.0, 1.0]];
        let full = encode_linestring_z(28992, &pts, false);
        for cut in [0, 1, 4, 8, 20, 60, full.len() - 1] {
            let _ = decode_geometry(&full[..cut]);
        }
        assert!(decode_geometry(b"not a blob").is_err());
    }

    // ---------------- whole files ----------------

    fn sample_layer(n_lines: usize, vertices: usize) -> GpkgLayer {
        let mut l = GpkgLayer::new("roads", 28992);
        for i in 0..n_lines {
            let pts: Vec<[f64; 3]> = (0..vertices).map(|v| [
                208_800.0 + i as f64 * 10.0 + v as f64 * 0.25,
                373_300.0 + v as f64 * 0.1,
                12.0 + (v % 7) as f64 * 0.01,
            ]).collect();
            let mut attrs = BTreeMap::new();
            attrs.insert("name".to_string(), format!("kerb {i}"));
            attrs.insert("code".to_string(), "KRB".to_string());
            attrs.insert("n_vertices".to_string(), vertices.to_string());
            l.features.push(Feature { fid: 0, points: pts, closed: false, attrs });
        }
        l
    }

    #[test]
    fn a_written_file_reads_back_through_our_own_reader() {
        let path = tmp("selfread.gpkg");
        fs::write(&path, encode(&[sample_layer(5, 4)]).expect("encodes")).unwrap();

        let layers = read(&path).expect("reads");
        assert_eq!(layers.len(), 1);
        let l = &layers[0];
        assert_eq!(l.name, "roads");
        assert_eq!(l.srs_id, 28992);
        assert_eq!(l.features.len(), 5);
        assert_eq!(l.features[2].points.len(), 4);
        assert_eq!(l.features[2].attrs.get("name").map(String::as_str), Some("kerb 2"));
        assert_eq!(l.features[0].fid, 1, "fid comes from the rowid, not a NULL column");
        // Coordinates must be bit-exact: the f64 master exists for this.
        assert_eq!(l.features[3].points[1][0], 208_830.25);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn overflow_pages_carry_a_line_too_long_for_one_page() {
        // 4096-byte pages hold ~170 vertices inline; 2000 forces a chain.
        let path = tmp("overflow.gpkg");
        fs::write(&path, encode(&[sample_layer(3, 2000)]).expect("encodes")).unwrap();
        let layers = read(&path).expect("reads");
        assert_eq!(layers[0].features.len(), 3);
        for f in &layers[0].features {
            assert_eq!(f.points.len(), 2000);
        }
        assert_eq!(layers[0].features[1].points[1999][2], 12.0 + (1999 % 7) as f64 * 0.01);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_table_spanning_many_pages_gets_interior_nodes() {
        // Enough small rows to need more than one leaf, so the b-tree grows a
        // level and the interior traversal is actually exercised.
        let path = tmp("manyrows.gpkg");
        fs::write(&path, encode(&[sample_layer(400, 3)]).expect("encodes")).unwrap();
        let layers = read(&path).expect("reads");
        assert_eq!(layers[0].features.len(), 400);
        assert_eq!(layers[0].features[399].attrs.get("name").map(String::as_str), Some("kerb 399"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn nulls_and_types_survive_a_round_trip() {
        let path = tmp("attrs.gpkg");
        let mut l = GpkgLayer::new("roads", 28992);
        let mut a = BTreeMap::new();
        a.insert("name".to_string(), "with name".to_string());
        a.insert("length_3d".to_string(), "12.5".to_string());
        l.features.push(Feature { fid: 0, points: vec![[0.0; 3], [1.0, 1.0, 1.0]], closed: false, attrs: a });
        // Second feature leaves every attribute unset.
        l.features.push(Feature { fid: 0, points: vec![[0.0; 3], [2.0, 2.0, 2.0]], closed: false, attrs: BTreeMap::new() });
        fs::write(&path, encode(&[l]).expect("encodes")).unwrap();

        let back = read(&path).expect("reads");
        let f0 = &back[0].features[0];
        assert_eq!(f0.attrs.get("length_3d").map(String::as_str), Some("12.5"));
        assert_eq!(f0.attrs.get("code"), None, "an unset column stays absent, not empty");
        assert!(back[0].features[1].attrs.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_write_ahead_log_is_refused_rather_than_read_stale() {
        let path = tmp("wal.gpkg");
        fs::write(&path, encode(&[sample_layer(1, 3)]).expect("encodes")).unwrap();
        fs::write(format!("{path}-wal"), b"not empty").unwrap();
        let err = read(&path).expect_err("must refuse");
        assert!(err.contains("write-ahead log"), "{err}");
        let _ = fs::remove_file(format!("{path}-wal"));
        // With the sidecar gone it reads normally again.
        assert!(read(&path).is_ok());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn garbage_is_rejected_without_panicking() {
        let path = tmp("garbage.gpkg");
        fs::write(&path, b"SQLite format 3\0but not really, nowhere near a page").unwrap();
        assert!(read(&path).is_err());
        fs::write(&path, b"plain text").unwrap();
        assert!(read(&path).is_err());
        // A valid SQLite file that is not a GeoPackage.
        let db = build_database(
            &[Table { name: "t".into(), sql: "CREATE TABLE t (a INTEGER)".into(),
                      rows: vec![(1, vec![Value::Int(7)])] }],
            &[],
        ).expect("builds");
        fs::write(&path, db).unwrap();
        let err = read(&path).expect_err("must refuse");
        assert!(err.contains("not a GeoPackage"), "{err}");
        let _ = fs::remove_file(&path);
    }

    // ---------------- differential ----------------

    #[test]
    fn sqlite3_agrees_that_the_file_is_sound() {
        let path = tmp("diff-sqlite.gpkg");
        fs::write(&path, encode(&[sample_layer(300, 500)]).expect("encodes")).unwrap();

        let Some((ok, out)) = tool("sqlite3", &[&path, "PRAGMA integrity_check;"]) else {
            eprintln!("sqlite3 not installed — skipping the differential check");
            let _ = fs::remove_file(&path);
            return;
        };
        assert!(ok && out.trim() == "ok", "integrity_check said: {out}");

        // Not just structurally sound — the rows have to be there, and the
        // indexes have to agree with the tables they index.
        let (_, out) = tool("sqlite3", &[&path, "PRAGMA foreign_key_check;"]).unwrap();
        assert!(out.trim().is_empty(), "foreign_key_check said: {out}");
        let (_, out) = tool("sqlite3", &[&path, "SELECT count(*) FROM roads;"]).unwrap();
        assert_eq!(out.trim(), "300");
        let (_, out) = tool("sqlite3", &[&path,
            "SELECT table_name, srs_id FROM gpkg_geometry_columns;"]).unwrap();
        assert_eq!(out.trim(), "roads|28992");
        // A query the planner will serve from sqlite_autoindex_gpkg_contents_1,
        // which is the only thing that proves the index b-tree is real.
        let (_, out) = tool("sqlite3", &[&path,
            "SELECT data_type FROM gpkg_contents WHERE table_name='roads';"]).unwrap();
        assert_eq!(out.trim(), "features");
        let (_, out) = tool("sqlite3", &[&path,
            "SELECT name FROM roads WHERE fid=42;"]).unwrap();
        assert_eq!(out.trim(), "kerb 41");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn sqlite3_agrees_when_the_indexes_outgrow_one_page() {
        // The single-page index path covers any realistic file, which is
        // exactly why the split path needs a test of its own: without one it
        // would ship unexecuted. ~300 layers pushes both gpkg_contents
        // indexes past a 4 KB leaf.
        let path = tmp("diff-manylayers.gpkg");
        let layers: Vec<GpkgLayer> = (0..300).map(|i| {
            let mut l = GpkgLayer::new(format!("layer_{i:04}"), 28992);
            l.features.push(Feature {
                fid: 0,
                points: vec![[i as f64, 0.0, 1.0], [i as f64, 1.0, 1.0]],
                closed: false,
                attrs: BTreeMap::new(),
            });
            l
        }).collect();
        fs::write(&path, encode(&layers).expect("encodes")).unwrap();

        // Our own reader first, so a failure says which half is wrong.
        let back = read(&path).expect("reads");
        assert_eq!(back.len(), 300);
        assert_eq!(back[299].name, "layer_0299");

        let Some((ok, out)) = tool("sqlite3", &[&path, "PRAGMA integrity_check;"]) else {
            eprintln!("sqlite3 not installed — skipping the differential check");
            let _ = fs::remove_file(&path);
            return;
        };
        assert!(ok && out.trim() == "ok", "integrity_check said: {out}");
        // A lookup that has to descend the split index, not scan the table.
        let (_, out) = tool("sqlite3", &[&path,
            "SELECT srs_id FROM gpkg_contents WHERE table_name='layer_0287';"]).unwrap();
        assert_eq!(out.trim(), "28992");
        let (_, out) = tool("sqlite3", &[&path, "SELECT count(*) FROM gpkg_contents;"]).unwrap();
        assert_eq!(out.trim(), "300");

        // Confirm the split actually happened, or this test is only repeating
        // the single-page one with more rows.
        let db = Db::open(&path).expect("opens");
        let idx = db.schema().expect("schema").into_iter()
            .find(|r| r.name == "sqlite_autoindex_gpkg_contents_1").expect("index exists");
        let page = db.page(idx.rootpage).expect("root page");
        assert_eq!(page[0], PAGE_INTERIOR_INDEX,
                   "the index root should be an interior page at this size");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn ogr_reads_the_layer_as_a_3d_line_string() {
        let path = tmp("diff-ogr.gpkg");
        fs::write(&path, encode(&[sample_layer(12, 6)]).expect("encodes")).unwrap();

        let Some((ok, out)) = tool("ogrinfo", &["-al", "-so", &path]) else {
            eprintln!("ogrinfo not installed — skipping the differential check");
            let _ = fs::remove_file(&path);
            return;
        };
        assert!(ok, "ogrinfo failed: {out}");
        assert!(out.contains("Layer name: roads"), "{out}");
        assert!(out.contains("3D Line String"), "geometry must read as 3D:\n{out}");
        assert!(out.contains("Feature Count: 12"), "{out}");
        assert!(out.contains("Amersfoort") || out.contains("28992"),
                "the SRS must resolve from the EPSG code:\n{out}");

        // And the coordinates themselves, not just the metadata.
        let Some((ok, out)) = tool("ogrinfo", &[&path, "roads", "-where", "fid=1"]) else { return };
        assert!(ok, "{out}");
        assert!(out.contains("208800.0"), "first vertex missing from:\n{out}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_file_written_by_gdal_reads_back_correctly() {
        // The other direction, and the one the risk register cares about: a
        // real GDAL file carries rtree virtual tables, a dozen triggers,
        // gpkg_ogr_contents and tile-matrix tables that this reader has to
        // step over rather than choke on.
        let dir = std::env::temp_dir();
        let csv = dir.join(format!("haiviz-gpkg-{}-src.csv", std::process::id()));
        let out = dir.join(format!("haiviz-gpkg-{}-gdal.gpkg", std::process::id()));
        let _ = fs::remove_file(&out);
        fs::write(&csv,
            "wkt,name,code\n\
             \"LINESTRING Z (208800 373300 12, 208810 373300 12.5, 208820 373305 13)\",kerb,KRB\n\
             \"LINESTRING Z (208830 373310 14, 208840 373320 14.5)\",ditch,DTC\n").unwrap();

        let args = [
            "-f", "GPKG", out.to_str().unwrap(), csv.to_str().unwrap(),
            "-oo", "GEOM_POSSIBLE_NAMES=wkt", "-oo", "KEEP_GEOM_COLUMNS=NO",
            "-a_srs", "EPSG:28992", "-nln", "roads", "-nlt", "LINESTRINGZ",
        ];
        let Some((ok, msg)) = tool("ogr2ogr", &args) else {
            eprintln!("ogr2ogr not installed — skipping the differential check");
            let _ = fs::remove_file(&csv);
            return;
        };
        assert!(ok, "ogr2ogr failed: {msg}");

        let layers = read(out.to_str().unwrap()).expect("reads a GDAL file");
        assert_eq!(layers.len(), 1, "the rtree and tile-matrix tables must not appear as layers");
        let l = &layers[0];
        assert_eq!(l.name, "roads");
        assert_eq!(l.srs_id, 28992);
        assert_eq!(l.features.len(), 2);
        assert_eq!(l.features[0].points,
                   vec![[208_800.0, 373_300.0, 12.0], [208_810.0, 373_300.0, 12.5],
                        [208_820.0, 373_305.0, 13.0]]);
        assert_eq!(l.features[0].attrs.get("name").map(String::as_str), Some("kerb"));
        assert_eq!(l.features[1].attrs.get("code").map(String::as_str), Some("DTC"));
        assert!(l.columns.iter().any(|(n, _)| n == "code"), "columns: {:?}", l.columns);

        let _ = fs::remove_file(&csv);
        let _ = fs::remove_file(&out);
    }

    #[test]
    fn a_gdal_file_round_trips_back_through_gdal() {
        // Read GDAL's file, write it out again with our own writer, and make
        // GDAL read that. This is the whole interoperability claim in one test.
        let dir = std::env::temp_dir();
        let csv = dir.join(format!("haiviz-gpkg-{}-rt.csv", std::process::id()));
        let a = dir.join(format!("haiviz-gpkg-{}-rt-a.gpkg", std::process::id()));
        let b = dir.join(format!("haiviz-gpkg-{}-rt-b.gpkg", std::process::id()));
        for f in [&a, &b] { let _ = fs::remove_file(f); }
        fs::write(&csv,
            "wkt,name,code\n\
             \"LINESTRING Z (208800.001 373300.002 12.003, 208810 373300 12.5)\",kerb,KRB\n").unwrap();
        let args = [
            "-f", "GPKG", a.to_str().unwrap(), csv.to_str().unwrap(),
            "-oo", "GEOM_POSSIBLE_NAMES=wkt", "-oo", "KEEP_GEOM_COLUMNS=NO",
            "-a_srs", "EPSG:28992", "-nln", "roads", "-nlt", "LINESTRINGZ",
        ];
        let Some((ok, msg)) = tool("ogr2ogr", &args) else {
            eprintln!("ogr2ogr not installed — skipping the differential check");
            let _ = fs::remove_file(&csv);
            return;
        };
        assert!(ok, "{msg}");

        let layers = read(a.to_str().unwrap()).expect("reads");
        fs::write(&b, encode(&layers).expect("encodes")).unwrap();

        let (ok, out) = tool("ogrinfo", &["-al", b.to_str().unwrap()]).unwrap();
        assert!(ok, "ogrinfo could not read our rewrite: {out}");
        assert!(out.contains("3D Line String"), "{out}");
        assert!(out.contains("kerb"), "{out}");
        // Sub-millimetre coordinates must come through untouched — this is
        // what the f64 master exists for.
        assert!(out.contains("208800.001"), "coordinates were quantised:\n{out}");

        let again = read(b.to_str().unwrap()).expect("reads our own rewrite");
        assert_eq!(again[0].features[0].points, layers[0].features[0].points);

        for f in [&csv, &a, &b] { let _ = fs::remove_file(f); }
    }
}
