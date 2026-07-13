// Four-way comparison on hexo-strix's hot loops:
//   A. (i32,i32) keys + SipHash        — current engine
//   B. (i32,i32) keys + FxHashMap      — the one-line change
//   C. u64 packed keys + FxHashMap     — full scheme, 32-bit fields
//   D. u32 packed keys + FxHashMap     — compact scheme, 16-bit fields
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::Instant;

type Coord = (i32, i32);

// ---------- packing: u64 (32-bit fields) ----------
#[inline] fn key64(q: i32, r: i32) -> u64 {
    ((((q as u32) ^ 0x8000_0000) as u64) << 32) | (((r as u32) ^ 0x8000_0000) as u64)
}
#[inline] fn unkey64(k: u64) -> (i32, i32) {
    ((((k >> 32) as u32) ^ 0x8000_0000) as i32, ((k as u32) ^ 0x8000_0000) as i32)
}
const fn d64(dq: i32, dr: i32) -> u64 { (((dq as i64) << 32) + (dr as i64)) as u64 }

// ---------- packing: u32 (16-bit fields, q,r ∈ [-32768, 32767]) ----------
#[inline] fn key32(q: i32, r: i32) -> u32 {
    debug_assert!((-32768..=32767).contains(&q) && (-32768..=32767).contains(&r));
    (((q as u16 as u32) ^ 0x8000) << 16) | ((r as u16 as u32) ^ 0x8000)
}
#[inline] fn unkey32(k: u32) -> (i32, i32) {
    ((((k >> 16) as u16) ^ 0x8000) as i16 as i32, ((k as u16) ^ 0x8000) as i16 as i32)
}
const fn d32(dq: i32, dr: i32) -> u32 { ((dq << 16).wrapping_add(dr)) as u32 }

// ---------- generic walk-based win check over any probe fn ----------
fn check_win_probe<K: Copy>(k: K, step: impl Fn(K, usize) -> K, occ: impl Fn(K) -> bool, win: u8) -> bool {
    // dirs indexed 0..6, axis a uses dirs (2a, 2a+1)
    let max = win - 1;
    for a in 0..3 {
        let mut n = 1u8;
        for dir in [2 * a, 2 * a + 1] {
            let mut cur = step(k, dir);
            while n <= max && occ(cur) { n += 1; cur = step(cur, dir); }
        }
        if n >= win { return true; }
    }
    false
}

// ---------- deterministic position ----------
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.0 = x; x }
    fn range(&mut self, n: i32) -> i32 { (self.next() % (2 * n as u64 + 1)) as i32 - n }
}
fn build_position(n: usize, seed: u64) -> Vec<(i32, i32, u8)> {
    let mut rng = Rng(seed);
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut p = 1u8;
    while out.len() < n {
        let (q, r) = (rng.range(12), rng.range(12));
        if seen.insert((q, r)) { out.push((q, r, p)); p = 3 - p; }
    }
    out
}

fn hex_offsets(radius: i32) -> Vec<Coord> {
    let mut o = Vec::new();
    for dq in -radius..=radius { for dr in -radius..=radius {
        if dq.abs().max(dr.abs()).max((dq + dr).abs()) <= radius { o.push((dq, dr)); }
    }}
    o
}

fn bench(iters: u32, mut f: impl FnMut() -> u64) -> f64 {
    for _ in 0..iters / 10 { black_box(f()); }
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..iters { acc = acc.wrapping_add(f()); }
    let el = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    black_box(acc);
    el
}
fn row(name: &str, us: f64, base: f64) {
    println!("  {name:<38} {us:>9.2} µs   {:>5.1}x", base / us);
}

fn main() {
    let pos = build_position(120, 0xC0FFEE);
    let sip: HashMap<Coord, u8> = pos.iter().map(|&(q, r, p)| ((q, r), p)).collect();
    let fxt: FxHashMap<Coord, u8> = pos.iter().map(|&(q, r, p)| ((q, r), p)).collect();
    let fx64: FxHashMap<u64, u8> = pos.iter().map(|&(q, r, p)| (key64(q, r), p)).collect();
    let fx32: FxHashMap<u32, u8> = pos.iter().map(|&(q, r, p)| (key32(q, r), p)).collect();

    const DIRS: [Coord; 6] = [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]; // axis pairs
    let dd64: [u64; 6] = [d64(1,0),d64(-1,0),d64(0,1),d64(0,-1),d64(1,-1),d64(-1,1)];
    let dd32: [u32; 6] = [d32(1,0),d32(-1,0),d32(0,1),d32(0,-1),d32(1,-1),d32(-1,1)];

    // sanity: all four agree on win verdicts + roundtrips
    for &(q, r, p) in &pos {
        let a = check_win_probe((q,r), |c: Coord, d| (c.0+DIRS[d].0, c.1+DIRS[d].1), |c| sip.get(&c) == Some(&p), 6);
        let b = check_win_probe((q,r), |c: Coord, d| (c.0+DIRS[d].0, c.1+DIRS[d].1), |c| fxt.get(&c) == Some(&p), 6);
        let c = check_win_probe(key64(q,r), |k: u64, d| k.wrapping_add(dd64[d]), |k| fx64.get(&k) == Some(&p), 6);
        let e = check_win_probe(key32(q,r), |k: u32, d| k.wrapping_add(dd32[d]), |k| fx32.get(&k) == Some(&p), 6);
        assert!(a == b && b == c && c == e);
        assert_eq!(unkey64(key64(q, r)), (q, r));
        assert_eq!(unkey32(key32(q, r)), (q, r));
    }
    // u32 borrow / roundtrip torture at field boundaries
    let mut steps = 0u64;
    for q in -300..300 { for r in -300..300 {
        for (i, &(dq, dr)) in DIRS.iter().enumerate() {
            assert_eq!(unkey32(key32(q, r).wrapping_add(dd32[i])), (q + dq, r + dr));
            steps += 1;
        }
    }}
    println!("sanity: verdicts identical across all four; {steps} u32 neighbor adds verified\n");

    println!("== check_win sweep, 120 stones ==");
    let base = bench(3000, || { let m = black_box(&sip); pos.iter().map(|&(q,r,p)|
        check_win_probe((q,r), |c: Coord, d| (c.0+DIRS[d].0, c.1+DIRS[d].1), |c| m.get(&c) == Some(&p), 6) as u64).sum() });
    row("A. tuple + SipHash (current)", base, base);
    row("B. tuple + FxHashMap (one-line)", bench(3000, || { let m = black_box(&fxt); pos.iter().map(|&(q,r,p)|
        check_win_probe((q,r), |c: Coord, d| (c.0+DIRS[d].0, c.1+DIRS[d].1), |c| m.get(&c) == Some(&p), 6) as u64).sum() }), base);
    row("C. u64 packed + FxHashMap", bench(3000, || { let m = black_box(&fx64); pos.iter().map(|&(q,r,p)|
        check_win_probe(key64(q,r), |k: u64, d| k.wrapping_add(dd64[d]), |k| m.get(&k) == Some(&p), 6) as u64).sum() }), base);
    row("D. u32 packed + FxHashMap", bench(3000, || { let m = black_box(&fx32); pos.iter().map(|&(q,r,p)|
        check_win_probe(key32(q,r), |k: u32, d| k.wrapping_add(dd32[d]), |k| m.get(&k) == Some(&p), 6) as u64).sum() }), base);

    println!("\n== legal_moves, 120 stones, radius 8 ==");
    let offs = hex_offsets(8);
    let od64: Vec<u64> = offs.iter().map(|&(a, b)| d64(a, b)).collect();
    let od32: Vec<u32> = offs.iter().map(|&(a, b)| d32(a, b)).collect();
    let base = bench(300, || { let m = black_box(&sip);
        let mut c: HashSet<Coord> = HashSet::with_capacity(offs.len());
        for &s in m.keys() { for &(dq, dr) in &offs { let cell = (s.0+dq, s.1+dr);
            if !m.contains_key(&cell) { c.insert(cell); } } }
        let mut v: Vec<_> = c.into_iter().collect(); v.sort_unstable(); v.len() as u64 });
    row("A. tuple + SipHash (current)", base, base);
    row("B. tuple + FxHashMap (one-line)", bench(300, || { let m = black_box(&fxt);
        let mut c: FxHashSet<Coord> = FxHashSet::with_capacity_and_hasher(offs.len(), Default::default());
        for &s in m.keys() { for &(dq, dr) in &offs { let cell = (s.0+dq, s.1+dr);
            if !m.contains_key(&cell) { c.insert(cell); } } }
        let mut v: Vec<_> = c.into_iter().collect(); v.sort_unstable(); v.len() as u64 }), base);
    row("C. u64 packed + FxHashMap", bench(300, || { let m = black_box(&fx64);
        let mut c: FxHashSet<u64> = FxHashSet::with_capacity_and_hasher(od64.len(), Default::default());
        for &s in m.keys() { for &d in &od64 { let cell = s.wrapping_add(d);
            if !m.contains_key(&cell) { c.insert(cell); } } }
        let mut v: Vec<_> = c.into_iter().collect(); v.sort_unstable(); v.len() as u64 }), base);
    row("D. u32 packed + FxHashMap", bench(300, || { let m = black_box(&fx32);
        let mut c: FxHashSet<u32> = FxHashSet::with_capacity_and_hasher(od32.len(), Default::default());
        for &s in m.keys() { for &d in &od32 { let cell = s.wrapping_add(d);
            if !m.contains_key(&cell) { c.insert(cell); } } }
        let mut v: Vec<_> = c.into_iter().collect(); v.sort_unstable(); v.len() as u64 }), base);

    println!("\n== 6-neighbor occupancy probes × 120 stones (GNN edges) ==");
    let base = bench(20000, || { let m = black_box(&sip); let p = black_box(&pos);
        p.iter().map(|&(q,r,_)| DIRS.iter().filter(|&&(dq,dr)| m.contains_key(&(q+dq, r+dr))).count() as u64).sum() });
    row("A. tuple + SipHash (current)", base, base);
    row("B. tuple + FxHashMap (one-line)", bench(20000, || { let m = black_box(&fxt); let p = black_box(&pos);
        p.iter().map(|&(q,r,_)| DIRS.iter().filter(|&&(dq,dr)| m.contains_key(&(q+dq, r+dr))).count() as u64).sum() }), base);
    row("C. u64 packed + FxHashMap", bench(20000, || { let m = black_box(&fx64); let p = black_box(&pos);
        p.iter().map(|&(q,r,_)| { let k = key64(q,r); dd64.iter().filter(|&&d| m.contains_key(&k.wrapping_add(d))).count() as u64 }).sum() }), base);
    row("D. u32 packed + FxHashMap", bench(20000, || { let m = black_box(&fx32); let p = black_box(&pos);
        p.iter().map(|&(q,r,_)| { let k = key32(q,r); dd32.iter().filter(|&&d| m.contains_key(&k.wrapping_add(d))).count() as u64 }).sum() }), base);

    println!("\nmap entry sizes: (Coord,u8)={}B  (u64,u8)={}B  (u32,u8)={}B",
        std::mem::size_of::<(Coord, u8)>(), std::mem::size_of::<(u64, u8)>(), std::mem::size_of::<(u32, u8)>());
}
