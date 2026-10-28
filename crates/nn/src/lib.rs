//! Inference for small fully connected networks (multi-layer perceptrons) trained
//! in PyTorch, so the Rust search can call a network without Python or a deep
//! learning library.
//!
//! A network is a stack of [`Dense`] layers with ReLU between them (none after
//! the last). Inputs are usually 0/1 features, so [`Mlp::forward_sparse`] takes
//! the indices of the inputs that are 1 and skips the multiplications by zero.
//!
//! A file can hold several such stacks ("towers") that the caller wires together,
//! e.g. a state tower, a move tower and a head: see [`Graph`].
//!
//! # File format (`.pknn`)
//!
//! Little-endian. **Version 1** (one tower, f32 weights), written by
//! `big2.rl.value.export_pknn` in Python:
//!
//! ```text
//! b"PKNN"  u32 version (= 1)
//! u32 meta_len, meta_len bytes of UTF-8 "key=value" lines
//! u32 n_layers
//! per layer: u32 n_in, u32 n_out, f32 weight[n_out × n_in] (row-major, as
//!            torch.nn.Linear stores it), f32 bias[n_out]
//! ```
//!
//! **Version 2** (named towers, smaller weights), written by
//! `big2.rl.export.export_qnet_pknn`:
//!
//! ```text
//! b"PKNN"  u32 version (= 2)
//! u32 meta_len, meta_len bytes of UTF-8 "key=value" lines
//! u32 n_towers
//! per tower: u32 name_len, name_len bytes of UTF-8 name, u32 n_layers
//!   per layer: u32 n_in, u32 n_out, u8 precision
//!     precision 0 (f32):  f32 weight[n_out × n_in]
//!     precision 1 (int8): u32 group, f16 scale[n_out × ⌈n_in / group⌉],
//!                         i8 q[n_out × n_in]
//!                         (weight[o][i] = scale[o][i / group] × q[o][i])
//!     precision 2 (f16):  f16 weight[n_out × n_in]
//!     f32 bias[n_out]
//! ```
//!
//! Int8 uses one scale per run of `group` weights within a row (`group = n_in`
//! gives one scale per row). Weights are expanded to f32 when the file is read,
//! so the smaller formats cost disk bytes only, not speed. The metadata says
//! what the network expects, e.g. `features=big2-full-v1`.

use std::collections::BTreeMap;

/// How a layer's weights are stored in a version 2 file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// 4 bytes per weight, exact.
    F32,
    /// 2 bytes per weight (IEEE half precision, about 3 significant digits).
    F16,
    /// 1 byte per weight plus one f16 scale per `group` weights of a row
    /// (symmetric, round to nearest); see [`quantize_groups`].
    Int8 { group: usize },
}

/// IEEE half precision bits → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        // Subnormal: man × 2^-24.
        0 => return sign_of(h) * man as f32 * (-24f32).exp2(),
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

fn sign_of(h: u16) -> f32 {
    if h & 0x8000 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// f32 → IEEE half precision bits, rounding to nearest even (as NumPy's
/// `astype(np.float16)`); too large becomes infinity.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let round = |v: u32, rem: u32, halfway: u32| {
        if rem > halfway || (rem == halfway && v & 1 == 1) {
            v + 1
        } else {
            v
        }
    };
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let v = m >> shift;
        return sign | round(v, m & ((1 << shift) - 1), 1 << (shift - 1)) as u16;
    }
    // A carry out of the mantissa moves into the exponent, as it should.
    sign | round(((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000) as u16
}

/// One fully connected layer: `y = W x + b`.
#[derive(Clone, Debug)]
pub struct Dense {
    pub n_in: usize,
    pub n_out: usize,
    /// `n_out × n_in`, row-major.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Dense {
    fn forward(&self, x: &[f32], y: &mut Vec<f32>) {
        y.clear();
        y.extend(
            self.weight
                .chunks_exact(self.n_in)
                .zip(&self.bias)
                .map(|(row, b)| b + dot(row, x)),
        );
    }
}

/// Dot product with eight independent partial sums. A single running sum forces
/// the additions into one sequence (floating-point addition is not reassociated
/// by the compiler); eight lanes let it use SIMD instructions.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// A multi-layer perceptron with ReLU activations between layers.
#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<Dense>,
    /// `key=value` metadata from the file.
    pub meta: BTreeMap<String, String>,
    /// The first layer's weights transposed (`n_in × n_out`) for sparse inputs.
    first_t: Vec<f32>,
}

impl Mlp {
    /// Builds a network from layers (each layer's `n_in` must match the previous `n_out`).
    pub fn new(layers: Vec<Dense>, meta: BTreeMap<String, String>) -> Result<Mlp, String> {
        if layers.is_empty() {
            return Err("a network needs at least one layer".into());
        }
        for (i, l) in layers.iter().enumerate() {
            if l.weight.len() != l.n_in * l.n_out || l.bias.len() != l.n_out {
                return Err(format!("layer {i}: wrong weight or bias size"));
            }
            if i > 0 && layers[i - 1].n_out != l.n_in {
                return Err(format!(
                    "layer {i}: input size does not match layer {}",
                    i - 1
                ));
            }
        }
        let f = &layers[0];
        let mut first_t = vec![0.0; f.n_in * f.n_out];
        for o in 0..f.n_out {
            for i in 0..f.n_in {
                first_t[i * f.n_out + o] = f.weight[o * f.n_in + i];
            }
        }
        Ok(Mlp {
            layers,
            meta,
            first_t,
        })
    }

    /// Reads a `.pknn` file (see the crate docs): version 1, or version 2 with
    /// exactly one tower. The file's metadata becomes [`Mlp::meta`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Mlp, String> {
        let mut g = Graph::from_bytes(bytes)?;
        if g.towers.len() != 1 {
            return Err(format!(
                "expected one network, the file has {} towers",
                g.towers.len()
            ));
        }
        let (_, mut mlp) = g.towers.remove(0);
        mlp.meta = g.meta;
        Ok(mlp)
    }

    /// Number of inputs.
    pub fn n_in(&self) -> usize {
        self.layers[0].n_in
    }

    /// Number of outputs.
    pub fn n_out(&self) -> usize {
        self.layers.last().unwrap().n_out
    }

    /// Forward pass on a dense input vector.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.n_in(), "input size");
        let mut h = Vec::new();
        self.layers[0].forward(x, &mut h);
        self.rest(h)
    }

    /// Forward pass on a 0/1 input given by the indices of its ones.
    pub fn forward_sparse(&self, active: &[usize]) -> Vec<f32> {
        let f = &self.layers[0];
        let mut h = f.bias.clone();
        for &i in active {
            let col = &self.first_t[i * f.n_out..(i + 1) * f.n_out];
            for (y, w) in h.iter_mut().zip(col) {
                *y += w;
            }
        }
        self.rest(h)
    }

    /// The remaining layers after the first, with ReLU before each.
    fn rest(&self, mut h: Vec<f32>) -> Vec<f32> {
        let mut next = Vec::new();
        for l in &self.layers[1..] {
            for v in h.iter_mut() {
                *v = v.max(0.0);
            }
            l.forward(&h, &mut next);
            std::mem::swap(&mut h, &mut next);
        }
        h
    }
}

/// Several named networks ("towers") from one `.pknn` file, plus the file's
/// metadata. The caller decides how the towers connect.
#[derive(Clone, Debug)]
pub struct Graph {
    /// `key=value` metadata from the file.
    pub meta: BTreeMap<String, String>,
    /// Towers in file order. A version 1 file gives one tower named `main`.
    pub towers: Vec<(String, Mlp)>,
}

/// The tower name a version 1 file's single network gets.
pub const V1_TOWER: &str = "main";

impl Graph {
    /// Reads a `.pknn` file of version 1 or 2 (see the crate docs).
    pub fn from_bytes(bytes: &[u8]) -> Result<Graph, String> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(4)? != b"PKNN" {
            return Err("not a .pknn file".into());
        }
        let version = r.u32()?;
        if version != 1 && version != 2 {
            return Err(format!("unsupported .pknn version {version}"));
        }
        let meta_len = r.u32()? as usize;
        let text = std::str::from_utf8(r.take(meta_len)?).map_err(|e| e.to_string())?;
        let meta = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        let mut towers = Vec::new();
        if version == 1 {
            let layers = r.layers(false)?;
            towers.push((V1_TOWER.to_string(), Mlp::new(layers, BTreeMap::new())?));
        } else {
            let n_towers = r.u32()? as usize;
            for _ in 0..n_towers {
                let name_len = r.u32()? as usize;
                let name = std::str::from_utf8(r.take(name_len)?)
                    .map_err(|e| e.to_string())?
                    .to_string();
                let layers = r.layers(true)?;
                let mlp = Mlp::new(layers, BTreeMap::new()).map_err(|e| format!("{name}: {e}"))?;
                towers.push((name, mlp));
            }
        }
        if r.pos != bytes.len() {
            return Err("trailing bytes after the last layer".into());
        }
        Ok(Graph { meta, towers })
    }

    /// The tower called `name`, if the file has one.
    pub fn tower(&self, name: &str) -> Option<&Mlp> {
        self.towers.iter().find(|(n, _)| n == name).map(|(_, m)| m)
    }

    /// Writes the graph as a version 2 file with every layer at `precision`.
    /// Mostly for tests: the Python exporter writes the files that ship.
    pub fn to_bytes(&self, precision: Precision) -> Vec<u8> {
        let mut b = b"PKNN".to_vec();
        b.extend(2u32.to_le_bytes());
        let text: String = self
            .meta
            .iter()
            .map(|(k, v)| format!("{k}={v}\n"))
            .collect();
        b.extend((text.len() as u32).to_le_bytes());
        b.extend(text.as_bytes());
        b.extend((self.towers.len() as u32).to_le_bytes());
        for (name, mlp) in &self.towers {
            b.extend((name.len() as u32).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend((mlp.layers.len() as u32).to_le_bytes());
            for l in &mlp.layers {
                b.extend((l.n_in as u32).to_le_bytes());
                b.extend((l.n_out as u32).to_le_bytes());
                match precision {
                    Precision::F32 => {
                        b.push(0);
                        b.extend(l.weight.iter().flat_map(|w| w.to_le_bytes()));
                    }
                    Precision::F16 => {
                        b.push(2);
                        b.extend(l.weight.iter().flat_map(|&w| f32_to_f16(w).to_le_bytes()));
                    }
                    Precision::Int8 { group } => {
                        let group = group.clamp(1, l.n_in.max(1));
                        b.push(1);
                        b.extend((group as u32).to_le_bytes());
                        let (scales, q) = quantize_groups(&l.weight, l.n_in, group);
                        b.extend(scales.iter().flat_map(|&s| f32_to_f16(s).to_le_bytes()));
                        b.extend(q.iter().map(|&v| v as u8));
                    }
                }
                b.extend(l.bias.iter().flat_map(|w| w.to_le_bytes()));
            }
        }
        b
    }
}

/// Symmetric int8 quantization in groups of `group` weights along each row, as
/// `big2.rl.export.quantize_groups` does it: `scale = max |w| / 127` per group,
/// rounded to f16 (1 for an all-zero group), then `q = round(w / scale)`.
/// Returns the scales (row by row, f16-exact) and the quantized weights.
pub fn quantize_groups(weight: &[f32], n_in: usize, group: usize) -> (Vec<f32>, Vec<i8>) {
    let mut scales = Vec::new();
    let mut q = Vec::with_capacity(weight.len());
    for row in weight.chunks_exact(n_in) {
        for g in row.chunks(group) {
            let max = g.iter().fold(0.0f32, |m, w| m.max(w.abs()));
            // A group too small for an f16 scale is stored as zeros (scale 1).
            let scale = f16_to_f32(f32_to_f16(max / 127.0));
            let scale = if scale > 0.0 { scale } else { 1.0 };
            scales.push(scale);
            q.extend(
                g.iter()
                    .map(|w| (w / scale).round().clamp(-127.0, 127.0) as i8),
            );
        }
    }
    (scales, q)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.bytes.len());
        let end = end.ok_or("file ends early")?;
        let s = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let len = n.checked_mul(4).ok_or("layer too large")?;
        Ok(self
            .take(len)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    fn f16s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let len = n.checked_mul(2).ok_or("layer too large")?;
        Ok(self
            .take(len)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect())
    }

    /// A layer list: `u32 n_layers`, then each layer. Version 2 layers carry a
    /// precision byte (`with_precision`); version 1 layers are always f32.
    fn layers(&mut self, with_precision: bool) -> Result<Vec<Dense>, String> {
        let n_layers = self.u32()? as usize;
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            let n_in = self.u32()? as usize;
            let n_out = self.u32()? as usize;
            let n = n_in.checked_mul(n_out).ok_or("layer too large")?;
            let precision = if with_precision { self.take(1)?[0] } else { 0 };
            let weight = match precision {
                0 => self.f32s(n)?,
                1 => {
                    let group = self.u32()? as usize;
                    if group == 0 || n_in == 0 {
                        return Err("int8 layer with group 0 or no inputs".into());
                    }
                    let per_row = n_in.div_ceil(group);
                    let scales = self.f16s(n_out * per_row)?;
                    let q = self.take(n)?;
                    let mut w = Vec::with_capacity(n);
                    for (row, s) in q.chunks_exact(n_in).zip(scales.chunks_exact(per_row)) {
                        for (g, s) in row.chunks(group).zip(s) {
                            w.extend(g.iter().map(|&v| v as i8 as f32 * s));
                        }
                    }
                    w
                }
                2 => self.f16s(n)?,
                p => return Err(format!("unknown weight precision {p}")),
            };
            let bias = self.f32s(n_out)?;
            layers.push(Dense {
                n_in,
                n_out,
                weight,
                bias,
            });
        }
        Ok(layers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Mlp {
        // 3 → 2 → 1:  h = relu([x0 + x1, x2 − x0 − 1]),  y = 2·h0 − h1 + 0.5
        let l1 = Dense {
            n_in: 3,
            n_out: 2,
            weight: vec![1.0, 1.0, 0.0, -1.0, 0.0, 1.0],
            bias: vec![0.0, -1.0],
        };
        let l2 = Dense {
            n_in: 2,
            n_out: 1,
            weight: vec![2.0, -1.0],
            bias: vec![0.5],
        };
        Mlp::new(vec![l1, l2], BTreeMap::new()).unwrap()
    }

    #[test]
    fn dense_and_sparse_forward_agree() {
        let m = tiny();
        assert_eq!(m.forward(&[1.0, 1.0, 0.0]), vec![4.5]); // h = [2, 0]
        assert_eq!(m.forward(&[0.0, 0.0, 1.0]), vec![0.5]); // h = [0, 0]
        for bits in 0..8usize {
            let x: Vec<f32> = (0..3).map(|i| ((bits >> i) & 1) as f32).collect();
            let active: Vec<usize> = (0..3).filter(|i| (bits >> i) & 1 == 1).collect();
            assert_eq!(m.forward(&x), m.forward_sparse(&active));
        }
    }

    #[test]
    fn reads_the_file_format() {
        let mut b = b"PKNN".to_vec();
        b.extend(1u32.to_le_bytes());
        let meta = b"features=test\nscale=2";
        b.extend((meta.len() as u32).to_le_bytes());
        b.extend(meta);
        b.extend(1u32.to_le_bytes());
        b.extend(2u32.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        for v in [3.0f32, -1.0, 0.25] {
            b.extend(v.to_le_bytes());
        }
        let m = Mlp::from_bytes(&b).unwrap();
        assert_eq!(m.meta["features"], "test");
        assert_eq!(m.forward(&[1.0, 2.0]), vec![1.25]);
        assert!(Mlp::from_bytes(&b[..b.len() - 1]).is_err());
        assert!(Mlp::from_bytes(b"NOPE").is_err());
        // Version 1 still loads as a graph, with one tower named "main".
        let g = Graph::from_bytes(&b).unwrap();
        assert_eq!(g.towers.len(), 1);
        assert_eq!(g.tower(V1_TOWER).unwrap().forward(&[1.0, 2.0]), vec![1.25]);
        assert_eq!(g.meta["scale"], "2");
    }

    fn two_towers() -> Graph {
        let other = Dense {
            n_in: 2,
            n_out: 2,
            weight: vec![0.5, -0.25, 1.0, 0.0],
            bias: vec![0.1, -0.2],
        };
        Graph {
            meta: BTreeMap::from([("kind".to_string(), "test".to_string())]),
            towers: vec![
                ("a".to_string(), tiny()),
                (
                    "b".to_string(),
                    Mlp::new(vec![other], BTreeMap::new()).unwrap(),
                ),
            ],
        }
    }

    #[test]
    fn version_2_round_trips_in_f32_exactly() {
        let g = two_towers();
        let back = Graph::from_bytes(&g.to_bytes(Precision::F32)).unwrap();
        assert_eq!(back.meta["kind"], "test");
        assert_eq!(
            back.towers
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        for x in [[1.0, 1.0, 0.0], [0.0, 2.0, 3.0]] {
            assert_eq!(back.tower("a").unwrap().forward(&x), tiny().forward(&x));
        }
        assert_eq!(
            back.tower("b").unwrap().forward(&[1.0, 1.0]),
            vec![0.35, 0.8]
        );
        assert!(back.tower("c").is_none());
        // Two towers are not one network.
        assert!(Mlp::from_bytes(&g.to_bytes(Precision::F32)).is_err());
    }

    #[test]
    fn half_precision_conversions() {
        for (x, h) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.5, 0x3800),
            (65504.0, 0x7bff),
            (1e6, 0x7c00),
            ((-24f32).exp2(), 0x0001), // smallest subnormal
            ((-14f32).exp2(), 0x0400), // smallest normal
        ] {
            assert_eq!(f32_to_f16(x), h, "{x}");
            if h != 0x7c00 {
                assert_eq!(f16_to_f32(h), x, "{h:#x}");
            }
        }
        // Halfway cases round to the even neighbour, down then up.
        assert_eq!(f32_to_f16(1.0 + (-11f32).exp2()), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * (-11f32).exp2()), 0x3c02);
        assert!(f16_to_f32(0x7e00).is_nan());
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0x8001), -(-24f32).exp2());
    }

    #[test]
    fn version_2_f16_round_trips_within_half_precision() {
        let g = two_towers();
        let back = Graph::from_bytes(&g.to_bytes(Precision::F16)).unwrap();
        for (a, b) in g.towers.iter().zip(&back.towers) {
            for (la, lb) in a.1.layers.iter().zip(&b.1.layers) {
                for (x, y) in la.weight.iter().zip(&lb.weight) {
                    assert!((x - y).abs() <= x.abs() / 1024.0, "{x} vs {y}");
                }
                assert_eq!(la.bias, lb.bias);
            }
        }
    }

    #[test]
    fn version_2_int8_groups_are_small_and_close() {
        let mut rng = 12345u64;
        let mut next = || {
            // xorshift: a fixed pseudo-random sequence in [-1, 1)
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng >> 11) as f32 / (1u64 << 52) as f32 - 1.0
        };
        let layer = Dense {
            n_in: 64,
            n_out: 32,
            weight: (0..64 * 32).map(|_| next()).collect(),
            bias: (0..32).map(|_| next()).collect(),
        };
        let g = Graph {
            meta: BTreeMap::new(),
            towers: vec![(
                "t".to_string(),
                Mlp::new(vec![layer.clone()], BTreeMap::new()).unwrap(),
            )],
        };
        let f = g.to_bytes(Precision::F32);
        for group in [64, 16, 5] {
            let q = g.to_bytes(Precision::Int8 { group });
            assert!(q.len() * 2 < f.len(), "{} vs {}", q.len(), f.len());
            let back = Mlp::from_bytes(&q).unwrap();
            for (row, back_row) in layer
                .weight
                .chunks(64)
                .zip(back.layers[0].weight.chunks(64))
            {
                for (gw, gb) in row.chunks(group).zip(back_row.chunks(group)) {
                    let max = gw.iter().fold(0.0f32, |m, w| m.max(w.abs()));
                    for (w, b) in gw.iter().zip(gb) {
                        // Half a step, plus the f16 rounding of the scale.
                        assert!((w - b).abs() <= max / 254.0 + max / 1000.0, "{w} vs {b}");
                    }
                }
            }
            assert_eq!(back.layers[0].bias, layer.bias, "biases stay f32");
        }
        // Smaller groups are closer.
        let err = |group| {
            let back = Mlp::from_bytes(&g.to_bytes(Precision::Int8 { group })).unwrap();
            let w = &back.layers[0].weight;
            w.iter()
                .zip(&layer.weight)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        };
        assert!(err(8) < err(64));
    }

    #[test]
    fn rejects_bad_version_2_files() {
        let mut b = two_towers().to_bytes(Precision::F32);
        assert!(Graph::from_bytes(&b[..b.len() - 2]).is_err());
        b.push(0);
        assert!(Graph::from_bytes(&b).is_err(), "trailing byte");
        let mut b = b"PKNN".to_vec();
        b.extend(2u32.to_le_bytes());
        b.extend(0u32.to_le_bytes()); // no metadata
        b.extend(1u32.to_le_bytes()); // one tower
        b.extend(1u32.to_le_bytes());
        b.push(b'x');
        b.extend(1u32.to_le_bytes()); // one layer
        b.extend(1u32.to_le_bytes());
        b.extend(1u32.to_le_bytes());
        b.push(7); // unknown precision
        b.extend([0u8; 8]);
        let err = Graph::from_bytes(&b).unwrap_err();
        assert!(err.contains("precision"), "{err}");
        let mut v3 = b"PKNN".to_vec();
        v3.extend(3u32.to_le_bytes());
        assert!(Graph::from_bytes(&v3).unwrap_err().contains("version"));
    }
}
