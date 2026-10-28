//! Inference for small fully connected networks (multi-layer perceptrons) trained
//! in PyTorch, so the Rust search can call a network without Python or a deep
//! learning library.
//!
//! A network is a stack of [`Dense`] layers with ReLU between them (none after
//! the last). Inputs are usually 0/1 features, so [`Mlp::forward_sparse`] takes
//! the indices of the inputs that are 1 and skips the multiplications by zero.
//!
//! # File format (`.pknn`)
//!
//! Little-endian, written by `big2.rl.value.export_pknn` in Python:
//!
//! ```text
//! b"PKNN"  u32 version (= 1)
//! u32 meta_len, meta_len bytes of UTF-8 "key=value" lines
//! u32 n_layers
//! per layer: u32 n_in, u32 n_out, f32 weight[n_out × n_in] (row-major, as
//!            torch.nn.Linear stores it), f32 bias[n_out]
//! ```
//!
//! The metadata says what the network expects, e.g. `features=big2-full-v1`.

use std::collections::BTreeMap;

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

    /// Reads the `.pknn` format (see the crate docs).
    pub fn from_bytes(bytes: &[u8]) -> Result<Mlp, String> {
        let mut r = Reader { bytes, pos: 0 };
        if r.take(4)? != b"PKNN" {
            return Err("not a .pknn file".into());
        }
        let version = r.u32()?;
        if version != 1 {
            return Err(format!("unsupported .pknn version {version}"));
        }
        let meta_len = r.u32()? as usize;
        let text = std::str::from_utf8(r.take(meta_len)?).map_err(|e| e.to_string())?;
        let meta = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        let n_layers = r.u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let n_in = r.u32()? as usize;
            let n_out = r.u32()? as usize;
            let weight = r.f32s(n_in * n_out)?;
            let bias = r.f32s(n_out)?;
            layers.push(Dense {
                n_in,
                n_out,
                weight,
                bias,
            });
        }
        if r.pos != bytes.len() {
            return Err("trailing bytes after the last layer".into());
        }
        Mlp::new(layers, meta)
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
        Ok(self
            .take(4 * n)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
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
    }
}
