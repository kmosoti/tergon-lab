//! Scalar definitions and packed ternary representations.
//!
//! This is a project-native two-bit layout, not a GGUF or BitNet format.
//! See `docs/wiki/Mathematics.md` for the numerical contract.

use std::{error::Error, fmt, num::NonZeroUsize};

/// Contiguous absmean groups with a stored f32 scale and ties away from zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    group_size: NonZeroUsize,
}

impl QuantSpec {
    /// Reject a zero group size. A final partial group is permitted.
    pub fn new(group_size: usize) -> Result<Self, QuantError> {
        Ok(Self {
            group_size: NonZeroUsize::new(group_size).ok_or(QuantError::ZeroGroupSize)?,
        })
    }

    /// Maximum number of weights sharing one scale.
    pub fn group_size(self) -> usize {
        self.group_size.get()
    }
}

/// Invalid inputs are errors rather than implicit numerical repairs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// A group cannot be empty by specification.
    ZeroGroupSize,
    /// Input contains NaN or infinity at this index.
    NonFinite(usize),
    /// A nonzero group's absmean rounded to zero in f32.
    ScaleUnderflow(usize),
    /// Encoded parts violate the canonical representation.
    InvalidParts(&'static str),
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for QuantError {}

/// Immutable validated two-bit weights, with one f32 scale per group.
///
/// Low bits hold the first weight: 00 = zero, 01 = +1, 10 = -1.
/// Code 11 is invalid; unused high bits in the final byte must be zero.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedTernary {
    spec: QuantSpec,
    len: usize,
    scales: Vec<f32>,
    bytes: Vec<u8>,
}

impl PackedTernary {
    /// Quantize finite f32 values using f64 absmean accumulation.
    ///
    /// The mean is stored as f32 before classification. Comparing against half
    /// that stored scale implements round-to-nearest, ties away from zero,
    /// clamped to {-1, 0, +1}, without overflowing an intermediate division.
    pub fn quantize(input: &[f32], spec: QuantSpec) -> Result<Self, QuantError> {
        if let Some(i) = input.iter().position(|x| !x.is_finite()) {
            return Err(QuantError::NonFinite(i));
        }
        let mut scales = Vec::with_capacity(input.len().div_ceil(spec.group_size()));
        let mut bytes = vec![0; input.len().div_ceil(4)];
        for (group_id, group) in input.chunks(spec.group_size()).enumerate() {
            let mean = group.iter().map(|&x| f64::from(x).abs()).sum::<f64>() / group.len() as f64;
            let scale = mean as f32;
            if mean > 0.0 && scale == 0.0 {
                return Err(QuantError::ScaleUnderflow(group_id));
            }
            scales.push(scale);
            if scale == 0.0 {
                continue;
            }
            for (offset, &weight) in group.iter().enumerate() {
                let i = group_id * spec.group_size() + offset;
                let threshold = f64::from(scale) / 2.0;
                let code = if f64::from(weight) >= threshold {
                    1
                } else if f64::from(weight) <= -threshold {
                    2
                } else {
                    0
                };
                bytes[i / 4] |= code << (2 * (i % 4));
            }
        }
        Ok(Self {
            spec,
            len: input.len(),
            scales,
            bytes,
        })
    }

    /// Validate externally supplied parts before they enter the engine.
    ///
    /// This is an in-memory API, not a versioned file format. No allocation is
    /// based on an unchecked length field; callers own the supplied buffers.
    pub fn from_parts(
        spec: QuantSpec,
        len: usize,
        scales: Vec<f32>,
        bytes: Vec<u8>,
    ) -> Result<Self, QuantError> {
        if scales.len() != len.div_ceil(spec.group_size()) || bytes.len() != len.div_ceil(4) {
            return Err(QuantError::InvalidParts("buffer lengths"));
        }
        if scales
            .iter()
            .any(|x| !x.is_finite() || x.is_sign_negative())
        {
            return Err(QuantError::InvalidParts(
                "scale must be finite and nonnegative; no -0",
            ));
        }
        if !len.is_multiple_of(4) {
            let used_bits = 2 * (len % 4);
            if bytes.last().copied().unwrap_or(0) >> used_bits != 0 {
                return Err(QuantError::InvalidParts("nonzero padding"));
            }
        }
        for i in 0..len {
            let code = (bytes[i / 4] >> (2 * (i % 4))) & 3;
            if code == 3 {
                return Err(QuantError::InvalidParts("reserved trit code"));
            }
            if scales[i / spec.group_size()] == 0.0 && code != 0 {
                return Err(QuantError::InvalidParts("nonzero trit in zero-scale group"));
            }
        }
        Ok(Self {
            spec,
            len,
            scales,
            bytes,
        })
    }

    /// Quantization specification used to interpret this representation.
    pub fn spec(&self) -> QuantSpec {
        self.spec
    }
    /// Number of represented weights, excluding padding.
    pub fn len(&self) -> usize {
        self.len
    }
    /// Whether the representation has no weights.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Packed weight bytes, excluding scales and allocation overhead.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// One absmean scale for each contiguous group.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Decode weights lazily. This is scalar reference execution, not a kernel.
    pub fn dequantize(&self) -> impl Iterator<Item = f32> + '_ {
        (0..self.len).map(|i| {
            let scale = self.scales[i / self.spec.group_size()];
            match (self.bytes[i / 4] >> (2 * (i % 4))) & 3 {
                0 => 0.0,
                1 => scale,
                2 => -scale,
                _ => unreachable!("constructors validate trit codes"),
            }
        })
    }
}
