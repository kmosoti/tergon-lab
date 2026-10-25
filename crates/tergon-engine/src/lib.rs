//! Explicit execution boundaries for Tergon Lab.
//!
//! The first bridge expands packed values into f32 tensors. It is intentionally
//! a reference path; it does not implement packed GEMM.

mod ternary;
mod training;
pub use training::{ClassifierSnapshot, ParameterSnapshot, TinyClassifier};
pub use ternary::TernaryLinear;

use burn::tensor::{Device, Tensor, TensorData};
use std::{error::Error, fmt};
use tergon_math::{PackedTernary, QuantError};

/// A requested execution device. There is no automatic fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeDevice {
    /// Burn Flex on the host CPU.
    Cpu,
    /// An explicitly selected NVIDIA device; requires the `cuda` feature.
    Cuda(usize),
}

/// Placement for the current single-device reference path.
///
/// Future offload and memory budgets need their own reviewed contract. This
/// type does not imply an allocator limit or transfer scheduler.
#[derive(Clone, Copy, Debug)]
pub struct PlacementPlan {
    /// Compute destination; packed weights remain owned by the caller on CPU.
    pub compute: ComputeDevice,
}

/// Failures detected before calling the tensor backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// Rows and columns must be nonzero, fit usize, and match the weight count.
    Shape,
    /// Rebuild explicitly with CUDA; CPU is not silently substituted.
    CudaNotCompiled,
    /// Canonical host quantization rejected the latent values.
    Quantization(QuantError),
    /// Bias or input contains a nonfinite value.
    NonFinite,
    /// Tensors must share one explicitly chosen device.
    Device,
    /// Backend data could not be decoded as f32.
    Data,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for EngineError {}

impl PlacementPlan {
    /// Resolve the requested backend. CUDA initialization failures from Burn
    /// remain backend failures and must not be interpreted as CPU measurements.
    pub fn device(self) -> Result<Device, EngineError> {
        match self.compute {
            ComputeDevice::Cpu => Ok(Device::flex()),
            ComputeDevice::Cuda(index) => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Device::cuda(index))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = index;
                    Err(EngineError::CudaNotCompiled)
                }
            }
        }
    }
}

/// Materialize a row-major `[rows, columns]` f32 tensor on the requested device.
/// This allocates a host decode buffer and a backend tensor; it is not zero-copy.
pub fn materialize(
    packed: &PackedTernary,
    rows: usize,
    columns: usize,
    placement: PlacementPlan,
) -> Result<Tensor<2>, EngineError> {
    if rows == 0 || columns == 0 || rows.checked_mul(columns) != Some(packed.len()) {
        return Err(EngineError::Shape);
    }
    let device = placement.device()?;
    Ok(Tensor::from_data(
        TensorData::new(packed.dequantize().collect::<Vec<_>>(), [rows, columns]),
        &device,
    ))
}

impl From<QuantError> for EngineError {
    fn from(value: QuantError) -> Self {
        Self::Quantization(value)
    }
}
