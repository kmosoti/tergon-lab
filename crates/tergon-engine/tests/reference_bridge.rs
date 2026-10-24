//! CPU reference and explicit placement checks.
use burn::tensor::{Tensor, TensorData};
use tergon_engine::{ComputeDevice, EngineError, PlacementPlan, materialize};
use tergon_math::{PackedTernary, QuantSpec};

#[test]
fn cpu_matrix_product_agrees_with_scalar_f64() {
    let packed =
        PackedTernary::quantize(&[-2., 0., 1., 3., -1., 0.5], QuantSpec::new(3).unwrap()).unwrap();
    let plan = PlacementPlan {
        compute: ComputeDevice::Cpu,
    };
    let weights = materialize(&packed, 2, 3, plan).unwrap();
    let input = [0.25_f32, -1., 2.];
    let rhs = Tensor::<2>::from_data(
        TensorData::new(input.to_vec(), [3, 1]),
        &plan.device().unwrap(),
    );
    let actual = weights.matmul(rhs).into_data().try_to_vec::<f32>().unwrap();
    for (row, &got) in packed
        .dequantize()
        .collect::<Vec<_>>()
        .chunks(3)
        .zip(&actual)
    {
        let expected: f64 = row
            .iter()
            .zip(input)
            .map(|(&w, x)| w as f64 * x as f64)
            .sum();
        assert!((got as f64 - expected).abs() <= 1e-6 + 1e-6 * expected.abs());
    }
}

#[test]
fn shape_mismatch_and_overflow_fail_before_backend_work() {
    let weights = PackedTernary::quantize(&[1.], QuantSpec::new(1).unwrap()).unwrap();
    let plan = PlacementPlan {
        compute: ComputeDevice::Cpu,
    };
    for (rows, cols) in [(0, 1), (2, 2), (usize::MAX, 2)] {
        assert_eq!(
            materialize(&weights, rows, cols, plan).unwrap_err(),
            EngineError::Shape
        );
    }
}

#[test]
#[cfg(not(feature = "cuda"))]
fn cuda_request_does_not_fall_back_to_cpu() {
    assert_eq!(
        PlacementPlan {
            compute: ComputeDevice::Cuda(0)
        }
        .device()
        .unwrap_err(),
        EngineError::CudaNotCompiled
    );
}
