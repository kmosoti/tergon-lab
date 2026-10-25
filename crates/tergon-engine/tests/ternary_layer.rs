//! Independent scalar forward and analytic identity-STE backward fixtures.
use burn::tensor::{Tensor, TensorData};
use tergon_engine::{ComputeDevice, EngineError, PlacementPlan, TernaryLinear};

fn values<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_to_vec().unwrap()
}
fn assert_close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            (f64::from(a) - e).abs() <= tolerance + tolerance * e.abs(),
            "{a} != {e}"
        );
    }
}
fn fixture(compute: ComputeDevice, tolerance: f64) {
    // Five columns exercise a packed-byte tail; the first row has threshold ties.
    let w = [0.5_f32, -0.5, 2., -2., 0., 0., 0., 0., 0., 0.];
    let b = [0.25_f32, -0.75];
    let x = [1.25_f32, -2., 0.5, 3., -1., -0.5, 0.25, 2., -1., 4.];
    let g = [0.5_f32, -2., 1.5, 0.25];
    let layer = TernaryLinear::new(&w, &b, 5, PlacementPlan { compute }).unwrap();
    let input = Tensor::<2>::from_data(
        TensorData::new(x.to_vec(), [2, 5]),
        &layer.weights().device(),
    )
    .require_grad();
    let upstream = Tensor::<2>::from_data(
        TensorData::new(g.to_vec(), [2, 2]),
        &layer.weights().device(),
    );
    // Independent division/rounding oracle, including stored f32 scale rounding.
    let q: Vec<f64> = w
        .chunks(5)
        .flat_map(|row| {
            let scale = (row.iter().map(|v| f64::from(*v).abs()).sum::<f64>() / 5.) as f32;
            row.iter().map(move |v| {
                if scale == 0. {
                    0.
                } else {
                    f64::from(scale) * (f64::from(*v) / f64::from(scale)).round().clamp(-1., 1.)
                }
            })
        })
        .collect();
    let mut y = vec![0.; 4];
    let mut dx = vec![0.; 10];
    let mut dw = vec![0.; 10];
    let mut db = vec![0.; 2];
    for n in 0..2 {
        for o in 0..2 {
            y[n * 2 + o] = f64::from(b[o]);
            db[o] += f64::from(g[n * 2 + o]);
            for i in 0..5 {
                y[n * 2 + o] += f64::from(x[n * 5 + i]) * q[o * 5 + i];
                dx[n * 5 + i] += f64::from(g[n * 2 + o]) * q[o * 5 + i];
                dw[o * 5 + i] += f64::from(g[n * 2 + o]) * f64::from(x[n * 5 + i]);
            }
        }
    }
    let output = layer.forward(input.clone()).unwrap();
    assert_close(&values(output.clone()), &y, tolerance);
    let mut gradients = (output * upstream).sum().backward();
    assert_close(
        &values(input.grad_remove(&mut gradients).unwrap()),
        &dx,
        tolerance,
    );
    assert_close(
        &values(layer.weights().grad_remove(&mut gradients).unwrap()),
        &dw,
        tolerance,
    );
    assert_close(
        &values(layer.bias().grad_remove(&mut gradients).unwrap()),
        &db,
        tolerance,
    );
    let export = layer.packed().unwrap();
    assert_eq!(export.dequantize().map(f64::from).collect::<Vec<_>>(), q);
}

#[test]
fn cpu_forward_and_identity_surrogate_match_independent_references() {
    fixture(ComputeDevice::Cpu, 1e-6);
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires an explicitly provisioned NVIDIA GPU; record named-device execution"]
fn cuda_forward_and_identity_surrogate_match_independent_references() {
    fixture(ComputeDevice::Cuda(0), 1e-5);
}

#[test]
fn invalid_parameters_fail_before_backend_work() {
    let plan = PlacementPlan {
        compute: ComputeDevice::Cpu,
    };
    for (w, b, inputs) in [
        (vec![], vec![0.], 0),
        (vec![1.], vec![], 1),
        (vec![1.], vec![0., 0.], usize::MAX),
    ] {
        assert_eq!(
            TernaryLinear::new(&w, &b, inputs, plan).unwrap_err(),
            EngineError::Shape
        );
    }
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            TernaryLinear::new(&[v], &[0.], 1, plan),
            Err(EngineError::Quantization(_))
        ));
        assert_eq!(
            TernaryLinear::new(&[1.], &[v], 1, plan).unwrap_err(),
            EngineError::NonFinite
        );
    }
    assert!(matches!(
        TernaryLinear::new(&[f32::from_bits(1), 0., 0.], &[0.], 3, plan),
        Err(EngineError::Quantization(_))
    ));
}

#[test]
fn forward_rejects_wrong_shape_and_preserves_export() {
    let layer = TernaryLinear::new(
        &[1., 2.],
        &[0.],
        2,
        PlacementPlan {
            compute: ComputeDevice::Cpu,
        },
    )
    .unwrap();
    let packed = layer.packed().unwrap();
    let input = Tensor::<2>::zeros([1, 3], &layer.weights().device());
    assert_eq!(layer.forward(input).unwrap_err(), EngineError::Shape);
    assert_eq!(layer.packed().unwrap(), packed);
}

#[test]
fn nonfinite_and_empty_inputs_are_rejected() {
    let layer = TernaryLinear::new(
        &[1.],
        &[0.],
        1,
        PlacementPlan {
            compute: ComputeDevice::Cpu,
        },
    )
    .unwrap();
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input =
            Tensor::<2>::from_data(TensorData::new(vec![v], [1, 1]), &layer.weights().device());
        assert_eq!(layer.forward(input).unwrap_err(), EngineError::NonFinite);
    }
    let input = Tensor::<2>::zeros([0, 1], &layer.weights().device());
    assert_eq!(layer.forward(input).unwrap_err(), EngineError::Shape);
}
