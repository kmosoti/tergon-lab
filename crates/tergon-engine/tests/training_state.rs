//! Fixed-recipe classifier and checkpoint state checks; no downloaded data.
use tergon_engine::{
    ClassifierSnapshot, ComputeDevice, ParameterSnapshot, PlacementPlan, TinyClassifier,
};
fn initial() -> ClassifierSnapshot {
    let mut state = 1_u64;
    ClassifierSnapshot {
        parameters: [128 * 784, 128, 10 * 128, 10].map(|n| ParameterSnapshot {
            values: (0..n)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((state >> 32) as i32 as f32) / (i32::MAX as f32) * 0.01
                })
                .collect(),
            moment_1: vec![],
            moment_2: vec![],
            time: 0,
        }),
    }
}
fn resume(compute: ComputeDevice, tolerance: f64) {
    for ternary in [false, true] {
        let plan = PlacementPlan { compute };
        let mut model = TinyClassifier::restore(&initial(), ternary, plan).unwrap();
        let x: Vec<_> = (0..2 * 784).map(|i| (i % 17) as f32 / 17.).collect();
        let labels = [1, 7];
        // Short deterministic fixture; measured MNIST runs check step200 +50.
        for _ in 0..3 {
            model.step(&x, &labels).unwrap();
        }
        let saved = model.snapshot().unwrap();
        let mut restored = TinyClassifier::restore(&saved, ternary, plan).unwrap();
        for _ in 0..4 {
            model.step(&x, &labels).unwrap();
            restored.step(&x, &labels).unwrap();
        }
        let a = model.snapshot().unwrap();
        let b = restored.snapshot().unwrap();
        for (a, b) in a.parameters.iter().zip(&b.parameters) {
            assert_eq!(a.time, b.time);
            for (a, b) in a
                .values
                .iter()
                .chain(&a.moment_1)
                .chain(&a.moment_2)
                .zip(b.values.iter().chain(&b.moment_1).chain(&b.moment_2))
            {
                if tolerance == 0. {
                    assert_eq!(a.to_bits(), b.to_bits());
                } else {
                    assert!(
                        (f64::from(*a) - f64::from(*b)).abs()
                            <= 1e-5 + tolerance * f64::from(*a).abs()
                    );
                }
            }
        }
        let before = model.snapshot().unwrap();
        assert!(model.step(&x, &[10, 0]).is_err());
        assert_eq!(model.snapshot().unwrap(), before);
    }
}
#[test]
fn cpu_optimizer_resume_is_bitwise() {
    resume(ComputeDevice::Cpu, 0.);
}
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires named NVIDIA GPU"]
fn cuda_optimizer_resume_matches() {
    resume(ComputeDevice::Cuda(0), 1e-4);
}
#[test]
fn malformed_optimizer_state_is_rejected() {
    let plan = PlacementPlan {
        compute: ComputeDevice::Cpu,
    };
    let mut bad = initial();
    bad.parameters[0].time = 1;
    assert!(TinyClassifier::restore(&bad, true, plan).is_err());
    let mut bad = initial();
    bad.parameters[0].values[0] = f32::NAN;
    assert!(TinyClassifier::restore(&bad, true, plan).is_err());
    let mut bad = initial();
    bad.parameters[3].values.pop();
    assert!(TinyClassifier::restore(&bad, true, plan).is_err());
}
