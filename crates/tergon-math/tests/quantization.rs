//! Independent numerical and representation invariants.
use tergon_math::{PackedTernary, QuantError, QuantSpec};

// Independent equation-based oracle: division + rounding instead of the
// production threshold implementation. f64 is deliberate in both paths.
fn reference(input: &[f32], group_size: usize) -> Vec<f32> {
    input
        .chunks(group_size)
        .flat_map(|group| {
            let mut total = 0_f64;
            for &x in group {
                total += (x as f64).abs();
            }
            let scale = (total / group.len() as f64) as f32;
            group.iter().map(move |&x| {
                if scale == 0.0 {
                    0.0
                } else {
                    let trit = ((x as f64) / (scale as f64)).round().clamp(-1.0, 1.0);
                    if trit == 0.0 {
                        0.0
                    } else {
                        (trit * scale as f64) as f32
                    }
                }
            })
        })
        .collect()
}

#[test]
fn agrees_with_independent_reference_across_groups_and_tails() {
    let mut state = 42_u64;
    for len in 0..193 {
        let input: Vec<f32> = (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 32) as i32) as f32 / 65536.0
            })
            .collect();
        for size in [1, 3, 4, 16, 64, 256] {
            let packed = PackedTernary::quantize(&input, QuantSpec::new(size).unwrap()).unwrap();
            assert_eq!(
                packed.dequantize().collect::<Vec<_>>(),
                reference(&input, size)
            );
            assert_eq!(
                PackedTernary::from_parts(
                    packed.spec(),
                    len,
                    packed.scales().to_vec(),
                    packed.bytes().to_vec()
                )
                .unwrap(),
                packed
            );
        }
    }
}

#[test]
fn ties_are_away_from_zero_and_zero_is_canonical() {
    // absmean = 1: both +/-0.5 are exact threshold ties.
    let packed =
        PackedTernary::quantize(&[0.5, -0.5, 1.5, -1.5], QuantSpec::new(4).unwrap()).unwrap();
    assert_eq!(packed.scales(), &[1.0]);
    assert_eq!(packed.bytes(), &[0b10011001]);
    assert_eq!(
        packed.dequantize().collect::<Vec<_>>(),
        vec![1.0, -1.0, 1.0, -1.0]
    );
    let zero = PackedTernary::quantize(&[-0.0, 0.0, 0.0], QuantSpec::new(8).unwrap()).unwrap();
    assert_eq!(zero.scales()[0].to_bits(), 0);
    assert_eq!(zero.bytes(), &[0]);
    assert!(zero.dequantize().all(|x| x.to_bits() == 0));
}

#[test]
fn exhaustive_small_packed_words_and_padding() {
    for len in 0_usize..=8 {
        for word in 0..3_usize.pow(len as u32) {
            let mut n = word;
            let mut bytes = vec![0; len.div_ceil(4)];
            let mut expected = Vec::new();
            for i in 0..len {
                let digit = (n % 3) as u8;
                n /= 3;
                bytes[i / 4] |= digit << (2 * (i % 4));
                expected.push([0.0, 2.0, -2.0][digit as usize]);
            }
            let value = PackedTernary::from_parts(
                QuantSpec::new(3).unwrap(),
                len,
                vec![2.0; len.div_ceil(3)],
                bytes,
            )
            .unwrap();
            assert_eq!(value.dequantize().collect::<Vec<_>>(), expected);
        }
    }
}

#[test]
fn rejects_nonfinite_underflow_and_malformed_inputs() {
    assert_eq!(QuantSpec::new(0), Err(QuantError::ZeroGroupSize));
    let spec = QuantSpec::new(4).unwrap();
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            PackedTernary::quantize(&[0.0, x], spec),
            Err(QuantError::NonFinite(1))
        );
    }
    assert_eq!(
        PackedTernary::quantize(&[f32::from_bits(1), 0.0, 0.0, 0.0], spec),
        Err(QuantError::ScaleUnderflow(0))
    );
    for (len, scales, bytes) in [
        (1, vec![1.0], vec![3]),
        (1, vec![1.0], vec![4]),
        (1, vec![0.0], vec![1]),
        (1, vec![-0.0], vec![0]),
        (1, vec![-1.0], vec![0]),
        (1, vec![f32::NAN], vec![0]),
        (1, vec![f32::INFINITY], vec![0]),
        (1, vec![], vec![0]),
        (usize::MAX, vec![], vec![]),
    ] {
        assert!(PackedTernary::from_parts(spec, len, scales, bytes).is_err());
    }
}

#[test]
fn maximum_finite_values_do_not_overflow() {
    let value =
        PackedTernary::quantize(&[f32::MAX, -f32::MAX], QuantSpec::new(2).unwrap()).unwrap();
    assert_eq!(
        value.dequantize().collect::<Vec<_>>(),
        vec![f32::MAX, -f32::MAX]
    );
}
