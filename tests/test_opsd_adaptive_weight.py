from opsd_utils.adaptive_weight import effective_opsd_weight, opsd_adaptive_multiplier


def test_adaptive_multiplier_zero_std_uses_max_mult():
    multiplier = opsd_adaptive_multiplier(
        0.0,
        enabled=True,
        std_target=0.25,
        max_mult=2.0,
    )
    weight, weight_multiplier = effective_opsd_weight(
        1.5,
        0.0,
        enabled=True,
        std_target=0.25,
        max_mult=2.0,
    )

    assert multiplier == 2.0
    assert weight_multiplier == 2.0
    assert weight == 3.0


def test_adaptive_multiplier_target_or_higher_is_one():
    assert opsd_adaptive_multiplier(0.25, enabled=True, std_target=0.25, max_mult=2.0) == 1.0
    assert opsd_adaptive_multiplier(0.75, enabled=True, std_target=0.25, max_mult=2.0) == 1.0


def test_adaptive_multiplier_disabled_is_one():
    weight, multiplier = effective_opsd_weight(
        1.5,
        0.0,
        enabled=False,
        std_target=0.25,
        max_mult=2.0,
    )

    assert multiplier == 1.0
    assert weight == 1.5
