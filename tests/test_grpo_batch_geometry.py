from trainer.DyMETrainer import validate_grpo_batch_geometry


def test_validate_grpo_batch_geometry_accepts_supported_local_shape():
    validate_grpo_batch_geometry(
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_processes=2,
    )


def test_validate_grpo_batch_geometry_rejects_local_shape_mismatch():
    try:
        validate_grpo_batch_geometry(
            num_generations=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_processes=2,
        )
    except ValueError as exc:
        assert "local effective batch size" in str(exc)
    else:
        raise AssertionError("expected local-shape validation failure")
