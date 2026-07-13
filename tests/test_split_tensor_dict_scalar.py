import torch

from opsd_utils.teacher_batching import split_tensor_dict_for_opsd
from trainer.DyMETrainer import split_tensor_dict


def test_split_tensor_dict_copies_scalar_tensor_metadata():
    batch = {
        "prompt_ids": torch.arange(12).reshape(4, 3),
        "reward_std_mean": torch.tensor(0.125),
    }

    chunks = split_tensor_dict(batch, num_chunks=2)

    assert len(chunks) == 2
    assert chunks[0]["prompt_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert chunks[1]["prompt_ids"].tolist() == [[6, 7, 8], [9, 10, 11]]
    assert chunks[0]["reward_std_mean"].shape == torch.Size([])
    assert chunks[1]["reward_std_mean"].shape == torch.Size([])
    assert chunks[0]["reward_std_mean"].item() == 0.125
    assert chunks[1]["reward_std_mean"].item() == 0.125


def test_split_tensor_dict_for_opsd_copies_scalar_tensor_metadata():
    batch = {
        "prompt_ids": torch.arange(12).reshape(4, 3),
        "teacher_prompt_ids": torch.zeros(4, 5),
        "teacher_num_images": torch.tensor([1, 1, 1, 1]),
        "teacher_pixel_values": torch.zeros(4, 2),
        "reward_std_mean": torch.tensor(0.25),
    }

    chunks = split_tensor_dict_for_opsd(batch, num_chunks=2)

    assert len(chunks) == 2
    assert chunks[0]["prompt_ids"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert chunks[1]["prompt_ids"].tolist() == [[6, 7, 8], [9, 10, 11]]
    assert chunks[0]["teacher_pixel_values"].shape == (2, 2)
    assert chunks[1]["teacher_pixel_values"].shape == (2, 2)
    assert chunks[0]["reward_std_mean"].shape == torch.Size([])
    assert chunks[1]["reward_std_mean"].shape == torch.Size([])
    assert chunks[0]["reward_std_mean"].item() == 0.25
    assert chunks[1]["reward_std_mean"].item() == 0.25
