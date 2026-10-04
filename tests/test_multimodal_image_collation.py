import pytest
import torch

from axolotl.utils.collators.multimodal import collate_image_inputs, image_model_inputs


def test_ragged_images_preserve_order_sizes_and_text_only_rows():
    batch = collate_image_inputs(
        [
            {"pixel_values": [torch.ones(3, 2, 4)], "image_sizes": [[2, 4]]},
            {},
            {
                "pixel_values": [
                    torch.full((3, 4, 2), 2.0),
                    torch.full((3, 1, 1), 3.0),
                ],
                "image_sizes": [[4, 2], [1, 1]],
            },
        ]
    )
    assert batch["pixel_values"].shape == (3, 3, 4, 4)
    assert batch["image_sizes"].tolist() == [[2, 4], [4, 2], [1, 1]]
    assert batch["pixel_values"][:, 0, 0, 0].tolist() == [1, 2, 3]
    assert not batch["pixel_values"][0, :, 2:, :].any()
    assert not batch["pixel_values"][1, :, :, 2:].any()
    assert collate_image_inputs([{}, {}]) == {}
    moved = image_model_inputs(batch, torch.device("cpu"))
    assert moved["image_sizes"].dtype == torch.long


@pytest.mark.parametrize(
    "inputs,match",
    [
        ({"pixel_values": []}, "together"),
        ({"pixel_values": [], "image_sizes": [[1, 1]]}, "image count"),
        ({"pixel_values": [torch.zeros(3, 2, 2)], "image_sizes": [[3, 2]]}, "extents"),
        ({"pixel_values": [torch.zeros(4, 2, 2)], "image_sizes": [[2, 2]]}, "shape"),
        ({"input_features": torch.zeros(2, 4)}, "Unsupported"),
    ],
)
def test_image_collation_rejects_ambiguous_inputs(inputs, match):
    with pytest.raises(ValueError, match=match):
        collate_image_inputs([inputs])
