import torch
from accelerate.utils.operations import convert_to_fp32

from axolotl.model_support.nemotron_diffusion.compat import (
    SelectedLogitsOutput,
    project_selected_logits,
)


def test_selected_logits_match_full_projection_and_head_gradients():
    torch.manual_seed(0)
    hidden = torch.randn(1, 9, 5, requires_grad=True)
    rows = torch.tensor([[0, 0, -1], [0, 0, -1]])
    positions = torch.tensor([[2, 7, -1], [4, 1, -1]])
    head = torch.nn.Linear(5, 11)
    selected = project_selected_logits(hidden, head, rows, positions)
    dense = head(hidden)[rows.clamp_min(0), positions.clamp_min(0)]
    torch.testing.assert_close(selected, dense)
    selected[..., :2].sum().backward()
    selected_grad = head.weight.grad.detach().clone()
    head.zero_grad()
    dense[..., :2].sum().backward()
    torch.testing.assert_close(selected_grad, head.weight.grad)


def test_selected_logits_output_survives_accelerate_conversion():
    output = SelectedLogitsOutput(
        logits=torch.ones(2, 3, 4, dtype=torch.bfloat16),
        axolotl_selected_logits=True,
    )
    converted = convert_to_fp32(output)
    assert isinstance(converted, SelectedLogitsOutput)
    assert converted.axolotl_selected_logits is True
    assert converted.logits.dtype is torch.float32
    assert converted.logits.shape == (2, 3, 4)
