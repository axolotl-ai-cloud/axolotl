import torch


def test_sonicmoe_dispatches_instance_multilora_hook_before_cuda_or_peft_resolution(
    monkeypatch,
):
    from axolotl.integrations.kernels.libs.sonicmoe import experts

    class Module:
        has_gate = True

    module = Module()
    seen = []

    def forward(owner, hidden, index, weights):
        seen.append((owner, hidden, index, weights))
        return hidden + 3

    module._axolotl_multilora_sonic_forward = forward
    monkeypatch.setattr(
        experts,
        "_resolve_weights_and_lora",
        lambda _: (_ for _ in ()).throw(AssertionError("resolver must not run")),
    )
    hidden = torch.ones(2, 4)
    result = experts.sonicmoe_experts_forward_with_lora(
        module, hidden, torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1)
    )
    assert result.equal(hidden + 3)
    assert seen[0][0] is module
