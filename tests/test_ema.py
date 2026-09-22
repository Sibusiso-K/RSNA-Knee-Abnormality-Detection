"""EMA must track a model's weights without ever touching the model itself."""
import pytest
import torch

from src.model.ema import EMA


def _tiny_model():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    return m


def test_rejects_out_of_range_decay():
    m = _tiny_model()
    with pytest.raises(ValueError):
        EMA(m, 1.0)
    with pytest.raises(ValueError):
        EMA(m, -0.1)


def test_shadow_starts_as_a_copy_of_the_model_at_construction():
    m = _tiny_model()
    ema = EMA(m, 0.9)
    for k, v in m.state_dict().items():
        assert torch.equal(ema.shadow[k], v)
        assert ema.shadow[k].data_ptr() != v.data_ptr()  # a real copy, not an alias


def test_update_moves_shadow_toward_current_by_exactly_one_minus_decay():
    m = _tiny_model()
    decay = 0.8
    ema = EMA(m, decay)
    before = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)  # simulate one optimizer step
    ema.update(m)
    for k, v in m.state_dict().items():
        want = decay * before[k] + (1 - decay) * v
        assert torch.allclose(ema.shadow[k], want, atol=1e-6)
    assert ema.num_updates == 1


def test_repeated_updates_converge_toward_a_fixed_model():
    m = _tiny_model()
    ema = EMA(m, 0.9)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(5.0)
    for _ in range(500):
        ema.update(m)
    for k, v in m.state_dict().items():
        assert torch.allclose(ema.shadow[k], v, atol=1e-3)


def test_update_never_mutates_the_source_models_own_tensors():
    m = _tiny_model()
    ema = EMA(m, 0.9)
    original = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    after_step = {k: v.clone() for k, v in m.state_dict().items()}
    ema.update(m)
    for k, v in m.state_dict().items():
        assert torch.equal(v, after_step[k])          # unchanged by ema.update
        assert not torch.equal(v, original[k])          # (sanity: the step itself did move it)
    # Mutating the shadow afterwards must not reach the source model either.
    for v in ema.shadow.values():
        v.add_(100.0)
    for k, v in m.state_dict().items():
        assert torch.equal(v, after_step[k])


def test_non_float_buffers_are_copied_not_averaged():
    m = _tiny_model()
    m.register_buffer("step_count", torch.tensor(0, dtype=torch.long))
    ema = EMA(m, 0.5)
    m.step_count.fill_(7)
    ema.update(m)
    assert ema.shadow["step_count"].item() == 7        # copied verbatim, not 0.5*0 + 0.5*7


def test_mismatched_state_dict_keys_are_rejected():
    m = _tiny_model()
    ema = EMA(m, 0.9)
    other = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError):
        ema.update(other)


def test_state_dict_is_a_plain_cpu_copy_independent_of_further_updates():
    m = _tiny_model()
    ema = EMA(m, 0.9)
    snap = ema.state_dict()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    for k, v in snap.items():
        assert v.device.type == "cpu"
        assert not torch.equal(v, ema.shadow[k].cpu())  # snapshot frozen at call time


def test_to_moves_shadow_device_and_keeps_values():
    m = _tiny_model()
    ema = EMA(m, 0.9)
    ema.to("cpu")  # exercised on CPU in CI; the same call path XLA/CUDA use
    for k, v in m.state_dict().items():
        assert torch.equal(ema.shadow[k], v)
