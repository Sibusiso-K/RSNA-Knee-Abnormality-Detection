"""Exercise the exact helpers baked into the Kaggle training script."""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch


SOURCE = Path(__file__).parents[1] / "notebooks/kaggle_06_train_slots.py"


def helpers():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"weight_fingerprint", "construct_fold_model"}]
    namespace = dict(torch=torch, np=np, SEED=17, PER_FOLD_SEED=True,
                     UNFREEZE_LAST=6, POOL="cls_mean_focal", HEAD="xattn",
                     log=lambda message: None,
                     SlotNet=lambda backbone, **kwargs: torch.nn.Linear(3, 2))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class State(torch.nn.Module):
    def __init__(self, value, name="value"):
        super().__init__()
        self.register_buffer(name, value)


@pytest.mark.parametrize("value", [torch.tensor(3), torch.tensor([1.], dtype=torch.bfloat16),
                                  torch.empty(0), torch.arange(6).reshape(2, 3).T])
def test_raw_hash_supports_buffer_dtypes_and_layouts(value):
    fingerprint = helpers()["weight_fingerprint"]
    assert fingerprint(State(value)) == fingerprint(State(value.clone().contiguous()))


def test_hash_records_dtype_shape_name_and_bits():
    fingerprint = helpers()["weight_fingerprint"]
    values = [State(torch.tensor([0.])), State(torch.tensor([-0.])),
              State(torch.tensor([0.], dtype=torch.float64)),
              State(torch.tensor([[0.]])), State(torch.tensor([0.]), "other")]
    assert len({fingerprint(value) for value in values}) == len(values)


def test_fold_seed_does_not_depend_on_previous_construction():
    ns = helpers()
    model, seed = ns["construct_fold_model"](None, 1)
    expected = ns["weight_fingerprint"](model)
    ns["construct_fold_model"](None, 0)
    model, seed_again = ns["construct_fold_model"](None, 1)
    assert seed == seed_again == 18
    assert ns["weight_fingerprint"](model) == expected


def test_verification_exits_before_device_and_training_data():
    source = SOURCE.read_text(encoding="utf-8")
    branch = source.index("if VERIFY_INIT_ONLY:")
    exit_at = source.index("raise SystemExit(0)", branch)
    assert exit_at < source.index("# --- device:") < source.index('cache_dir =')
    assert source.index("model, fold_seed = construct_fold_model(DINOV2_MODEL, fold)") < source.index("model = model.to(device)")
