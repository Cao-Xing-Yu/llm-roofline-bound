"""The generated GLM-5.3 tensor inventory must match the checkpoint exactly."""

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from rooflinebound.spec import load_model

FIXTURE = Path(__file__).parent / "fixtures" / "glm-5.3.safetensors-summary.json"


def fold(tensors):
    g = defaultdict(lambda: [0, set()])
    for name, dtype, shape in tensors:
        m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
        if m:
            key = "layers.{L}." + re.sub(r"experts\.\d+\.", "experts.{E}.", m.group(2))
            g[(key, dtype, tuple(shape))][1].add(int(m.group(1)))
        else:
            key = name
        g[(key, dtype, tuple(shape))][0] += 1
    return {k: (n, sorted(layers)) for k, (n, layers) in g.items()}


@pytest.fixture(scope="module")
def checkpoint():
    return json.loads(FIXTURE.read_text("utf-8"))


@pytest.fixture(scope="module")
def model():
    return load_model("glm-5.3")


def test_every_tensor_matches_checkpoint(model, checkpoint):
    expected = {(r["pattern"], r["dtype"], tuple(r["shape"])): (r["count"], r["layers"])
                for r in checkpoint["tensors"]}
    got = fold((t.name, t.dtype, t.shape) for t in model.inventory)
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    assert not missing and not extra, f"missing={missing[:5]} extra={extra[:5]}"
    assert got == expected


def test_total_bytes_and_count_exact(model, checkpoint):
    assert sum(t.nbytes for t in model.inventory) == checkpoint["total_size"] == 755_617_140_416
    assert len(model.inventory) == checkpoint["n_tensors"] == 118_629


def test_tensor_names_unique(model):
    names = [t.name for t in model.inventory]
    assert len(names) == len(set(names))
