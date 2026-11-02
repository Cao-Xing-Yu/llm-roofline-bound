"""Attention DP (SGLang --attn-dp-size): weights, KV and what fits.

Layout per SGLang @ e3718813ec5f: attention projections and the embedding
split over attention TP (tp / attn_dp), FFN, experts and lm_head over the full
TP; each DP-attention group caches only its own requests.
"""

import shlex

import pytest

from rooflinebound.bounds import bounds_report
from rooflinebound.cli import main
from rooflinebound.engines import parse_sglang
from rooflinebound.memory import Deployment, memory_report
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Quantity
from rooflinebound.spec import SpecError, load_hardware, load_model


@pytest.fixture(scope="module")
def glm():
    return load_model("glm-5.3")


@pytest.fixture(scope="module")
def h20():
    return load_hardware("h20-3e")


def _attn_tp_bytes(model):
    """Bytes of the tensors split over attention TP (attention and embedding shards)."""
    return sum(t.nbytes * n for t, n in model.tensor_groups(mtp=False)
               if t.component in ("attn", "embed") and t.shard.name != "REPLICATED")


def test_weights_grow_by_the_attention_shards(glm, h20):
    base = memory_report(glm, h20, Deployment(Parallel(8))).get("weights_per_gpu").value
    for dp in (2, 4, 8):
        w = memory_report(glm, h20, Deployment(Parallel(8, attn_dp=dp))
                          ).get("weights_per_gpu").value
        # each rank holds 1/(8/dp) of the attention shards instead of 1/8
        assert w - base == _attn_tp_bytes(glm) * dp // 8 - _attn_tp_bytes(glm) // 8


def test_dp8_glm_figures(glm, h20):
    res = Quantity.assumed(12.06e9, "B", "user")
    r = memory_report(glm, h20, Deployment(Parallel(8, attn_dp=8), kv_dtype="fp8",
                                           index_kv_dtype="fp8"),
                      context=131072, reserve=res)
    assert r.get("weights_per_gpu").value == 106_513_090_752
    assert r.get("gpu_weights_per_dp_group").value == _attn_tp_bytes(glm)
    kv = r.get("kv_per_token").value
    assert kv == memory_report(glm, h20, Deployment(Parallel(8), kv_dtype="fp8",
                                                    index_kv_dtype="fp8")
                               ).get("kv_per_token").value  # MLA: whole latent either way
    cap = h20["mem_capacity"].value
    pool = cap - 106_513_090_752
    assert r.get("kv_tokens").value == 8 * (pool // kv)
    assert r.get("max_concurrency").value == 8 * (pool // (131072 * kv))
    assert r.get("max_concurrency_assumed").value == 8 * ((pool - 12.06e9) // (131072 * kv))


def test_gqa_kv_heads_follow_attention_tp():
    q, h100 = load_model("qwen3-32b"), load_hardware("h100-sxm")
    tp4 = memory_report(q, h100, Deployment(Parallel(4), kv_dtype="fp8")).get("kv_per_token")
    dp4 = memory_report(q, h100, Deployment(Parallel(4, attn_dp=4), kv_dtype="fp8")
                        ).get("kv_per_token")
    assert dp4.value == 64 * 2 * 8 * 128 == 4 * tp4.value  # all 8 KV heads on one rank


def test_dp_lm_head_splits_over_attention_tp(glm, h20):
    def lm(par):
        return memory_report(glm, h20, Deployment(par)).get("gpu_weights_lm_head").value
    full = sum(t.nbytes for t in glm.inventory if t.component == "lm_head")
    assert lm(Parallel(8, attn_dp=8)) == full // 8
    assert lm(Parallel(8, attn_dp=8, dp_lm_head=True)) == full
    assert lm(Parallel(8, attn_dp=2, dp_lm_head=True)) == full // 4


@pytest.mark.parametrize("launch,attn_dp,dp", [
    ("--tp 8 --dp 8 --enable-dp-attention", 8, 1),  # deprecated spelling
    ("--tp 8 --attn-dp-size 4 --dp 2", 4, 2),  # replicas stay replicas
    ("--tp 8 --attention-data-parallel-size 2", 2, 1),
    ("--tp 8 --dp 2", 1, 2),
])
def test_launch_spellings(glm, h20, launch, attn_dp, dp):
    cfg = parse_sglang(shlex.split("sglang serve " + launch), glm, h20)
    assert (cfg.attn_dp, cfg.dp) == (attn_dp, dp)


def test_dp_lm_head_without_attention_dp_is_noted(glm, h20):
    cfg = parse_sglang(shlex.split("sglang serve --tp 8 --enable-dp-lm-head"), glm, h20)
    assert not cfg.dp_lm_head and any("no effect" in n for n in cfg.notes)


def test_unmodelled_layouts_rejected(glm, h20):
    with pytest.raises(SpecError, match="does not divide"):
        Deployment(Parallel(8, attn_dp=3)).validate(glm)
    with pytest.raises(SpecError, match="linear attention"):
        Deployment(Parallel(8, attn_dp=8)).validate(load_model("kimi-k3"))
    with pytest.raises(SpecError, match="not modelled yet"):
        bounds_report(glm, h20, Deployment(Parallel(8, attn_dp=8)), context=1000)


def test_cli_attention_dp(capsys):
    assert main(["memory", "--model", "glm-5.3", "--hw", "h20-3e", "--context", "131072",
                 "--", "sglang", "serve", "--tp", "8", "--dp", "8", "--enable-dp-attention",
                 "--kv-cache-dtype", "fp8_e4m3"]) == 0
    out = capsys.readouterr().out
    assert "TP8 PP1 EP1 attnDP8" in out and "attn_dp=8" in out
    assert main(["memory", "--model", "glm-5.3", "--hw", "h20-3e", "--attn-dp", "4", "--",
                 "sglang", "serve", "--tp", "8", "--attn-dp-size", "8"]) == 2
    assert "conflicts" in capsys.readouterr().err
