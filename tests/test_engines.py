"""KV layouts and SGLang launch commands."""

import shlex

import pytest

from rooflinebound.bounds import bounds_report
from rooflinebound.cli import main
from rooflinebound.engines import parse_sglang
from rooflinebound.kvlayout import sglang_dsa, uniform
from rooflinebound.memory import Deployment, memory_report
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Level
from rooflinebound.spec import SpecError, load_hardware, load_model

# A typical GLM-5.3 launch on one 8-GPU node.
LAUNCH = (
    "sglang serve --model-path /models --served-model-name z-ai/glm-5.3 --tp 8 "
    "--context-length 393216 --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton_sparse_mla "
    "--dsa-decode-backend flashmla_kv --mem-fraction-static 0.92 --chunked-prefill-size 16384 "
    "--host 127.0.0.1 --port 8000 --cuda-graph-max-bs-decode 32 --max-running-requests 32 "
    "--reasoning-parser glm45 --tool-call-parser glm47 --enable-metrics --enable-cache-report "
    "--preferred-sampling-params '{\"max_new_tokens\":131072}' --enable-hierarchical-cache "
    "--hicache-size 100 --hicache-write-policy write_through --enable-strict-thinking "
    "--max-queued-requests 32"
)


@pytest.fixture(scope="module")
def glm():
    return load_model("glm-5.3")


@pytest.fixture(scope="module")
def h20():
    return load_hardware("h20-3e")


# --------------------------------------------------------------- layouts


def test_sglang_fp8_layout_bytes(glm):
    layer = glm.layers[0]
    lay = sglang_dsa("fp8_e4m3")
    # 512 fp8 latent + 512/128 fp32 scales + 64 bf16 rope
    assert lay.main_bytes(layer.attn) == 512 + 4 * 4 + 64 * 2 == 656
    # 128 fp8 index key + one fp32 scale
    assert lay.index_key_bytes(layer.indexer) == 128 + 4 == 132
    assert sglang_dsa("fp8_e4m3", scaled=False).main_bytes(layer.attn) == 576
    assert sglang_dsa("auto").main_bytes(layer.attn) == 576 * 2
    assert sglang_dsa("auto").index_key_bytes(layer.indexer) == 132  # index key: always fp8


def test_glm_kv_per_token_sglang_fp8(glm, h20):
    dep = Deployment(Parallel(tp=8), kv_layout=sglang_dsa("fp8_e4m3"), page_size=64)
    r = memory_report(glm, h20, dep)
    assert r.get("kv_per_token").value == 78 * 656 + 21 * 132 == 53_940


def test_uniform_layout_is_never_larger(glm):
    attn, ix = glm.layers[0].attn, glm.layers[0].indexer
    u, s = uniform("fp8", "fp8"), sglang_dsa("fp8_e4m3")
    assert u.main_bytes(attn) <= s.main_bytes(attn)
    assert u.index_key_bytes(ix) <= s.index_key_bytes(ix)


# ---------------------------------------------------------------- parsing


def test_parse_maintainer_launch(glm, h20):
    cfg = parse_sglang(shlex.split(LAUNCH), glm, h20)
    assert (cfg.tp, cfg.ep, cfg.pp, cfg.dp) == (8, 1, 1, 1)
    assert cfg.layout == sglang_dsa("fp8_e4m3")
    assert (cfg.chunk, cfg.page_size, cfg.max_running) == (16384, 64, 32)
    assert cfg.context_length == 393216 and cfg.draft == 0
    assert cfg.reserve.level == Level.ASSUMED
    assert cfg.reserve.value == pytest.approx(0.08 * 143771 * 2**20)
    assert "sglang:--mem-fraction-static" in {p.ref for p in cfg.reserve.provenance}
    assert any("hierarchical" in n for n in cfg.notes)
    assert any("CUDA graphs" in n for n in cfg.notes)
    assert {"--host", "--port", "--reasoning-parser"} <= set(cfg.ignored)
    assert cfg.unrecognised == []


@pytest.mark.parametrize("extra,match", [
    ("--enable-dp-attention", "data-parallel attention"),
    ("--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 4",
     "tree speculation"),
    ("--quantization w4afp8", "quantization"),
    ("--page-size 1", "64-token pages"),
    ("--kv-cache-dtype fp8_e5m2", "not modelled"),
])
def test_unmodelled_launches_rejected(glm, h20, extra, match):
    with pytest.raises(SpecError, match=match):
        parse_sglang(shlex.split(f"sglang serve --tp 8 {extra}"), glm, h20)


def test_trtllm_backend_uses_raw_layout(glm, h20):
    cfg = parse_sglang(shlex.split("python -m sglang.launch_server --tp-size=8 "
                                   "--kv-cache-dtype fp8_e4m3 --dsa-decode-backend trtllm"),
                       glm, h20)
    assert cfg.tp == 8 and cfg.layout.main_bytes(glm.layers[0].attn) == 576


def test_speculative_nextn_maps_to_mtp(glm, h20):
    cfg = parse_sglang(shlex.split("sglang serve --tp 8 --speculative-algorithm NEXTN "
                                   "--speculative-num-steps 3 --speculative-eagle-topk 1 "
                                   "--speculative-num-draft-tokens 4"), glm, h20)
    assert cfg.draft == 3


def test_unknown_flags_are_listed_not_dropped(glm, h20):
    cfg = parse_sglang(shlex.split("sglang serve --tp 8 --some-new-flag 7 --enable-thing"),
                       glm, h20)
    assert cfg.unrecognised == ["--some-new-flag", "--enable-thing"]


def test_not_an_sglang_command(glm, h20):
    with pytest.raises(SpecError, match="not an SGLang"):
        parse_sglang(shlex.split("vllm serve --tp 8"), glm, h20)


# -------------------------------------------------------------------- CLI


def test_cli_memory_with_launch(capsys):
    assert main(["memory", "--model", "glm-5.3", "--hw", "h20-3e", "--context", "32768",
                 "--", *shlex.split(LAUNCH)]) == 0
    out = capsys.readouterr().out
    assert "engine=sglang" in out and "53,940 B/token [EXACT]" in out
    assert "[ASSUMED]" in out  # pool under the memory fraction
    assert "no effect on these figures" in out


def test_cli_conflicting_flag_rejected(capsys):
    assert main(["memory", "--model", "glm-5.3", "--hw", "h20-3e", "--tp", "4",
                 "--", *shlex.split(LAUNCH)]) == 2
    assert "conflicts with the launch command" in capsys.readouterr().err


def test_cli_context_beyond_server_limit(capsys):
    assert main(["bounds", "--model", "glm-5.3", "--hw", "h20-3e", "--context", "400000",
                 "--", *shlex.split(LAUNCH)]) == 2
    assert "--context-length" in capsys.readouterr().err


def test_max_running_caps_throughput(glm, h20):
    dep = Deployment(Parallel(tp=8), kv_layout=sglang_dsa("fp8_e4m3"), page_size=64,
                     max_running=4)
    r = bounds_report(glm, h20, dep, context=4096, compare=False)
    note = next(i.note for i in r.items["decode"] if i.key == "tps_max")
    assert note.startswith("at batch 4, the engine's max running requests")
    assert r.get("tps_max").level == Level.BOUND  # the cap is exact; not a memory bound
