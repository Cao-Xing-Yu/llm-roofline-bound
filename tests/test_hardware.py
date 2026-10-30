import json

import pytest

from rooflinebound.cli import main
from rooflinebound.quantity import Level, Trust
from rooflinebound.spec import SpecError, list_hardware, load_hardware
from rooflinebound.spec.hardware import parse_hardware


def _spec(**overrides):
    fields = {
        "mem_capacity": {"value": 80, "unit": "GB", "trust": "OFFICIAL",
                         "source": "https://example.com/a"},
        "mem_bw": {"value": 3350, "unit": "GB/s", "trust": "OFFICIAL",
                   "source": "https://example.com/a"},
        "tflops_bf16": {"value": 989, "unit": "TFLOP/s", "trust": "OFFICIAL",
                        "source": "https://example.com/a"},
        "link_bw": {"value": 900, "unit": "GB/s", "trust": "USER"},
        "devices_per_node": {"value": 8, "unit": "device", "trust": "USER"},
    }
    fields.update(overrides)
    return {"name": "toy", "fields": {k: v for k, v in fields.items() if v is not None}}


def test_bundled_h20_3e_loads_in_base_units():
    spec = load_hardware("h20-3e")
    assert "h20-3e" in list_hardware()
    assert spec["mem_bw"].value == 4000e9 and str(spec["mem_bw"].unit) == "B/s"
    assert spec["tflops_fp8"].value == 296e12
    assert spec["mem_bw"].level == Level.EXACT
    assert spec["mem_bw"].trust == Trust.THIRD_PARTY
    assert spec.disputed_fields == ["mem_bw"]


def test_peak_flops_falls_back_to_wider_dtype():
    spec = load_hardware("h20-3e")
    assert spec.peak_flops("fp4").value == 296e12  # no fp4 unit -> fp8
    assert spec.peak_flops("bf16").value == 148e12


def test_unsourced_official_value_rejected():
    raw = _spec(mem_bw={"value": 3350, "unit": "GB/s", "trust": "OFFICIAL"})
    with pytest.raises(SpecError, match="source"):
        parse_hardware(raw)


@pytest.mark.parametrize("override,match", [
    ({"mem_bw": {"value": 3.35, "unit": "TB/s", "trust": "USER"}}, "unit"),
    ({"mem_bw": {"value": -1, "unit": "GB/s", "trust": "USER"}}, "positive"),
    ({"mem_bw": {"value": 1, "unit": "GB/s", "trust": "GUESS"}}, "trust"),
    ({"mem_bw": None}, "missing"),
    ({"hbm_temp": {"value": 1, "unit": "C", "trust": "USER"}}, "unknown"),
])
def test_bad_specs_rejected(override, match):
    with pytest.raises(SpecError, match=match):
        parse_hardware(_spec(**override))


def test_cli_hw_text_flags_trust(capsys):
    assert main(["hw", "h20-3e"]) == 0
    out = capsys.readouterr().out
    assert "4,000 GB/s [EXACT]" in out
    assert "THIRD_PARTY" in out and "DISPUTED" in out
    assert "conditional" in out


def test_cli_hw_json(capsys):
    assert main(["hw", "h20-3e", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    bw = data["fields"]["mem_bw"]
    assert bw["disputed"] is True
    assert bw["provenance"][0]["trust"] == "THIRD_PARTY"


def test_cli_unknown_hw(capsys):
    assert main(["hw", "nope"]) == 2
    assert "unknown hardware" in capsys.readouterr().err
