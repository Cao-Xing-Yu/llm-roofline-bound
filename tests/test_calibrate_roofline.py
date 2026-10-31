"""Calibration, estimates and the roofline chart."""

import re

import pytest
from test_memory import hw, toy

from rooflinebound.bounds import bounds_report
from rooflinebound.calibrate import Point, deployment_key, efficiency, find, save
from rooflinebound.cli import main
from rooflinebound.memory import Deployment
from rooflinebound.quantity import Level
from rooflinebound.roofline import roofline_html
from rooflinebound.spec import SpecError


@pytest.fixture
def calib(tmp_path, monkeypatch):
    path = tmp_path / "calibrations.json"
    monkeypatch.setenv("ROOFLINEBOUND_CALIBRATIONS", str(path))
    return path


def test_efficiency_rejects_beating_the_bound():
    assert efficiency(1.0, 4.0) == 0.25
    with pytest.raises(SpecError, match="below the lower bound"):
        efficiency(2.0, 1.0)


def test_save_replaces_same_conditions_and_find_nearest(calib):
    key = "k"
    save(key, Point("decode", 0.1, 10, 1, batch=1, context=1000), calib)
    save(key, Point("decode", 0.2, 10, 2, batch=1, context=1000), calib)  # replaces
    save(key, Point("decode", 0.5, 10, 5, batch=32, context=1000), calib)
    assert find(key, "decode", batch=2, context=1000, path=calib).efficiency == 0.2
    assert find(key, "decode", batch=24, context=1000, path=calib).efficiency == 0.5
    assert find(key, "prefill", prompt=100, path=calib) is None
    assert find("other", "decode", batch=1, context=1, path=calib) is None


def test_estimates_are_labelled_and_scaled(calib):
    m, spec, dep = toy(), hw(64 * 2**20), Deployment()
    key = deployment_key(m.name, spec.name, dep.layout.name, 1, 1)
    base = bounds_report(m, spec, dep, context=100, prompt=64, compare=False)
    save(key, Point("decode", 0.5, 1, 1, batch=1, context=100), calib)
    r = bounds_report(m, spec, dep, context=100, prompt=64, compare=False, estimate=True)
    est = r.get("est_tpot_b1")
    assert est.level == Level.ESTIMATE and est.direction is None
    assert est.value == pytest.approx(base.get("tpot_b1").value / 0.5)
    assert r.get("est_tps_seq_b1").value == pytest.approx(base.get("tps_seq_b1").value * 0.5)
    assert any(p.ref.startswith("calib:decode") for p in est.provenance)
    assert any("prefill estimates: no calibration" in x for x in r.missing)


def test_estimate_names_differing_conditions(calib):
    m, spec, dep = toy(), hw(64 * 2**20), Deployment()
    key = deployment_key(m.name, spec.name, dep.layout.name, 1, 1)
    save(key, Point("decode", 0.5, 1, 1, batch=1, context=100), calib)
    r = bounds_report(m, spec, dep, context=500, batch=8, compare=False, estimate=True)
    note = next(i.note for i in r.items["estimate"] if i.key == "est_tpot_batch")
    assert "measured at batch 1, context 100" in note and "applied at batch 8" in note


def test_cli_calibrate_round_trip(calib, capsys):
    args = ["--model", "glm-5.3", "--hw", "h20-3e", "--tp", "8"]
    assert main(["calibrate", *args, "--measured-tpot-ms", "0.5", "--batch", "1",
                 "--context", "8192", "--save"]) == 2
    assert "below the lower bound" in capsys.readouterr().err and not calib.exists()
    assert main(["calibrate", *args, "--measured-tpot-ms", "20", "--batch", "1",
                 "--context", "8192", "--save"]) == 0
    assert calib.exists()
    capsys.readouterr()
    assert main(["bounds", *args, "--context", "8192", "--estimate"]) == 0
    assert "[ESTIMATE]" in capsys.readouterr().out


def test_roofline_html_is_self_contained():
    m, spec = toy(), hw(64 * 2**20)
    r = bounds_report(m, spec, Deployment(), context=100, batch=4, prompt=64, compare=False)
    page = roofline_html(m.name, spec, r.tables, r.notes)
    assert page.startswith("<!doctype html>") and page.count("<svg") >= len(r.tables)
    assert not re.search(r'(src|href)="https?://', page)  # no external resources
    assert "<table>" in page and "prefers-color-scheme: dark" in page
    zh = roofline_html(m.name, spec, r.tables, r.notes, lang="zh")
    assert "全部算子" in zh


def test_cli_roofline_out(tmp_path):
    out = tmp_path / "chart.html"
    assert main(["bounds", "--model", "glm-5.3", "--hw", "h20-3e", "--tp", "8",
                 "--context", "4096", "--roofline-out", str(out)]) == 0
    assert out.read_text("utf-8").count('class="pt"') > 5
