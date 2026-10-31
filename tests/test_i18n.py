"""Every report string has a Chinese translation; numbers survive translation."""

import os
import re
import shlex
import tempfile

import pytest
from test_engines import LAUNCH

from rooflinebound import i18n
from rooflinebound.bounds import bounds_report
from rooflinebound.calibrate import Point, deployment_key, save
from rooflinebound.capacity import Targets
from rooflinebound.cli import _SECTIONS, main
from rooflinebound.engines import parse_sglang
from rooflinebound.i18n import t
from rooflinebound.memory import Deployment, memory_report
from rooflinebound.parallel import Parallel
from rooflinebound.quantity import Quantity
from rooflinebound.spec import load_hardware, load_model


def _reports():
    m, hw = load_model("glm-5.3"), load_hardware("h20-3e")
    cfg = parse_sglang(shlex.split(LAUNCH + " --dp 2 --some-new-flag 1"), m, hw)
    dep = Deployment(Parallel(8), kv_layout=cfg.layout, page_size=64, max_running=32)
    res = Quantity.assumed(12e9, "B", "user:--reserve-gb")
    out = [
        memory_report(m, hw, Deployment(Parallel(8)), context=32768),
        memory_report(m, hw, Deployment(Parallel(4, 2)), context=32768, reserve=res),
        memory_report(m, hw, Deployment(Parallel(2)), context=1000),
        memory_report(m, hw, Deployment(Parallel(8, 1, 8), mtp=True), context=1000),
        memory_report(m, hw, Deployment(Parallel(8)), context=1000,
                      reserve=Quantity.assumed(200e9, "B", "user:--reserve-gb")),
        bounds_report(m, hw, Deployment(Parallel(8)), context=32768, batch=64, prompt=32768,
                      chunk=16384),
        bounds_report(m, hw, Deployment(Parallel(8), mtp=True), context=8192, batch=16,
                      draft=3),
        bounds_report(m, hw, Deployment(Parallel(8), mtp=True), context=8192, batch=16,
                      draft=2, accept=[0.8, 0.5]),
        bounds_report(m, hw, dep, context=2048, batch=40, prompt=8192, reserve=cfg.reserve,
                      targets=Targets(1024, 30, 1500, 10, 2.0)),
        bounds_report(m, hw, dep, prompt=8192, targets=Targets(1024, 15, 100)),
        bounds_report(m, hw, dep, prompt=8192, targets=Targets(1024, 0.0001)),
        bounds_report(m, hw, dep, prompt=8192, targets=Targets(1024, 1e6)),
        bounds_report(m, hw, dep, context=100, targets=Targets(1024)),
        bounds_report(m, hw, Deployment(Parallel(8), mtp=True), prompt=100, draft=1,
                      targets=Targets(10)),
        bounds_report(m, hw, Deployment(Parallel(8)), prompt=8192, strict=True),
    ]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        os.environ["ROOFLINEBOUND_CALIBRATIONS"] = path
        try:
            out.append(bounds_report(m, hw, dep, context=8192, prompt=8192, estimate=True))
            key = deployment_key(m.name, hw.name, dep.layout.name, 8, 1)
            save(key, Point("decode", 0.2, 10, 2, batch=1, context=8192))
            save(key, Point("prefill", 0.3, 10, 3, prompt=8192, chunk=16384))
            out.append(bounds_report(m, hw, dep, context=8192, batch=8, prompt=4096,
                                     estimate=True))
        finally:
            del os.environ["ROOFLINEBOUND_CALIBRATIONS"]
    return out, cfg


@pytest.fixture(scope="module")
def strings():
    reports, cfg = _reports()
    found = set(_SECTIONS.values()) | {"Not computed", "Notes", "Inputs", "on"}
    for r in reports:
        for items in r.items.values():
            for it in items:
                found |= {s for s in (it.label, it.note) if s}
        found |= set(r.missing) | set(r.notes)
    found |= {f"launch: {n}" for n in cfg.notes}
    found.add("launch flags with no effect on these figures: " + " ".join(cfg.ignored))
    found.add("launch flags not recognised (check they do not change memory or speed): "
              + " ".join(cfg.unrecognised))
    return found


def test_every_string_translates(strings):
    i18n.UNTRANSLATED.clear()
    for s in strings:
        t(s, "zh")
    assert not i18n.UNTRANSLATED, sorted(i18n.UNTRANSLATED)


def test_numbers_and_quantities_survive(strings):
    for s in strings:
        zh = t(s, "zh")
        for q in re.findall(r"[≥≤] \S+(?: \S+)? \[[^\]]+\]", s):
            assert q in zh, (s, zh)
        for n in re.findall(r"\d+(?:[,.]\d+)*", s):
            assert n in zh, (n, s, zh)


def test_english_passes_through():
    assert t("TPOT, batch 1", "en") == "TPOT, batch 1"
    assert t(None, "zh") is None
    with pytest.raises(ValueError):
        t("x", "fr")


def test_cli_zh_report(capsys):
    assert main(["memory", "--lang", "zh", "--model", "glm-5.3", "--hw", "h20-3e",
                 "--context", "32768", "--", *shlex.split(LAUNCH)]) == 0
    out = capsys.readouterr().out
    assert "每卡权重" in out and "53,940 B/token [EXACT]" in out and "说明" in out


def test_json_stays_english(capsys):
    assert main(["memory", "--lang", "zh", "--json", "--model", "glm-5.3", "--hw", "h20-3e",
                 "--tp", "8"]) == 0
    assert "weights per GPU" in capsys.readouterr().out
