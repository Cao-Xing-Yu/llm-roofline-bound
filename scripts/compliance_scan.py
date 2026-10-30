#!/usr/bin/env python3
"""Fail if tracked files or commit metadata contain things that must not be published.

Generic checks (private network addresses, credentials, key material) live
here. Organisation-specific terms (company names, internal hosts, colleague
names) must not be written into a public repository, even as scan patterns, so
they are read from ``.compliance-patterns.local`` (git-ignored, one regex per
line, ``#`` comments) or the ``COMPLIANCE_PATTERNS_FILE`` environment variable.

Usage: python scripts/compliance_scan.py [--history]
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GENERIC = {
    "private IPv4 (10/8)": r"\b10\.(?:\d{1,3}\.){2}\d{1,3}\b",
    "private IPv4 (172.16/12)": r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b",
    "private IPv4 (192.168/16)": r"\b192\.168\.\d{1,3}\.\d{1,3}\b",
    "credential assignment": (
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"][^'\"\s]{4,}"
    ),
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "OpenAI-style key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{30,}",
    "GitHub token": r"\bgh[pousr]_[A-Za-z0-9]{30,}",
    "AWS access key": r"\bAKIA[0-9A-Z]{16}\b",
}

# This file defines the patterns above, so it would match itself.
SKIP = {"scripts/compliance_scan.py", "LICENSE"}


def local_patterns() -> dict:
    path = Path(os.environ.get("COMPLIANCE_PATTERNS_FILE", ROOT / ".compliance-patterns.local"))
    if not path.exists():
        return {}
    out = {}
    for i, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        line = line.strip()
        if line and not line.startswith("#"):
            out[f"local pattern #{i}"] = line
    return out


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, encoding="utf-8").stdout


def main() -> int:
    patterns = {k: re.compile(v) for k, v in {**GENERIC, **local_patterns()}.items()}
    hits = []

    for rel in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines():
        if rel in SKIP:
            continue
        try:
            text = (ROOT / rel).read_text("utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for name, rx in patterns.items():
                if rx.search(line):
                    hits.append(f"{rel}:{n}: {name}")

    if "--history" in sys.argv:
        try:
            log = git("log", "--all", "--format=%H %an <%ae> | %cn <%ce> | %B")
        except subprocess.CalledProcessError:
            log = ""  # no commits yet
        for name, rx in patterns.items():
            for line in log.splitlines():
                if rx.search(line):
                    hits.append(f"git history: {name}: {line[:12]}")
        remotes = git("remote", "-v")
        for name, rx in patterns.items():
            if rx.search(remotes):
                hits.append(f"git remote: {name}")

    if not local_patterns():
        print("note: no .compliance-patterns.local; only generic checks ran", file=sys.stderr)
    for h in hits:
        print(h)
    print(f"{len(hits)} finding(s)", file=sys.stderr)
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
