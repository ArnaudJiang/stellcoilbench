#!/usr/bin/env python3
"""
Run autopilot cases from cases/pending/*.json in parallel with per-case timeouts.

Launches stellcoilbench run-ci-case for each pending case, enforces per-case
timeouts, polls for progress, and prints log summaries. Writes status to
/tmp/autopilot_logs/_status.txt: [OK] case_id, [FAIL] case_id, [TIMEOUT] case_id.

Usage:
  python -m tools.ci_autopilot_runner

Expects: conda env activated, stellcoilbench installed, run from repo root.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

LOG_DIR = Path("/tmp/autopilot_logs")
PENDING_DIR = Path("cases/pending")
DONE_DIR = Path("cases/done")
STATUS_FILE = LOG_DIR / "_status.txt"
MAX_CASES = 10
POLL_INTERVAL = 60
DEFAULT_TIMEOUT_MIN = 60


def _timeout_min(case_path: Path) -> int:
    try:
        d = json.loads(case_path.read_text())
        return int(d.get("resource", {}).get("timeout_minutes", DEFAULT_TIMEOUT_MIN))
    except Exception:
        return DEFAULT_TIMEOUT_MIN


def _summary(case_id: str) -> str:
    p = DONE_DIR / case_id / "summary.json"
    if not p.exists():
        return "(no summary.json)"
    try:
        d = json.loads(p.read_text())
        if d.get("success"):
            return f"OK  score={d['total_score']:.4e}  iters={d['iterations_used']}  wall={d['walltime_sec']:.0f}s"
        return f"FAIL  {d.get('failure_class', '?')}: {(d.get('failure_reason') or '?')[:80]}"
    except Exception:
        return "summary read failed"


def _launch(case_path: Path, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["STELLCOILBENCH_CI_VERBOSE_STDOUT"] = "1"
    cmd = shutil.which("stellcoilbench") or str(
        Path(sys.executable).parent / ("stellcoilbench.exe" if sys.platform == "win32" else "stellcoilbench")
    )
    with open(log_path, "w") as f:
        return subprocess.Popen(
            [
                cmd,
                "run-ci-case",
                str(case_path),
                "--output-dir",
                str(DONE_DIR),
                "--policy",
                "policy/proposer_policy.yaml",
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )


def _status_append(status: str, case_id: str, extra: str = "") -> None:
    with open(STATUS_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{status}]  {case_id}{' ' + extra if extra else ''}\n")


def _dump_log(log_path: Path, tail: int | None = None) -> None:
    if not log_path.exists():
        print("(no log)")
        return
    lines = log_path.read_text().splitlines()
    for ln in lines[-tail:] if tail else lines:
        print(ln)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text("")
    pending = sorted(PENDING_DIR.glob("*.json"))[:MAX_CASES]
    if not pending:
        return 0

    procs: list[tuple[subprocess.Popen, Path, Path, int]] = []
    for cp in pending:
        cid, to_min = cp.stem, _timeout_min(cp)
        log_path = LOG_DIR / f"{cid}.log"
        print(f"Launching autopilot case: {cid} (timeout: {to_min}m)")
        procs.append((_launch(cp, log_path), cp, log_path, to_min * 60))

    print(f"Waiting for {len(procs)} autopilot case(s)...")
    start, timed_out = time.monotonic(), set()

    while True:
        elapsed = int(time.monotonic() - start)
        done = True
        for proc, cp, _, to_sec in procs:
            if proc.poll() is None:
                if elapsed > to_sec:
                    print(
                        f"TIMEOUT: killing {cp.stem} (pid {proc.pid}) after {elapsed}s"
                    )
                    proc.terminate()
                    time.sleep(5)
                    proc.kill()
                    _status_append("TIMEOUT", cp.stem, f"(killed after {elapsed}s)")
                    timed_out.add(cp.stem)
                    cp.unlink(missing_ok=True)
                else:
                    done = False

        if done:
            break

        print(f"--- {time.strftime('%H:%M:%S')} elapsed={elapsed}s ---")
        running = completed = 0
        for proc, cp, lp, to_sec in procs:
            cid = cp.stem
            if proc.poll() is None:
                running += 1
                sz = lp.stat().st_size if lp.exists() else 0
                print(f"  [running] {cid}  log={sz}B  timeout={to_sec}s")
                if lp.exists() and lp.stat().st_size > 0:
                    for ln in [
                        x.strip() for x in lp.read_text().splitlines() if x.strip()
                    ][-5:]:
                        print(f"            {ln[:140]}")
            else:
                completed += 1
                print(f"  [done]    {cid}  {_summary(cid)}")
        print(f"  === {running} running, {completed} completed of {len(procs)} ===")
        time.sleep(POLL_INTERVAL)

    for proc, cp, _, _ in procs:
        cid = cp.stem
        if cid not in timed_out and proc.poll() is not None:
            rc = proc.returncode
            _status_append(
                "OK" if rc == 0 else "FAIL", cid, "" if rc == 0 else f"(exit {rc})"
            )
            cp.unlink(missing_ok=True)
        proc.wait()

    print("=== Autopilot status ===")
    status_lines = STATUS_FILE.read_text().splitlines() if STATUS_FILE.exists() else []
    print("\n".join(status_lines) or "(no status)")

    # Parse status: [STATUS] case_id ...
    entries = [
        (ln.split()[0].strip("[]"), ln.split()[1])
        for ln in status_lines
        if len(ln.split()) >= 2
    ]

    for section, filter_status, tail in [
        ("Failed/Timed-out case logs", ("FAIL", "TIMEOUT"), None),
        ("Successful case log tails", ("OK",), 20),
    ]:
        print(f"=== {section} ===")
        for st, cid in entries:
            if st in filter_status:
                print(
                    f"--- {'Full log' if tail is None else 'Last 20 lines'}: {cid} ---"
                )
                _dump_log(LOG_DIR / f"{cid}.log", tail)
                print(f"--- End: {cid} ---")

    return 0


if __name__ == "__main__":
    sys.exit(main())
