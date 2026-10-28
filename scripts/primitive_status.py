"""Read-only status of the independent primitive training jobs and acceptance results."""
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone

root = Path(__file__).resolve().parents[1]
folder = root / "outputs/primitive_cycle"
jobs = json.loads((folder / "jobs.json").read_text()) if (folder / "jobs.json").exists() else {}
result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "jobs": {}, "evaluations": {}}
for name, job in jobs.items():
    pid = job["pid"]
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        alive = any(script in cmdline for script in ("train_primitive.py", "eval_primitive.py", "collect_tactile_data.py", "train_blind_ppo.py", "train_sequence_detector.py", "train_blind_actor.py"))
    except (OSError, ProcessLookupError):
        alive, cmdline = False, ""
    log = root / "logs" / f"{name}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    iterations = re.findall(r"Learning iteration ([0-9]+)/([0-9]+)", text)
    recurrent_rows = [line.split('[blind_ppo] ',1)[1] for line in text.splitlines() if line.startswith('[blind_ppo] {')]
    recurrent_latest = json.loads(recurrent_rows[-1]) if recurrent_rows else None
    result["jobs"][name] = {"pid": pid, "alive": alive,
        "iteration": list(map(int, iterations[-1])) if iterations else None,
        "recurrent_training_latest": recurrent_latest,
        "traceback": "Traceback (most recent call last)" in text,
        "finished_marker": "[cycle_train] finished" in text or "[eval] report" in text or "[collect] wrote" in text or "[sequence_detector] finished" in text or "[blind_ppo] finished" in text,
        "log_modified_utc": datetime.fromtimestamp(log.stat().st_mtime, timezone.utc).isoformat() if log.exists() else None}
for path in sorted(folder.glob("*.json")):
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if "summary" in report and "episodes" in report:
        result["evaluations"][path.name] = {"mode": report.get("mode"), "sequence": report.get("sequence"),
            "scripted_return_probe": report.get("scripted_return_probe", False), "summary": report["summary"],
            "trigger":report.get("trigger"),"actor_scope":report.get("actor_observation_scope"),
            "actor_scope_up":report.get("actor_observation_scope_up"),"seed":report.get("seed"),
            "initial_settle_steps":report.get("initial_settle_steps",0),
            "contract_version":report.get("config",{}).get("cycle_contract_version"),
            "grasp_cache":report.get("config",{}).get("grasp_cache_path"),
            "per_trial":[{"trial":trial,"n":sum(e['trial']==trial for e in report['episodes']),
                "successes":sum(e['trial']==trial and e['success'] for e in report['episodes'])}
                for trial in sorted({e['trial'] for e in report['episodes']})]}
print(json.dumps(result, indent=2))
