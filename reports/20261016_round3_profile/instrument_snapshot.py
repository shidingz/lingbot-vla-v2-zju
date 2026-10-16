#!/usr/bin/env python3
"""Add attribution-only ranges to an isolated training source snapshot.

No tensor operations, synchronization, optimizer settings, or model math change.
The caller owns the snapshot and stores the returned diff in its run manifest.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path


_METHOD = '''    def _instrument_round3_attribution(self):
        """Snapshot-only ranges; preserve function identity aliases and math."""
        if getattr(self, "_round3_scope_wrappers_installed", False):
            return
        self._round3_scope_wrappers_installed = True
        import sys

        targets = (
            ("moge.utils.geometry_torch", "recover_focal_shift", "teacher.geometry_recover"),
            ("moge.utils.geometry_numpy", "solve_optimal_focal_shift", "teacher.geometry_scipy"),
            ("vision_models.module_utils", "_rgb_to_unit_float", "teacher.rgb_normalize"),
            ("lingbotvla.optim.muon", "batched_newton_schulz", "optimizer.muon_newton_schulz"),
        )
        installed = []
        for suffix, attr, scope_name in targets:
            modules = list(sys.modules.items())
            originals = {}
            for module_name, module in modules:
                if module_name.endswith(suffix) and module is not None:
                    original = vars(module).get(attr)
                    if callable(original) and not getattr(original, "_round3_profile_scope", False):
                        originals[id(original)] = original
            for original in originals.values():
                @wraps(original)
                def wrapped(*args, _original=original, _scope_name=scope_name, **kwargs):
                    # Avoid even a nullcontext outside the selected capture window.
                    if not self.start_step - 1 <= self.current_step < self.end_step:
                        return _original(*args, **kwargs)
                    with self.scope(_scope_name):
                        return _original(*args, **kwargs)

                wrapped._round3_profile_scope = True
                alias_names = []
                for module_name, module in modules:
                    if module is not None and vars(module).get(attr) is original:
                        setattr(module, attr, wrapped)
                        alias_names.append(module_name + "." + attr)
                installed.append({"scope": scope_name, "aliases": alias_names})
        print("[round3_profile_scopes] " + repr(installed), flush=True)

'''


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one source anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def apply(snapshot: str | Path) -> dict:
    snapshot = Path(snapshot).resolve()
    production = Path(__file__).resolve().parents[2]
    if snapshot == production:
        raise ValueError("Refusing to instrument the production checkout")
    relative = (
        "lingbotvla/utils/training_profiler.py",
        "tasks/vla/train_lingbotvla.py",
    )
    before = {p: (snapshot / p).read_text() for p in relative}
    after = dict(before)

    profiler = after[relative[0]]
    profiler = _replace_once(
        profiler,
        "        # Wrapping forward also captures activation checkpoint recomputation.\n",
        "        self._instrument_round3_attribution()\n"
        "        # Wrapping forward also captures activation checkpoint recomputation.\n",
    )
    profiler = _replace_once(profiler, "    def start(self):\n", _METHOD + "    def start(self):\n")
    after[relative[0]] = profiler

    training = after[relative[1]]
    start = training.index("                total_loss += loss.item()\n")
    end = training.index("                del micro_batch\n", start)
    scalar_block = training[start:end]
    training = training[:start] + '                with training_profiler.scope("train.scalar_reads"):\n' + "".join("    " + line if line.strip() else line for line in scalar_block.splitlines(keepends=True)) + training[end:]
    training = _replace_once(
        training,
        "            torch.cuda.synchronize()\n            start_time = time.time()\n",
        '            with training_profiler.scope("train.pre_step_sync"):\n'
        "                torch.cuda.synchronize()\n            start_time = time.time()\n",
    )
    training = _replace_once(
        training,
        "            torch.cuda.synchronize()\n            delta_time = time.time() - start_time\n",
        '            with training_profiler.scope("train.post_step_sync"):\n'
        "                torch.cuda.synchronize()\n            delta_time = time.time() - start_time\n",
    )
    after[relative[1]] = training

    # Parse all candidates before writing either file.
    for name, content in after.items():
        ast.parse(content, filename=str(snapshot / name))
    edits = []
    for name in relative:
        (snapshot / name).write_text(after[name])
        edits.append({
            "path": name,
            "sha256_before": hashlib.sha256(before[name].encode()).hexdigest(),
            "sha256_after": hashlib.sha256(after[name].encode()).hexdigest(),
            "diff": "".join(difflib.unified_diff(before[name].splitlines(keepends=True), after[name].splitlines(keepends=True), fromfile="before/" + name, tofile="after/" + name)),
        })
    return {"snapshot": str(snapshot), "changes": edits, "additional_sync_calls": 0, "algorithm_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    result = apply(args.snapshot)
    if args.manifest:
        args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
