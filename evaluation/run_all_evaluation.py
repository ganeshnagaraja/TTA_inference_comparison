#!/usr/bin/env python3
"""
run_all_evaluation.py
=====================
Run evaluation across all model inference outputs.

Scans the inference_outputs directory for completed runs (those with a
manifest.json), evaluates each against the reference dataset, and produces
a consolidated comparison report.

Usage:
    python run_all_evaluation.py
    python run_all_evaluation.py --output-dir ./inference_outputs
    python run_all_evaluation.py --models audioldm tango tangoflux
    python run_all_evaluation.py --skip-fad --skip-fd   # faster eval
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_DIR = REPO_ROOT / "inference_outputs"


def find_manifests(output_dir: Path, models: Optional[List[str]] = None) -> List[Path]:
    """Find all manifest.json files under the output directory."""
    manifests = []
    if not output_dir.exists():
        return manifests

    for manifest_path in sorted(output_dir.rglob("manifest.json")):
        if models:
            # Filter by model name (parent directory)
            model_dir = manifest_path.parent.parent.name
            if model_dir not in models:
                continue
        manifests.append(manifest_path)

    return manifests


def run_single_evaluation(
    manifest_path: Path,
    skip_fad: bool = False,
    skip_fd: bool = False,
    skip_clap: bool = False,
    skip_kl: bool = False,
    skip_is: bool = False,
) -> Dict:
    """Run evaluation for a single manifest."""
    from evaluate import run_evaluation

    class Args:
        pass

    args = Args()
    args.manifest = str(manifest_path)
    args.generated = None
    args.reference = None
    args.csv = None
    args.output = str(manifest_path.parent / "evaluation_report.json")
    args.skip_fad = skip_fad
    args.skip_fd = skip_fd
    args.skip_clap = skip_clap
    args.skip_kl = skip_kl
    args.skip_is = skip_is

    try:
        report = run_evaluation(args)
        return {"status": "success", "report": report, "manifest": str(manifest_path)}
    except Exception as e:
        return {"status": "failed", "error": str(e), "manifest": str(manifest_path)}


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation across all model outputs"
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Root directory containing inference outputs")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Only evaluate specific models")
    parser.add_argument("--skip-fad", action="store_true")
    parser.add_argument("--skip-fd", action="store_true")
    parser.add_argument("--skip-clap", action="store_true")
    parser.add_argument("--skip-kl", action="store_true")
    parser.add_argument("--skip-is", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifests = find_manifests(output_dir, args.models)

    if not manifests:
        print(f"No manifest.json files found in {output_dir}")
        print("Run inference first: python inference_scripts/run_all_inference.py")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  Batch Evaluation — {len(manifests)} runs found")
    print(f"{'='*70}")
    for m in manifests:
        print(f"  {m.relative_to(output_dir)}")
    print()

    total_start = time.perf_counter()
    results = []

    for i, manifest_path in enumerate(manifests, 1):
        model_name = manifest_path.parent.parent.name
        run_tag = manifest_path.parent.name
        print(f"\n[{i}/{len(manifests)}] Evaluating: {model_name} / {run_tag}")
        print("-" * 60)

        result = run_single_evaluation(
            manifest_path,
            skip_fad=args.skip_fad,
            skip_fd=args.skip_fd,
            skip_clap=args.skip_clap,
            skip_kl=args.skip_kl,
            skip_is=args.skip_is,
        )
        result["model"] = model_name
        result["run_tag"] = run_tag
        results.append(result)

    total_time = time.perf_counter() - total_start

    # Print summary
    print(f"\n{'='*70}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'Run':<25} {'FAD':>8} {'FD':>8} {'CLAP':>8} {'KL':>8} {'IS':>8}")
    print(f"  {'-'*18} {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in results:
        if r["status"] != "success" or not r.get("report"):
            print(f"  {r['model']:<18} {r['run_tag']:<25} {'FAILED':>8}")
            continue
        metrics = r["report"].get("metrics", {})
        fad = metrics.get("FAD", {}).get("value", "—")
        fd = metrics.get("FD", {}).get("value", "—")
        clap = metrics.get("CLAP", {}).get("value", "—")
        kl = metrics.get("KL", {}).get("value", "—")
        is_val = metrics.get("IS", {}).get("value", "—")

        fmt = lambda v: f"{v:>8.4f}" if isinstance(v, (int, float)) else f"{v:>8}"
        print(f"  {r['model']:<18} {r['run_tag']:<25} {fmt(fad)} {fmt(fd)} {fmt(clap)} {fmt(kl)} {fmt(is_val)}")

    print(f"\n  Total evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save consolidated report
    consolidated = {
        "total_evaluation_time_s": round(total_time, 1),
        "num_runs_evaluated": len(results),
        "results": [],
    }
    for r in results:
        entry = {
            "model": r["model"],
            "run_tag": r["run_tag"],
            "status": r["status"],
        }
        if r["status"] == "success" and r.get("report"):
            entry["metrics"] = {
                k: v.get("value") for k, v in r["report"].get("metrics", {}).items()
                if "value" in v
            }
        consolidated["results"].append(entry)

    consolidated_path = output_dir / "evaluation_summary.json"
    with open(consolidated_path, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"  Consolidated report: {consolidated_path}")


if __name__ == "__main__":
    main()
