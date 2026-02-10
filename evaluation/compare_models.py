#!/usr/bin/env python3
"""
compare_models.py
=================
Cross-model comparison and ranking from evaluation reports.

Reads evaluation_report.json files produced by evaluate.py, aggregates
metrics across models, ranks models per metric, and produces:
  1. A comparison table (printed and saved as CSV)
  2. An overall ranking using normalized scores
  3. Category-wise breakdown (by Num Layers / duration bins)

Usage:
    python compare_models.py --output-dir ./inference_outputs
    python compare_models.py --output-dir ./inference_outputs --export comparison.csv
    python compare_models.py --output-dir ./inference_outputs --rank
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_OUTPUT_DIR = REPO_ROOT / "inference_outputs"
DEFAULT_CSV = REPO_ROOT / "Dataset" / "Inference_Dataset" / "metadata.csv"

# Metric direction: True = higher is better, False = lower is better
METRIC_DIRECTION = {
    "FAD": False,
    "FD": False,
    "CLAP": True,
    "KL": False,
    "IS": True,
    "inference_time": False,
}

# Expected ranges for normalization (from literature / thesis)
METRIC_RANGES = {
    "FAD": (0.0, 30.0),
    "FD": (0.0, 60.0),
    "CLAP": (0.0, 1.0),
    "KL": (0.0, 5.0),
    "IS": (1.0, 15.0),
}


def load_evaluation_reports(
    output_dir: Path,
    models: Optional[List[str]] = None,
) -> List[Dict]:
    """Load all evaluation_report.json files from inference output dirs."""
    reports = []
    if not output_dir.exists():
        return reports

    for report_path in sorted(output_dir.rglob("evaluation_report.json")):
        model_name = report_path.parent.parent.name
        run_tag = report_path.parent.name

        if models and model_name not in models:
            continue

        with open(report_path) as f:
            report = json.load(f)

        reports.append({
            "model": model_name,
            "run_tag": run_tag,
            "path": str(report_path),
            "report": report,
        })

    return reports


def extract_metrics_table(reports: List[Dict]) -> List[Dict]:
    """Extract a flat table of model → metric values."""
    rows = []
    for entry in reports:
        metrics = entry["report"].get("metrics", {})
        row = {
            "model": entry["model"],
            "run_tag": entry["run_tag"],
        }
        for metric_name in METRIC_DIRECTION:
            m = metrics.get(metric_name, {})
            if "value" in m:
                row[metric_name] = m["value"]
            elif "avg_per_sample_s" in m:
                row[metric_name] = m["avg_per_sample_s"]
            else:
                row[metric_name] = None
        rows.append(row)
    return rows


def normalize_score(value: float, metric: str) -> float:
    """Normalize metric to 0–1 where 1 is always best."""
    if value is None:
        return 0.0
    lo, hi = METRIC_RANGES.get(metric, (0.0, 1.0))
    clamped = max(lo, min(hi, value))
    normalized = (clamped - lo) / (hi - lo) if hi > lo else 0.5
    # Flip for lower-is-better metrics
    if not METRIC_DIRECTION.get(metric, True):
        normalized = 1.0 - normalized
    return round(normalized, 4)


def rank_models(rows: List[Dict]) -> List[Dict]:
    """Compute overall ranking using normalized scores."""
    for row in rows:
        scores = []
        for metric in ["FAD", "FD", "CLAP", "KL", "IS"]:
            if row.get(metric) is not None:
                scores.append(normalize_score(row[metric], metric))
        row["overall_score"] = round(sum(scores) / len(scores), 4) if scores else 0.0

    rows.sort(key=lambda r: r["overall_score"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    return rows


def print_comparison_table(rows: List[Dict]) -> None:
    """Print a formatted comparison table."""
    header = f"  {'Rank':>4} {'Model':<18} {'Run':<25} {'FAD':>8} {'FD':>8} {'CLAP':>8} {'KL':>8} {'IS':>8} {'Score':>8}"
    divider = f"  {'—'*4} {'—'*18} {'—'*25} {'—'*8} {'—'*8} {'—'*8} {'—'*8} {'—'*8} {'—'*8}"

    print(header)
    print(divider)

    for row in rows:
        fmt = lambda v: f"{v:>8.4f}" if isinstance(v, (int, float)) else f"{'—':>8}"
        print(
            f"  {row.get('rank', ''):>4} "
            f"{row['model']:<18} "
            f"{row['run_tag']:<25} "
            f"{fmt(row.get('FAD'))} "
            f"{fmt(row.get('FD'))} "
            f"{fmt(row.get('CLAP'))} "
            f"{fmt(row.get('KL'))} "
            f"{fmt(row.get('IS'))} "
            f"{fmt(row.get('overall_score'))}"
        )


def export_csv(rows: List[Dict], output_path: Path) -> None:
    """Export comparison table as CSV."""
    fieldnames = ["rank", "model", "run_tag", "FAD", "FD", "CLAP", "KL", "IS",
                  "inference_time", "overall_score"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Exported: {output_path}")


def load_metadata_categories(csv_path: str) -> Dict[str, Dict]:
    """Load metadata for category-wise analysis."""
    records = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records[row["Filename"].strip()] = {
                "num_layers": int(row["Num Layers"]),
                "duration": float(row["Duration (s)"]),
            }
    return records


def category_analysis(reports: List[Dict], csv_path: str) -> Dict:
    """Break down per-sample metrics by category (num_layers, duration bins)."""
    try:
        metadata = load_metadata_categories(csv_path)
    except FileNotFoundError:
        print(f"  Metadata CSV not found at {csv_path}, skipping category analysis.")
        return {}

    analysis = {}
    for entry in reports:
        model = entry["model"]
        metrics = entry["report"].get("metrics", {})

        # Per-sample CLAP scores
        clap_data = metrics.get("CLAP", {})
        clap_per_sample = clap_data.get("per_sample", [])

        # Per-sample KL values
        kl_data = metrics.get("KL", {})
        kl_per_sample = kl_data.get("per_sample", [])

        # We need the manifest to map sample indices to filenames
        manifest_path = Path(entry["path"]).parent / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path) as f:
            manifest = json.load(f)

        results_list = manifest.get("results", [])

        # Bin by num_layers
        layers_clap = {}
        layers_kl = {}
        for idx, result in enumerate(results_list):
            orig = result.get("original_filename", "")
            meta = metadata.get(orig, {})
            nl = meta.get("num_layers", 0)

            if idx < len(clap_per_sample):
                layers_clap.setdefault(nl, []).append(clap_per_sample[idx])
            if idx < len(kl_per_sample):
                layers_kl.setdefault(nl, []).append(kl_per_sample[idx])

        import numpy as np
        model_analysis = {"by_num_layers": {}}
        for nl in sorted(layers_clap.keys()):
            model_analysis["by_num_layers"][f"{nl}_layers"] = {
                "count": len(layers_clap.get(nl, [])),
                "avg_clap": round(float(np.mean(layers_clap.get(nl, [0]))), 4),
                "avg_kl": round(float(np.mean(layers_kl.get(nl, [0]))), 4),
            }

        # Bin by duration
        dur_clap = {"short": [], "medium": [], "long": []}
        dur_kl = {"short": [], "medium": [], "long": []}
        for idx, result in enumerate(results_list):
            orig = result.get("original_filename", "")
            meta = metadata.get(orig, {})
            dur = meta.get("duration", 10.0)
            if dur < 10:
                bucket = "short"
            elif dur < 20:
                bucket = "medium"
            else:
                bucket = "long"

            if idx < len(clap_per_sample):
                dur_clap[bucket].append(clap_per_sample[idx])
            if idx < len(kl_per_sample):
                dur_kl[bucket].append(kl_per_sample[idx])

        model_analysis["by_duration"] = {}
        for bucket in ["short", "medium", "long"]:
            if dur_clap[bucket]:
                model_analysis["by_duration"][bucket] = {
                    "count": len(dur_clap[bucket]),
                    "avg_clap": round(float(np.mean(dur_clap[bucket])), 4),
                    "avg_kl": round(float(np.mean(dur_kl[bucket])), 4) if dur_kl[bucket] else None,
                }

        analysis[model] = model_analysis

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description="Cross-model comparison and ranking"
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--export", type=str, default=None,
                        help="Export comparison table as CSV")
    parser.add_argument("--rank", action="store_true",
                        help="Compute and display overall ranking")
    parser.add_argument("--categories", action="store_true",
                        help="Run category-wise breakdown analysis")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV),
                        help="Metadata CSV for category analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    reports = load_evaluation_reports(output_dir, args.models)

    if not reports:
        print(f"No evaluation_report.json files found in {output_dir}")
        print("Run evaluation first: python evaluation/run_all_evaluation.py")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Cross-Model Comparison — {len(reports)} runs")
    print(f"{'='*70}\n")

    rows = extract_metrics_table(reports)

    if args.rank:
        rows = rank_models(rows)

    print_comparison_table(rows)

    if args.export:
        export_csv(rows, Path(args.export))

    if args.categories:
        print(f"\n{'='*70}")
        print(f"  Category-Wise Analysis")
        print(f"{'='*70}")
        analysis = category_analysis(reports, args.csv)
        for model, data in analysis.items():
            print(f"\n  {model}:")
            if "by_num_layers" in data:
                print(f"    By complexity (num layers):")
                for layer_key, vals in data["by_num_layers"].items():
                    print(f"      {layer_key}: n={vals['count']}  "
                          f"CLAP={vals['avg_clap']:.4f}  KL={vals['avg_kl']:.4f}")
            if "by_duration" in data:
                print(f"    By duration:")
                for bucket, vals in data["by_duration"].items():
                    if vals:
                        kl_str = f"KL={vals['avg_kl']:.4f}" if vals.get("avg_kl") else ""
                        print(f"      {bucket}: n={vals['count']}  "
                              f"CLAP={vals['avg_clap']:.4f}  {kl_str}")

        # Save analysis
        analysis_path = output_dir / "category_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n  Category analysis saved: {analysis_path}")

    # Save comparison JSON
    comparison_path = output_dir / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\n  Comparison report: {comparison_path}")


if __name__ == "__main__":
    main()
