"""
Unified Benchmark Runner & Master Consolidated JSON Generator for Cellular ViT.
Executes 5-Fold Cross-Validation across:
1. DayViT (Native 7x24, Compact 58.8K params, From-Scratch)
2. ViT-Tiny Pretrained (ImageNet-1k, 5.5M params, LP-FT)
3. ViT-Tiny From-Scratch (Random Init, 5.5M params, End-to-End)

Supported CV Modes:
  --cv_mode stratified (Default: Sample-independent temporal split)
  --cv_mode group      (Site-independent 19 eNodeB spatial split)

Consolidates all models into:
  - outputs/leaderboard/<cv_mode>/unified_benchmark_results.json (Single Master JSON)
  - outputs/leaderboard/<cv_mode>/table_comparison.tex (LaTeX Table for Academic Paper)
  - outputs/leaderboard/<cv_mode>/01_boxplot_comparison.png
  - outputs/leaderboard/<cv_mode>/02_leaderboard_acc_vs_params.png

Usage:
    python experiments/run_benchmark.py --model all --cv_mode stratified
    python experiments/run_benchmark.py --model all --cv_mode group
"""
import sys
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from experiments.evaluate_cv_dayvit import run_cv_dayvit
from experiments.evaluate_cv_vit_tiny_pretrained import run_cv_vit_tiny_pretrained
from experiments.evaluate_cv_vit_tiny_scratch import run_cv_vit_tiny_scratch
from src.utils import plot_cv_boxplot, plot_leaderboard, format_cv_mode_name


def generate_latex_table(models_data: list, cv_mode_title: str) -> str:
    """Generates ready-to-copy LaTeX table for academic journal paper."""
    latex_code = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{Comprehensive Benchmark of Vision Transformer Paradigms for Cellular Traffic Classification ({cv_mode_title})}}",
        r"\label{tab:vit_benchmark_comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccccc}",
        r"\hline",
        r"\textbf{Architecture} & \textbf{5-Fold Accuracy} & \textbf{Macro F1} & \textbf{Macro ROC-AUC} & \textbf{Cohen's $\kappa$} & \textbf{FLOPs} & \textbf{Peak VRAM} & \textbf{Latency / Sample} & \textbf{Params} \\",
        r"\hline"
    ]

    for m in models_data:
        sc = m["summary_card"]
        eff = m["efficiency_profiling"]
        name = sc["model_architecture"]
        acc = sc["5fold_cv_accuracy"]
        f1 = sc["5fold_cv_macro_f1"]
        roc = f"{sc['macro_roc_auc']:.4f}" if isinstance(sc.get("macro_roc_auc"), float) else str(sc.get("macro_roc_auc", "-"))
        kappa = f"{sc['cohen_kappa']:.4f}" if isinstance(sc.get("cohen_kappa"), float) else str(sc.get("cohen_kappa", "-"))
        flops = eff.get("flops_formatted", "-")
        vram = f"{eff.get('peak_vram_mb', 0.0)} MB"
        lat = f"{eff.get('inference_latency_ms', 0.0)} ms"
        params = sc["total_parameters"]

        latex_code.append(f"{name} & {acc} & {f1} & {roc} & {kappa} & {flops} & {vram} & {lat} & {params} \\\\")

    latex_code.extend([
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}"
    ])
    return "\n".join(latex_code)


def main():
    parser = argparse.ArgumentParser(description="Cellular Vision Transformer 5-Fold CV Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "dayvit", "vit_tiny_pretrained", "vit_tiny_scratch"],
        help="Model architecture to evaluate."
    )
    parser.add_argument(
        "--cv_mode",
        type=str,
        default="stratified",
        choices=["stratified", "group"],
        help="Cross-validation mode: 'stratified' (Temporal split) or 'group' (Site-independent 19 eNodeB spatial split)"
    )
    parser.add_argument("--epochs", type=int, default=70, help="Epochs for standard from-scratch models")
    parser.add_argument("--lp_epochs", type=int, default=5, help="Epochs for LP stage")
    parser.add_argument("--ft_epochs", type=int, default=15, help="Epochs for FT stage")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    cv_folder = "group" if args.cv_mode in ["group", "group_kfold"] else "stratified"
    leaderboard_dir = ROOT_DIR / "outputs" / "leaderboard" / cv_folder
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    mode_title = format_cv_mode_name(args.cv_mode)

    reports = []
    box_data = {}

    if args.model in ["all", "dayvit"]:
        rep_dayvit = run_cv_dayvit(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed_base=args.seed,
            cv_mode=args.cv_mode
        )
        reports.append(rep_dayvit)
        box_data["DayViT (Ours)"] = [f["accuracy"] * 100 for f in rep_dayvit["cross_validation_distribution"]["per_fold"]]

    if args.model in ["all", "vit_tiny_pretrained"]:
        rep_pt = run_cv_vit_tiny_pretrained(
            lp_epochs=args.lp_epochs,
            ft_epochs=args.ft_epochs,
            batch_size=args.batch_size,
            seed_base=args.seed,
            cv_mode=args.cv_mode
        )
        reports.append(rep_pt)
        box_data["ViT-Tiny (Pretrained LP-FT)"] = [f["accuracy"] * 100 for f in rep_pt["cross_validation_distribution"]["per_fold"]]

    if args.model in ["all", "vit_tiny_scratch"]:
        rep_sc = run_cv_vit_tiny_scratch(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed_base=args.seed,
            cv_mode=args.cv_mode
        )
        reports.append(rep_sc)
        box_data["ViT-Tiny (From-Scratch)"] = [f["accuracy"] * 100 for f in rep_sc["cross_validation_distribution"]["per_fold"]]

    # When multiple models are benchmarked, consolidate into ONE MASTER JSON
    if len(reports) > 1:
        # Statistical significance test (Paired t-test against DayViT if DayViT is present)
        dayvit_accs = box_data.get("DayViT (Ours)", None)
        stat_significance = {}
        if dayvit_accs is not None and len(dayvit_accs) == 5:
            for model_name, fold_accs in box_data.items():
                if model_name != "DayViT (Ours)" and len(fold_accs) == 5:
                    t_stat, p_val = stats.ttest_rel(dayvit_accs, fold_accs)
                    stat_significance[model_name] = {
                        "paired_t_statistic": round(float(t_stat), 4),
                        "p_value": round(float(p_val), 5),
                        "is_significant_p05": bool(p_val < 0.05),
                        "interpretation": "DayViT significantly superior (p < 0.05)" if p_val < 0.05 else "Not statistically significant"
                    }

        # Build Master Consolidated Leaderboard Dictionary
        master_leaderboard_json = {
            "benchmark_title": f"Cellular Vision Transformer Benchmark ({mode_title})",
            "cv_mode": args.cv_mode,
            "cv_mode_description": mode_title,
            "number_of_models_evaluated": len(reports),
            "summary_comparison_table": [r["summary_card"] for r in reports],
            "statistical_significance_vs_dayvit": stat_significance,
            "individual_model_detailed_reports": {r["model_name"]: r for r in reports}
        }

        # 1. Save MASTER CONSOLIDATED JSON
        master_json_path = leaderboard_dir / "unified_benchmark_results.json"
        with open(master_json_path, "w", encoding="utf-8") as f:
            json.dump(master_leaderboard_json, f, indent=2)

        # 2. Save LaTeX Table
        latex_str = generate_latex_table(reports, mode_title)
        tex_path = leaderboard_dir / "table_comparison.tex"
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)

        # 3. Save Leaderboard CSV
        df_summary = pd.DataFrame([r["summary_card"] for r in reports])
        df_summary.to_csv(leaderboard_dir / "benchmark_leaderboard.csv", index=False)

        # 4. Plots
        plot_cv_boxplot(
            box_data,
            leaderboard_dir / "01_boxplot_comparison.png",
            title=f"5-Fold Cross Validation Accuracy ({mode_title})"
        )

        plot_data = []
        for r in reports:
            sc = r["summary_card"]
            eff = r["efficiency_profiling"]
            cd = r["cross_validation_distribution"]
            plot_data.append({
                "name": r["model_name"],
                "mean_acc": cd["accuracy_mean"],
                "std_acc": cd["accuracy_std"],
                "params": eff.get("total_parameters", 0)
            })

        plot_leaderboard(
            plot_data,
            leaderboard_dir / "02_leaderboard_acc_vs_params.png",
            title=f"Accuracy vs Parameter Footprint ({cv_folder.capitalize()} Mode)"
        )

        # Print executive summary
        print("\n" + "=" * 90)
        print(f"  MASTER LEADERBOARD: Cellular ViT [{cv_folder.upper()} MODE]")
        print("=" * 90)
        print(df_summary.to_string(index=False))
        print("=" * 90)
        if stat_significance:
            print(f"\n[Statistical Significance (Paired t-test vs DayViT)]: ")
            for m, s in stat_significance.items():
                print(f"  vs {m:30s}: t = {s['paired_t_statistic']:+.3f}, p = {s['p_value']:.5f} ({s['interpretation']})")
        print(f"\n[Master Artifact] Single Master JSON saved to: {master_json_path}")
        print(f"[LaTeX Table] Ready-to-copy LaTeX code saved to: {tex_path}")


if __name__ == "__main__":
    main()
