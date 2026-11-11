"""
Unified Benchmark Runner & Leaderboard Generator for Cellular Vision Transformers.
Compares:
1. DayViT (Native 7x24, Compact, 58.8K params, Trained from scratch)
2. ViT-Tiny Pretrained (ImageNet-1k, 5.5M params, Multi-View 224x224 RGB, LP-FT)
3. ViT-Tiny From-Scratch (Random Init, 5.5M params, Multi-View 224x224 RGB, End-to-End)

Usage:
    python experiments/run_benchmark.py --model all
    python experiments/run_benchmark.py --model dayvit
    python experiments/run_benchmark.py --model vit_tiny_pretrained
    python experiments/run_benchmark.py --model vit_tiny_scratch
"""
import sys
import argparse
from pathlib import Path
import json
import pandas as pd

# Ensure repo root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from experiments.evaluate_cv_dayvit import run_cv_dayvit
from experiments.evaluate_cv_vit_tiny_pretrained import run_cv_vit_tiny_pretrained
from experiments.evaluate_cv_vit_tiny_scratch import run_cv_vit_tiny_scratch
from src.utils import plot_cv_boxplot, plot_leaderboard

LEADERBOARD_DIR = ROOT_DIR / "outputs" / "leaderboard"
LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Cellular Vision Transformer 5-Fold CV Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "dayvit", "vit_tiny_pretrained", "vit_tiny_scratch"],
        help="Model architecture to evaluate."
    )
    parser.add_argument("--epochs", type=int, default=70, help="Epochs for standard from-scratch models")
    parser.add_argument("--lp_epochs", type=int, default=5, help="Epochs for LP stage")
    parser.add_argument("--ft_epochs", type=int, default=15, help="Epochs for FT stage")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    results = []
    box_data = {}

    if args.model in ["all", "dayvit"]:
        res_dayvit = run_cv_dayvit(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed_base=args.seed
        )
        results.append(res_dayvit)
        box_data["DayViT (Ours)"] = res_dayvit["fold_accs"]

    if args.model in ["all", "vit_tiny_pretrained"]:
        res_pt = run_cv_vit_tiny_pretrained(
            lp_epochs=args.lp_epochs,
            ft_epochs=args.ft_epochs,
            batch_size=args.batch_size,
            seed_base=args.seed
        )
        results.append(res_pt)
        box_data["ViT-Tiny (Pretrained LP-FT)"] = res_pt["fold_accs"]

    if args.model in ["all", "vit_tiny_scratch"]:
        res_sc = run_cv_vit_tiny_scratch(
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed_base=args.seed
        )
        results.append(res_sc)
        box_data["ViT-Tiny (From-Scratch)"] = res_sc["fold_accs"]

    # If multiple models were evaluated, build comparative leaderboard
    if len(results) > 1:
        leaderboard_rows = []
        for r in results:
            leaderboard_rows.append({
                "Architecture": r["model"],
                "Mean Accuracy (%)": f"{r['mean_acc']:.2f}% ± {r['std_acc']:.2f}%",
                "Mean Macro F1 (%)": f"{r['mean_f1']:.2f}%",
                "Parameters": f"{r['params']:,}",
                "CV Runtime (s)": f"{r['total_time']:.1f}s"
            })

        df_leaderboard = pd.DataFrame(leaderboard_rows)
        print("\n" + "=" * 80)
        print("                   CELLULAR VISION TRANSFORMER LEADERBOARD")
        print("=" * 80)
        print(df_leaderboard.to_string(index=False))
        print("=" * 80)

        # Save to outputs/leaderboard
        df_leaderboard.to_csv(LEADERBOARD_DIR / "benchmark_leaderboard.csv", index=False)
        with open(LEADERBOARD_DIR / "benchmark_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        # Plot comparison boxplot
        plot_cv_boxplot(
            box_data,
            LEADERBOARD_DIR / "01_boxplot_comparison.png",
            title="5-Fold Cross Validation Accuracy Across ViT Paradigms"
        )

        # Plot leaderboard bars vs parameters
        plot_data = [
            {"name": r["model"], "mean_acc": r["mean_acc"], "std_acc": r["std_acc"], "params": r["params"]}
            for r in results
        ]
        plot_leaderboard(
            plot_data,
            LEADERBOARD_DIR / "02_leaderboard_acc_vs_params.png",
            title="Cellular ViT Benchmark: Accuracy vs. Parameter Footprint"
        )
        print(f"\n[Artifacts] Leaderboard charts and tables saved to: {LEADERBOARD_DIR}")


if __name__ == "__main__":
    main()
