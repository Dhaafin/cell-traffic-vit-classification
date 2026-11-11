"""
Legacy entrypoint for DayViT 5-Fold Cross Validation.
Forwards execution to the modular runner: experiments.evaluate_cv_dayvit.
"""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from experiments.evaluate_cv_dayvit import run_cv_dayvit

if __name__ == "__main__":
    run_cv_dayvit()
