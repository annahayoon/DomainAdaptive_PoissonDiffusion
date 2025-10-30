"""Analysis module for evaluation and metrics computation."""

from .stratified_evaluation import StratifiedEvaluator, format_stratified_results_table

__all__ = [
    "StratifiedEvaluator",
    "format_stratified_results_table",
]
