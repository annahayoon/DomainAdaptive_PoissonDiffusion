"""Analysis module for evaluation and metrics computation."""

from core.utils.analysis_utils import (
    compute_aggregate_stats,
    ensure_tensor_format,
    json_default,
    load_json_safe,
    save_json_safe,
)
from core.utils.file_utils import (
    count_regimes,
    find_latest_model,
    find_metadata_json,
    find_scene_directories,
    find_split_file,
    load_stitched_image,
    parse_exposure_from_scene_dir,
)
from core.utils.sensor_utils import (
    NOISE_REGIME_BAR_COLORS,
    NOISE_REGIME_COLORS,
    NOISE_REGIME_LABELS,
    NOISE_REGIME_NAMES,
    READ_NOISE_THRESHOLD,
    SHOT_NOISE_THRESHOLD,
    NoiseRegimeClassifier,
)

from .stratified_evaluation import (
    StratifiedEvaluator,
    format_significance_marker,
    format_stratified_results_table,
)

__all__ = [
    "StratifiedEvaluator",
    "format_stratified_results_table",
    "format_significance_marker",
    "save_json_safe",
    "load_json_safe",
    "ensure_tensor_format",
    "compute_aggregate_stats",
    "json_default",
    "find_split_file",
    "find_metadata_json",
    "find_latest_model",
    "NoiseRegimeClassifier",
    "load_stitched_image",
    "parse_exposure_from_scene_dir",
    "find_scene_directories",
    "NOISE_REGIME_NAMES",
    "NOISE_REGIME_LABELS",
    "NOISE_REGIME_COLORS",
    "NOISE_REGIME_BAR_COLORS",
    "SHOT_NOISE_THRESHOLD",
    "READ_NOISE_THRESHOLD",
    "count_regimes",
]
