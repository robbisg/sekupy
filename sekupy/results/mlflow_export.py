"""Import already-saved sekupy analysis results (BIDS-derivatives directories
written by `Analyzer.save()`) into MLflow for browsing/comparison.

This is a batch/retroactive importer, not a live-tracking hook: it re-reads
`get_values()`'s output (config fields + per-fold scores, already used by
`get_results()` to build a pandas summary) and logs it into MLflow's run
model instead. Nothing in the core analysis pipeline depends on mlflow --
it's only imported here, lazily, so it stays an optional extra.

Mapping used:
  - one (result directory, permutation) pair -> one MLflow run
  - config fields (sample_slicer, target_transformer, cv, estimator, ...)
    -> run params
  - each score, per fold -> a step-indexed metric (one point per fold, so
    the MLflow UI can plot fold-to-fold variance), named "<roi>_<score>"
    when more than one ROI/.mat file lives in the directory
  - configuration.json, the raw .mat files, and any images sitting next to
    them (e.g. brain maps saved via sekupy.plot) -> run artifacts
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict

import numpy as np

from sekupy.results.base import get_values

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".svg", ".gif")

# MLflow only allows alphanumerics, underscore, dash, period, space, colon,
# slash in metric/param names.
_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_\-. :/]")


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow_export needs the optional 'mlflow' package: pip install mlflow"
        ) from exc
    return mlflow


def _safe_name(name: str) -> str:
    return _INVALID_NAME_CHARS.sub("_", str(name))


def _param_value(value, max_len: int = 250) -> str:
    return str(value)[:max_len]


def _group_rows(rows: list[dict]) -> dict:
    """Group get_values()'s flat rows by permutation, so each permutation
    (0 = the real run, >0 = a null-distribution run) becomes one MLflow run."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("permutation", 0)].append(row)
    return groups


def _mat_file_permutation(fname: str):
    """Parse the permutation index out of a `<roi>_<roi_value>_perm_<n>_<suffix>.mat`
    filename, matching get_values()'s own parsing -- used to attach only the
    matching .mat files to each permutation's run."""
    parts = fname.split("_")
    if len(parts) < 4:
        return None
    try:
        return np.float16(parts[-2])
    except ValueError:
        return None


def log_directory_to_mlflow(path: str, directory: str, field_list=("sample_slicer",),
                             result_keys=None, log_artifacts: bool = True,
                             log_images: bool = True) -> list[str]:
    """Log a single analysis result directory (as produced by
    `Analyzer.save()`, e.g. one entry of an `AnalysisIterator` sweep) into
    MLflow. Returns the list of MLflow run IDs created (one per permutation
    value found in the directory).
    """
    mlflow = _require_mlflow()

    rows = get_values(path, directory, list(field_list), result_keys)
    if not rows:
        logger.warning("No results found in %s -- skipping", directory)
        return []

    dir_path = os.path.join(path, directory)
    run_ids = []

    for permutation, perm_rows in _group_rows(rows).items():
        with mlflow.start_run(run_name=f"{directory}_perm{permutation}") as run:
            mlflow.set_tag("sekupy.directory", directory)
            mlflow.set_tag(
                "sekupy.permutation_type", "real" if permutation == 0 else "null"
            )

            non_param_keys = {"roi", "roi_value", "permutation", "fold"}
            base_fields = {
                k: v for k, v in perm_rows[0].items()
                if k not in non_param_keys and not k.startswith("score_")
            }
            mlflow.log_params(
                {_safe_name(k): _param_value(v) for k, v in base_fields.items()}
            )

            multi_roi = len({row.get("roi") for row in perm_rows}) > 1
            for row in perm_rows:
                fold = int(row.get("fold", 0))
                roi = row.get("roi")
                for key, value in row.items():
                    if not key.startswith("score_"):
                        continue
                    metric_name = f"{roi}_{key}" if multi_roi and roi else key
                    try:
                        mlflow.log_metric(_safe_name(metric_name), float(value), step=fold)
                    except (TypeError, ValueError):
                        logger.debug("Skipping non-numeric metric %s=%r", key, value)

            if log_artifacts:
                conf_fname = os.path.join(dir_path, "configuration.json")
                if os.path.exists(conf_fname):
                    mlflow.log_artifact(conf_fname)
                for fname in os.listdir(dir_path):
                    # only this run's own .mat files -- a directory holding
                    # multiple permutations would otherwise attach every
                    # permutation's raw results to every run
                    if fname.endswith(".mat") and _mat_file_permutation(fname) == permutation:
                        mlflow.log_artifact(os.path.join(dir_path, fname))

            if log_images:
                for fname in os.listdir(dir_path):
                    if fname.lower().endswith(IMAGE_EXTENSIONS):
                        mlflow.log_artifact(os.path.join(dir_path, fname))

            run_ids.append(run.info.run_id)

    return run_ids


def import_results_to_mlflow(path: str, pipeline_name: str, field_list=("sample_slicer",),
                              result_keys=None, experiment_name: str | None = None,
                              tracking_uri: str | None = None,
                              log_artifacts: bool = True, log_images: bool = True) -> list[str]:
    """Import every result directory matching `pipeline_name` under `path`
    into MLflow, one experiment per pipeline. Mirrors `get_results()`'s own
    directory filtering, so this imports exactly what that function would
    have summarized into a dataframe.

    Parameters
    ----------
    path : str
        Directory containing the analysis result subdirectories (what you'd
        pass as `get_results(path, ...)`).
    pipeline_name : str
        Substring used to select result directories, same as `get_results`.
    experiment_name : str, optional
        MLflow experiment name; defaults to `pipeline_name`.
    tracking_uri : str, optional
        Passed to `mlflow.set_tracking_uri()` if given (e.g. a local
        `./mlruns` path, or a tracking server URL). Defaults to whatever
        MLflow is already configured to use.

    Returns
    -------
    list of str
        MLflow run IDs created.
    """
    mlflow = _require_mlflow()

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name or pipeline_name)

    dir_analysis = [
        d for d in os.listdir(path)
        if d.find(pipeline_name) != -1 and d.find(".") == -1
    ]
    dir_analysis.sort()

    logger.info("Importing %d result directories into MLflow experiment %s",
                len(dir_analysis), experiment_name or pipeline_name)

    run_ids = []
    for directory in dir_analysis:
        run_ids.extend(
            log_directory_to_mlflow(path, directory, field_list, result_keys,
                                     log_artifacts=log_artifacts, log_images=log_images)
        )
    return run_ids
