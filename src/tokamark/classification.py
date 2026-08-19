"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import label as ndimage_label

from MAST_tools.MAST_dataset import MastDataset

# ----------------------------------------------------------------------------------------------------------------------
# ANNOTATION LOADING


# ----------------------------------------------------------------------------------------------------------------------
def load_event_annotations(labels_file: str) -> pd.DataFrame:
    """
    Load an event-annotation JSON file (as ported from cocoa's `data/annotations/`) into a DataFrame of
    per-shot time-region labels.

    Parameters
    ----------
    labels_file : str
        Path to the annotations JSON file. Expected schema: a list of records each containing (at least)
        `shot_id`, `time_min`, and `time_max` fields.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `shot_id` (int), `time_min` (float), `time_max` (float).

    """

    with open(labels_file, "r") as f:
        records = json.load(f)

    df = pd.DataFrame.from_records(records)

    return df[["shot_id", "time_min", "time_max"]].astype({"shot_id": int, "time_min": float, "time_max": float})


# ----------------------------------------------------------------------------------------------------------------------
def get_annotated_shots(annotations: pd.DataFrame, shots_list: Sequence[int]) -> list[int]:
    """
    Filter a split's shot list down to only shots that have at least one annotation for the target task.

    Parameters
    ----------
    annotations : pd.DataFrame
        Annotations DataFrame, as returned by `load_event_annotations`.
    shots_list : Sequence[int]
        Candidate list of shot IDs (e.g. from `tokamark.data_split.get_train_test_val_shots`).

    Returns
    -------
    list[int]
        Subset of `shots_list` for which annotations exist, in the original order.

    """

    annotated_shot_ids = set(annotations["shot_id"].unique().tolist())

    return [shot_id for shot_id in shots_list if shot_id in annotated_shot_ids]


# ----------------------------------------------------------------------------------------------------------------------
# FLAT-TOP / DISRUPTION DETECTION
#
# REMARK: tokamark's `summary-ip` signal is in Amps, while the `ip_threshold_ka` values ported from cocoa are in kA.
# Values are converted to kA before thresholding so the ported thresholds remain valid.


# ----------------------------------------------------------------------------------------------------------------------
def detect_flattop_window(
    times: np.ndarray, ip_values_amps: np.ndarray, ip_threshold_ka: float, flattop_min_frac: float
) -> tuple[float, float] | None:
    """
    Detect the longest contiguous flat-top region of a shot's plasma current trace. Port of cocoa's
    `_detect_flattop_window` (`cocoa/dataset.py`).

    Parameters
    ----------
    times : np.ndarray
        Time values for the Ip signal, in seconds.
    ip_values_amps : np.ndarray
        Ip signal values, in Amps (as returned by tokamark's `summary-ip` signal).
    ip_threshold_ka : float
        Minimum plasma current, in kA, to consider the plasma "on".
    flattop_min_frac : float
        Fraction of the 95th-percentile peak current above which the current is considered flat-top.

    Returns
    -------
    tuple[float, float] | None
        (t_start, t_end) of the longest flat-top region, in seconds, or None if no flat-top region is found.

    """

    times = np.asarray(times, dtype=float)
    ip_ka = np.asarray(ip_values_amps, dtype=float) / 1000.0

    valid = np.isfinite(times) & np.isfinite(ip_ka)
    if not np.any(valid):
        return None
    times, ip_ka = times[valid], ip_ka[valid]

    above = ip_ka > ip_threshold_ka
    if not above.any():
        return None

    ip_peak = float(np.percentile(ip_ka[above], 95))
    flat_thresh = max(ip_threshold_ka, flattop_min_frac * ip_peak)
    flat_mask = ip_ka > flat_thresh
    if not flat_mask.any():
        return None

    labeled, n_regions = ndimage_label(flat_mask)
    if n_regions == 0:
        return None

    longest = max(range(1, n_regions + 1), key=lambda i: int((labeled == i).sum()))
    region = labeled == longest

    t_start = float(times[region].min())
    t_end = float(times[region].max())

    return (t_start, t_end) if t_start < t_end else None


# ----------------------------------------------------------------------------------------------------------------------
def find_disruption_time(times: np.ndarray, ip_values_amps: np.ndarray, ip_threshold_ka: float) -> float | None:
    """
    Find the last time at which the plasma current is at or above the threshold (used to clip windows near a
    disruption for the IRE task). Port of cocoa's disruption-time finder.

    Parameters
    ----------
    times : np.ndarray
        Time values for the Ip signal, in seconds.
    ip_values_amps : np.ndarray
        Ip signal values, in Amps (as returned by tokamark's `summary-ip` signal).
    ip_threshold_ka : float
        Plasma current threshold, in kA, defining "the plasma is still up".

    Returns
    -------
    float | None
        Last time (in seconds) at which Ip >= ip_threshold_ka, or None if Ip never reaches the threshold.

    """

    times = np.asarray(times, dtype=float)
    ip_ka = np.asarray(ip_values_amps, dtype=float) / 1000.0

    valid = np.isfinite(times) & np.isfinite(ip_ka)
    if not np.any(valid):
        return None
    times, ip_ka = times[valid], ip_ka[valid]

    above = ip_ka >= ip_threshold_ka
    if not above.any():
        return None

    return float(times[above][-1])


# ----------------------------------------------------------------------------------------------------------------------
# WINDOW LABELLING RULES


# ----------------------------------------------------------------------------------------------------------------------
def label_window(labelling_rule: str, t_start: float, t_cut: float, gt_min: np.ndarray, gt_max: np.ndarray) -> bool:
    """
    Assign a binary label to a window based on its overlap with a shot's annotated event intervals.

    Parameters
    ----------
    labelling_rule : str
        Either "overlap" (window is positive if it overlaps any annotated interval at all; the ELM rule) or
        "center_in_interval" (window is positive if its centre falls inside an annotated interval; the
        IRE/sawteeth rule).
    t_start : float
        Window start time, in seconds.
    t_cut : float
        Window end time, in seconds.
    gt_min : np.ndarray
        Array of annotated interval start times for the shot.
    gt_max : np.ndarray
        Array of annotated interval end times for the shot.

    Returns
    -------
    bool
        True if the window is labelled as containing the event.

    """

    if labelling_rule == "overlap":
        return bool(np.any((gt_min < t_cut) & (gt_max > t_start)))

    if labelling_rule == "center_in_interval":
        t_center = (t_start + t_cut) / 2.0
        return bool(np.any((gt_min <= t_center) & (t_center <= gt_max)))

    raise ValueError(f"Unknown labelling_rule: {labelling_rule!r}")


# ----------------------------------------------------------------------------------------------------------------------
# PER-SHOT WINDOWING BOUNDS


# ----------------------------------------------------------------------------------------------------------------------
def precompute_shot_windowing_bounds(
    raw_ip_dataset: MastDataset, config_task: Mapping[str, Any], annotations: pd.DataFrame
) -> dict[int, dict[str, Any]]:
    """
    Precompute, once per shot, the flat-top region and any task-specific extra bounds (disruption clip time for
    IRE, annotation-span restriction for sawteeth) needed to filter and label windows.

    Must be run against a *raw* (unscaled) Ip dataset, since flat-top/disruption detection compare against an
    absolute kA threshold, and `initialize_MAST_dataset` applies std-scaling by default.

    Parameters
    ----------
    raw_ip_dataset : MastDataset
        Dataset built with `use_std_scaling=False`, containing (at least) the task's `ip_signal`.
    config_task : Mapping[str, Any]
        Task configuration dictionary, containing a `classification` section.
    annotations : pd.DataFrame
        Annotations DataFrame, as returned by `load_event_annotations`.

    Returns
    -------
    dict[int, dict[str, Any]]
        Mapping from shot_id to a dict with keys "flattop" (tuple[float, float] | None),
        "disruption_time" (float | None), and "annotation_span" (tuple[float, float] | None).

    """

    cls_config = config_task["classification"]

    ip_signal = cls_config["ip_signal"]
    ip_threshold_ka = cls_config["ip_threshold_ka"]
    flattop_min_frac = cls_config["flattop_min_frac"]
    disruption_clip_s = cls_config.get("disruption_clip_s")
    annotation_span_margin_s = cls_config.get("annotation_span_margin_s")

    bounds_by_shot: dict[int, dict[str, Any]] = {}

    for idx in range(len(raw_ip_dataset)):
        shot_id = raw_ip_dataset.get_shot_id(idx)
        ip_series = raw_ip_dataset[idx][ip_signal]
        # REMARK: `MastDataset` returns single-channel signals with a leading channel dim of size 1 (shape (1, N)),
        # unlike `TokaMarkDataset._build_window`, which squeezes it. Flatten it here since this precompute step
        # reads straight from the raw `MastDataset`.
        ip_values_amps = np.asarray(ip_series["values"]).reshape(-1)

        flattop = detect_flattop_window(
            times=ip_series["time"],
            ip_values_amps=ip_values_amps,
            ip_threshold_ka=ip_threshold_ka,
            flattop_min_frac=flattop_min_frac,
        )

        disruption_time = None
        if disruption_clip_s is not None:
            disruption_time = find_disruption_time(
                times=ip_series["time"], ip_values_amps=ip_values_amps, ip_threshold_ka=ip_threshold_ka
            )

        annotation_span = None
        if annotation_span_margin_s is not None:
            shot_rows = annotations[annotations["shot_id"] == shot_id]
            if len(shot_rows) > 0:
                annotation_span = (
                    float(shot_rows["time_min"].min()) - annotation_span_margin_s,
                    float(shot_rows["time_max"].max()) + annotation_span_margin_s,
                )

        bounds_by_shot[shot_id] = {
            "flattop": flattop,
            "disruption_time": disruption_time,
            "annotation_span": annotation_span,
        }

    return bounds_by_shot


# ----------------------------------------------------------------------------------------------------------------------
# CUSTOM_TRANSFORM FACTORY


# ----------------------------------------------------------------------------------------------------------------------
def _expand_channel_selection(
    input_slice: Mapping[str, Any], channel_selection: Mapping[str, Mapping[str, int]]
) -> dict[str, Any]:
    """
    Expand any multi-channel signal listed in `channel_selection` into separately named single-channel entries,
    alongside all other (already single-channel) signals.

    Parameters
    ----------
    input_slice : Mapping[str, Any]
        A window's "input" dict, as built by `TokaMarkDataset._build_window`: `{"source-signal": {"time": ...,
        "values": ...}}`.
    channel_selection : Mapping[str, Mapping[str, int]]
        Map of `{"source-signal": {sub_channel_name: channel_index}}`, e.g.
        `{"soft_x_rays-horizontal_cam_upper": {"sxr_core": 0, "sxr_edge": 7}}`.

    Returns
    -------
    dict[str, Any]
        Input dict with multi-channel signals replaced by their named sub-channels.

    """

    out = {key: val for key, val in input_slice.items() if key not in channel_selection}

    for source_signal_key, sub_channels in channel_selection.items():
        values = input_slice[source_signal_key]["values"]
        times = input_slice[source_signal_key]["time"]
        for name, channel_idx in sub_channels.items():
            out[name] = {"time": times, "values": values[channel_idx]}

    return out


# ----------------------------------------------------------------------------------------------------------------------
def make_classification_transform(
    config_task: Mapping[str, Any], annotations: pd.DataFrame, bounds_by_shot: Mapping[int, Mapping[str, Any]]
) -> Callable[[Mapping[str, Any]], dict[str, Any] | None]:
    """
    Build the per-window `custom_transform` for a group-5 classification task, suitable for passing to
    `tokamark.data.initialize_TokaMark_dataset`.

    For each window, the returned callable restricts to the shot's flat-top region (and any task-specific extra
    bounds), assigns a binary event/background label from the annotations, and reshapes the window into
    `{"input": ..., "label": ..., "t_cut": ...}`. Windows outside the relevant region return None and are
    dropped by `TokaMarkDataset._process_shot`.

    Parameters
    ----------
    config_task : Mapping[str, Any]
        Task configuration dictionary, containing a `classification` section.
    annotations : pd.DataFrame
        Annotations DataFrame, as returned by `load_event_annotations`.
    bounds_by_shot : Mapping[int, Mapping[str, Any]]
        Per-shot windowing bounds, as returned by `precompute_shot_windowing_bounds`.

    Returns
    -------
    Callable[[Mapping[str, Any]], dict[str, Any] | None]
        Per-window transform function.

    """

    cls_config = config_task["classification"]
    labelling_rule = cls_config["labelling_rule"]
    disruption_clip_s = cls_config.get("disruption_clip_s")
    channel_selection = cls_config.get("channel_selection", {})
    window_length = config_task["task_window_segmenter"]["input_length"]

    annotations_by_shot = {shot_id: rows for shot_id, rows in annotations.groupby("shot_id")}

    # ------------------------------------------------------------------------------------------------------------------
    def _transform(obj: Mapping[str, Any]) -> dict[str, Any] | None:
        shot_id = obj["shot_id"]
        t_cut = float(obj["t_cut"])
        t_start = t_cut - window_length

        bounds = bounds_by_shot.get(shot_id)
        if (bounds is None) or (bounds["flattop"] is None):
            return None

        flat_start, flat_end = bounds["flattop"]
        if not (flat_start <= t_start and t_cut <= flat_end):
            return None

        if (bounds["disruption_time"] is not None) and (t_cut >= bounds["disruption_time"] - disruption_clip_s):
            return None

        if bounds["annotation_span"] is not None:
            span_start, span_end = bounds["annotation_span"]
            if not (span_start <= t_start and t_cut <= span_end):
                return None

        shot_rows = annotations_by_shot.get(shot_id)
        gt_min = shot_rows["time_min"].to_numpy() if shot_rows is not None else np.array([])
        gt_max = shot_rows["time_max"].to_numpy() if shot_rows is not None else np.array([])

        is_event = label_window(
            labelling_rule=labelling_rule, t_start=t_start, t_cut=t_cut, gt_min=gt_min, gt_max=gt_max
        )

        expanded_input = _expand_channel_selection(input_slice=obj["input"], channel_selection=channel_selection)

        return {"input": expanded_input, "label": int(is_event), "t_cut": t_cut}

    return _transform
