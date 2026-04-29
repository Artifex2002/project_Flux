#!/usr/bin/env python3
"""
Utility helpers for saving rollout videos from observation frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError as e:
    raise ImportError(
        "rollout_recorder.py requires imageio. Install it with `pip install imageio imageio-ffmpeg`."
    ) from e


def _to_uint8_image(frame):
    """Convert an HWC image array to uint8."""
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image with 3 dims, got shape {arr.shape}")

    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.clip(0, 255).astype(np.uint8)

    return arr


def _stack_side_by_side(left, right):
    """Pad heights if needed and stack two HWC frames horizontally."""
    left_h, left_w = left.shape[:2]
    right_h, right_w = right.shape[:2]
    out_h = max(left_h, right_h)

    if left_h != out_h:
        pad = np.zeros((out_h - left_h, left_w, left.shape[2]), dtype=left.dtype)
        left = np.concatenate([left, pad], axis=0)
    if right_h != out_h:
        pad = np.zeros((out_h - right_h, right_w, right.shape[2]), dtype=right.dtype)
        right = np.concatenate([right, pad], axis=0)

    return np.concatenate([left, right], axis=1)


class RolloutRecorder:
    """Records rollout frames from observations and writes mp4 videos."""

    def __init__(
        self,
        output_dir,
        *,
        fps=10,
        include_wrist=False,
        save_failed=True,
        save_success=True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.include_wrist = bool(include_wrist)
        self.save_failed = bool(save_failed)
        self.save_success = bool(save_success)
        self.manifest_path = self.output_dir / "manifest.json"
        self._manifest = []
        self._frames = []
        self._rollout_index = None
        self._rollout_meta = None

    def start_rollout(self, rollout_index, *, metadata=None):
        self._frames = []
        self._rollout_index = int(rollout_index)
        self._rollout_meta = dict(metadata or {})

    def add_observation(self, obs):
        frame = self._build_frame(obs)
        if frame is not None:
            self._frames.append(frame)

    def finish_rollout(self, *, success, total_steps, extra_metadata=None):
        should_save = (success and self.save_success) or ((not success) and self.save_failed)

        record = dict(self._rollout_meta or {})
        record.update(extra_metadata or {})
        record["rollout_index"] = self._rollout_index
        record["success"] = bool(success)
        record["total_steps"] = int(total_steps)
        record["num_frames"] = len(self._frames)

        if should_save and self._frames:
            filename = f"rollout_{self._rollout_index:03d}.mp4"
            video_path = self.output_dir / filename
            with imageio.get_writer(video_path, fps=self.fps) as writer:
                for frame in self._frames:
                    writer.append_data(frame)
            record["video_file"] = filename
        else:
            record["video_file"] = None

        self._manifest.append(record)
        self._write_manifest()

        self._frames = []
        self._rollout_index = None
        self._rollout_meta = None
        return record

    def _write_manifest(self):
        with self.manifest_path.open("w") as f:
            json.dump({"rollouts": self._manifest}, f, indent=2)

    def _build_frame(self, obs):
        primary = self._get_first_image(obs, ["agentview_image", "agentview_rgb"])
        if primary is None:
            return None

        primary = _to_uint8_image(primary)
        if not self.include_wrist:
            return primary

        wrist = self._get_first_image(obs, ["robot0_eye_in_hand_image", "wrist_image"])
        if wrist is None:
            return primary

        wrist = _to_uint8_image(wrist)
        return _stack_side_by_side(primary, wrist)

    @staticmethod
    def _get_first_image(obs, keys):
        for key in keys:
            if key in obs and obs[key] is not None:
                return obs[key]
        return None
