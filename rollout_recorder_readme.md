# Rollout Recorder

[`rollout_recorder.py`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/rollout_recorder.py) is a small helper for saving rollout observations as `.mp4` videos plus a `manifest.json`.

It is designed to plug into an existing rollout loop such as [`baseline_eval.py`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/baseline_eval.py), without changing the policy or environment logic very much.

## What it saves

For each rollout, it can save:

- `rollout_000.mp4`
- `rollout_001.mp4`
- `manifest.json`

The manifest stores metadata like:

- rollout index
- success / failure
- step count
- number of captured frames
- task / condition metadata you pass in

## Basic usage

```python
from rollout_recorder import RolloutRecorder

recorder = RolloutRecorder(
    "rollout_videos/task0_none_seed42",
    fps=10,
    include_wrist=False,
)
```

Then inside your rollout loop:

```python
def run_rollout(env, policy, task_description, args, recorder=None, rollout_index=0):
    obs = env.reset()

    if recorder is not None:
        recorder.start_rollout(
            rollout_index,
            metadata={
                "task_description": task_description,
                "condition": args.condition,
                "seed": args.seed,
            },
        )
        recorder.add_observation(obs)

    success = False
    total_steps = 0

    for step_idx in range(args.max_steps):
        formatted_obs = format_obs(obs, task_description, device)
        action = policy.select_action(formatted_obs)
        obs, reward, done, info = env.step(action)
        total_steps = step_idx + 1

        if recorder is not None:
            recorder.add_observation(obs)

        if env.check_success():
            success = True
            break
        if done:
            break

    if recorder is not None:
        recorder.finish_rollout(
            success=success,
            total_steps=total_steps,
            extra_metadata={
                "note": "example rollout",
            },
        )
```

## How it works

- `start_rollout(...)` resets the internal frame buffer and stores metadata.
- `add_observation(obs)` pulls images out of the observation dict.
- `finish_rollout(...)` writes the `.mp4` and updates `manifest.json`.

## Expected observation keys

The recorder looks for:

- primary camera:
  - `agentview_image`
  - or `agentview_rgb`
- wrist camera:
  - `robot0_eye_in_hand_image`
  - or `wrist_image`

If `include_wrist=True`, it stacks the wrist image next to the main image.

## Example command

With the current [`baseline_eval.py`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/baseline_eval.py), recording is enabled like this:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=$PWD/.mplconfig \
LIBERO_CONFIG_PATH=$PWD/.libero_config \
/opt/anaconda3/envs/project_Lumina/bin/python -u baseline_eval.py \
  --condition none \
  --task_idx 0 \
  --num_rollouts 1 \
  --max_steps 5 \
  --record \
  --video_dir test_rollout_videos
```

This writes videos under:

[`test_rollout_videos/task0_none_seed42`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/test_rollout_videos/task0_none_seed42)

