# Installer Notes

This repo needed a few extra environment fixes to get `baseline_eval.py` and the rollout recorder closer to runnable inside the `project_Lumina` conda env.

## Goal

Enable:

- `baseline_eval.py`
- `rollout_recorder.py`
- recorded rollout smoke tests such as:

```bash
mkdir -p .mplconfig
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
MPLCONFIGDIR=$PWD/.mplconfig \
LIBERO_CONFIG_PATH=$PWD/.libero_config \
conda run -n project_Lumina python baseline_eval.py \
  --condition none \
  --task_idx 0 \
  --num_rollouts 1 \
  --max_steps 5 \
  --log_every 1 \
  --record \
  --video_dir test_rollout_videos
```

## Packages already present

`project_Lumina` already had:

- `cmake`
- `mujoco`
- `robosuite`

## Extra packages installed

Installed into `project_Lumina`:

```bash
conda run -n project_Lumina pip install bddl==1.0.1
conda run -n project_Lumina pip install future==0.18.2
conda run -n project_Lumina pip install easydict==1.9
conda run -n project_Lumina pip install matplotlib==3.5.3
conda run -n project_Lumina pip install gym==0.25.2
conda run -n project_Lumina pip install 'numpy<2'
```

## LIBERO source checkout

LIBERO was cloned locally at:

`/tmp/LIBERO`

The package metadata from that repo did not expose imports cleanly in this env, so the Python path was extended with:

`/opt/anaconda3/envs/project_Lumina/lib/python3.10/site-packages/libero_local.pth`

with contents:

```text
/tmp/LIBERO/libero
```

## Local LIBERO config

Created:

- [`/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/.libero_config/config.yaml`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/.libero_config/config.yaml)

Use it by exporting:

```bash
LIBERO_CONFIG_PATH=$PWD/.libero_config
```

## Matplotlib config

Matplotlib wanted to write under the home directory, which is not writable from this sandboxed workflow.

Use:

```bash
MPLCONFIGDIR=$PWD/.mplconfig
mkdir -p .mplconfig
```

## robosuite fix

`robosuite` was crashing on Numba cache initialization. A private macro override was added at:

`/opt/anaconda3/envs/project_Lumina/lib/python3.10/site-packages/robosuite/macros_private.py`

with:

```python
ENABLE_NUMBA = True
CACHE_NUMBA = False
```

## LIBERO compatibility shim

The LIBERO source mixes `libero.*` and `libero.libero.*` imports. A compatibility shim was added at:

`/tmp/LIBERO/libero/libero/libero/__init__.py`

to make nested imports resolve in this environment.

## Hugging Face offline cache

The SmolVLA model was already cached locally under:

`~/.cache/huggingface/hub/models--HuggingFaceVLA--smolvla_libero`

To avoid network retries in this environment, use:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

## accelerate compatibility patch

After downgrading NumPy to 1.26, the installed `accelerate` package still used a NumPy 2-style internal path.

Patched file:

`/opt/anaconda3/envs/project_Lumina/lib/python3.10/site-packages/accelerate/utils/other.py`

Current local patch forces:

```python
np_core = np.core
```

## robosuite macOS rendering patch

The stock `robosuite` install on this machine:

- set `MUJOCO_GL=cgl` on macOS
- but still fell through to the GLFW renderer path
- which hung inside `glfw.init()` in the sandbox

Patched file:

`/opt/anaconda3/envs/project_Lumina/lib/python3.10/site-packages/robosuite/utils/binding_utils.py`

Local fixes:

- Darwin + `MUJOCO_GL=cgl` now uses MuJoCo's `cgl` context
- the `cgl` context is wrapped to accept the `device_id` argument expected by robosuite

## Sandbox note

On this macOS setup, LIBERO offscreen rendering did **not** fully work inside the sandboxed command environment because MuJoCo/CGL failed with:

`invalid CoreGraphics connection`

The same probe succeeded when run outside the sandbox.

So the real rollout test was validated with an unsandboxed command.

## Repo code changes

### `baseline_eval.py`

- added optional rollout recording flags:
  - `--record`
  - `--video_dir`
  - `--video_fps`
  - `--record_wrist`
- integrated the reusable recorder hooks into the rollout loop
- added more tolerant LIBERO import fallback logic
- added a fallback for LIBERO init-state loading on PyTorch 2.6+ using `weights_only=False`

### `rollout_recorder.py`

- new reusable observation-based rollout recorder
- writes:
  - `rollout_XXX.mp4`
  - `manifest.json`

## Known status

- The recorder module itself passed a synthetic smoke test.
- A real recorded rollout smoke test succeeded outside the sandbox with:
  - 1 rollout
  - task 0
  - 5 max steps
- Output artifacts created:
  - [`/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/test_rollout_videos/task0_none_seed42/rollout_000.mp4`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/test_rollout_videos/task0_none_seed42/rollout_000.mp4)
  - [`/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/test_rollout_videos/task0_none_seed42/manifest.json`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/test_rollout_videos/task0_none_seed42/manifest.json)
  - [`/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/baseline_results/task0_none_seed42.json`](/Users/ashutoshpanda/Documents/Research/project_Flux/project_Flux/baseline_results/task0_none_seed42.json)
