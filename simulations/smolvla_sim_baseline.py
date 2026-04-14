import os
import torch
from pathlib import Path

# Disable WandB before importing lerobot
os.environ["WANDB_MODE"] = "disabled"

from lerobot.envs.factory import make_env
from lerobot.policies.factory import make_policy
from lerobot.scripts.evaluate import evaluate

def main():
    # 1. Define Paths and Settings
    policy_path = "HuggingFaceVLA/smolvla_libero"
    output_dir = Path("outputs/eval/smolvla_base/libero_spatial")
    
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Policy from: {policy_path}")
    
    # 2. Load the Policy (SmolVLA)
    # The factory will automatically download the weights from HF if needed
    policy = make_policy(
        pretrained_policy_name_or_path=policy_path,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    
    # Adjust policy generation parameters based on your CLI setup
    policy.n_action_steps = 10
    
    # 3. Create the LIBERO Environment
    # We configure it to save video frames during the evaluation
    print("Initializing LIBERO Environment...")
    env = make_env(
        env_type="libero",
        env_name="libero_spatial", # Maps to the --env.task argument
        video_backend="ffmpeg",
        # Ensures videos are saved to the output directory
        video_path=str(output_dir / "videos") 
    )

    # 4. Run the Evaluation
    print("Starting Evaluation...")
    eval_info = evaluate(
        env=env,
        policy=policy,
        n_episodes=5,
        batch_size=1,
        # Ensure we don't try to sync to cloud logging
        log_to_wandb=False 
    )

    print("\nEvaluation Complete!")
    print(f"Success Rate: {eval_info['success_rate'] * 100:.2f}%")
    print(f"Videos saved to: {output_dir / 'videos'}")

if __name__ == "__main__":
    main()