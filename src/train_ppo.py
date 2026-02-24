"""
train_boat_ppo.py

PPO instead of DQN — better suited for ice boat physics because:
  - On-policy rollouts keep full action sequences together, so PPO sees
    "I turned -> slid -> overcorrected -> fell" as one coherent trajectory.
    DQN's replay buffer shuffles these steps apart and loses that context.
  - Ice boats have high momentum and delayed consequences. PPO's value
    function can learn "this turn will cost me in 5 steps" whereas DQN
    only bootstraps one step at a time.
  - Continuous-feeling control (steering while sliding) maps better to
    PPO's stochastic policy than DQN's epsilon-greedy discrete jumps.

Usage:
    python train_boat_ppo.py
"""

import os
import signal
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from malmo_boat_env import MalmoBoatEnv

RUN_NAME = datetime.now().strftime("ppo_%Y%m%d_%H%M%S")

SAVE_DIR = os.path.join("./models/", RUN_NAME)
LOG_DIR  = os.path.join("./tensorboard_logs/", RUN_NAME)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

ENV_CONFIG = {
    "mission_xml_path":  "boat_mission.xml",
    "millisec_per_tick": 20,
    "num_tracks":        5,
}

PPO_CONFIG = dict(
    policy      = "MlpPolicy",
    verbose     = 1,

    # Learning rate — PPO is less sensitive than DQN, 3e-4 is a reliable default
    learning_rate = 3e-4,

    # n_steps: steps collected per env before each update.
    # Larger = more context per update = better for momentum-heavy environments.
    # 2048 is standard. With slow Malmo this means ~2048 real steps before
    # any learning — patient but stable.
    n_steps = 2048,

    # batch_size: minibatch size within each update epoch.
    # Must divide n_steps evenly. 64 is standard.
    batch_size = 64,

    # n_epochs: how many times to reuse each rollout batch.
    # Higher = more sample efficiency but risks overfitting old data.
    n_epochs = 10,

    # gamma: discount factor. 0.99 values future rewards highly.
    # Good for a task where the goal (completing the track) is far away.
    gamma = 0.99,

    # gae_lambda: smoothing for advantage estimation.
    # 0.95 balances bias vs variance — standard PPO default.
    gae_lambda = 0.95,

    # ent_coef: entropy bonus encourages exploration.
    # 0.01 is a gentle push. Raise to 0.05 if agent stops exploring early.
    ent_coef = 0.01,

    # clip_range: how much the policy can change per update.
    # 0.2 is the standard PPO clip.
    clip_range = 0.2,

    tensorboard_log = LOG_DIR,
)

TOTAL_TIMESTEPS = 30_000
CHECKPOINT_FREQ = 2_500

# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------

_env   = None
_model = None


def _shutdown(signum, frame):
    print("\n[Shutdown] Ctrl+C -- saving and cleaning up...")
    if _model is not None:
        path = os.path.join(SAVE_DIR, "ppo_boat_interrupted")
        _model.save(path)
        print(f"[Shutdown] Saved -> {path}")
    if _env is not None:
        _env.close()
        print("[Shutdown] Waiting 8s for Malmo server to go DORMANT...")
        time.sleep(8)
    print("[Shutdown] Safe to run next script.")
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    global _env, _model

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Waiting 8s for Malmo server to be ready...")
    time.sleep(8)

    print("Initializing environment...")
    _env = MalmoBoatEnv(
        mission_xml_path  = ENV_CONFIG["mission_xml_path"],
        millisec_per_tick = ENV_CONFIG["millisec_per_tick"],
        num_tracks        = ENV_CONFIG["num_tracks"],
    )

    print("Building PPO model...")
    _model = PPO(env=_env, **PPO_CONFIG)

    print(f"Tensorboard: tensorboard --logdir ./tensorboard_logs/")
    print(f"Starting PPO training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Note: first update happens after {PPO_CONFIG['n_steps']:,} steps -- be patient.")

    _model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = CheckpointCallback(
            save_freq   = CHECKPOINT_FREQ,
            save_path   = SAVE_DIR,
            name_prefix = "ppo_boat",
            verbose     = 1,
        ),
        tb_log_name = "ppo_boat",
    )

    path = os.path.join(SAVE_DIR, "ppo_boat_final")
    _model.save(path)
    print(f"Done. Model saved -> {path}")
    _env.close()


if __name__ == "__main__":
    main()