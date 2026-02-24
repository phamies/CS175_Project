"""

Usage:
    python train_boat.py
"""

import os
import signal
import time

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
from malmo_boat_env import MalmoBoatEnv


RUN_NAME = datetime.now().strftime("dqn_%Y%m%d_%H%M%S")

BASE_SAVE_DIR = "./models/"
BASE_LOG_DIR  = "./tensorboard_logs/"

SAVE_DIR = os.path.join(BASE_SAVE_DIR, RUN_NAME)
LOG_DIR  = os.path.join(BASE_LOG_DIR, RUN_NAME)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

ENV_CONFIG = {
    "mission_xml_path":  "boat_mission.xml",
    "millisec_per_tick": 20,    # 20 = 2.5x faster than normal Minecraft
    "num_tracks":        5,
}

DQN_CONFIG = dict(
    policy                 = "MlpPolicy",
    learning_rate          = 1e-3,
    buffer_size            = 50_000,
    learning_starts        = 1_000,
    batch_size             = 32,
    gamma                  = 0.99,
    target_update_interval = 1_000,
    exploration_fraction   = 0.2,
    exploration_final_eps  = 0.05,
    verbose                = 1,
    tensorboard_log        = LOG_DIR,
)

TOTAL_TIMESTEPS = 10_000
CHECKPOINT_FREQ = 2_000

# -----------------------------------------------------------------------------
# Graceful shutdown — saves model and waits for Malmo to reach DORMANT
# so the next script run doesn't collide with leftover server state
# -----------------------------------------------------------------------------

_env   = None
_model = None


def _shutdown(signum, frame):
    print("\n[Shutdown] Ctrl+C — saving and cleaning up...")
    if _model is not None:
        path = os.path.join(SAVE_DIR, "dqn_boat_interrupted")
        _model.save(path)
        print(f"[Shutdown] Saved → {path}")
    if _env is not None:
        _env.close()
        print("[Shutdown] Waiting 8s for Malmo server to go DORMANT...")
        time.sleep(8)
    print("[Shutdown] Safe to run next script.")
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# -----------------------------------------------------------------------------
# Env factory — same pattern as maze create_env(config)
# -----------------------------------------------------------------------------

def create_env(config):
    return MalmoBoatEnv(
        mission_xml_path  = config["mission_xml_path"],
        millisec_per_tick = config["millisec_per_tick"],
        num_tracks        = config["num_tracks"],
    )

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    global _env, _model

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Give Malmo time to clear after any previous run — same issue the
    # maze example avoids by always letting episodes finish cleanly
    print("Waiting 8s for Malmo server to be ready...")
    time.sleep(8)

    print("Initializing environment...")
    _env = create_env(ENV_CONFIG)

    print("Building DQN model...")
    _model = DQN(env=_env, **DQN_CONFIG)

    print(f"Tensorboard: tensorboard --logdir ./tensorboard_logs/")
    print(f"Starting DQN training for {TOTAL_TIMESTEPS:,} timesteps...")

    _model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = CheckpointCallback(
            save_freq   = CHECKPOINT_FREQ,
            save_path   = SAVE_DIR,
            name_prefix = "dqn_boat",
            verbose     = 1,
        ),
        tb_log_name = "dqn_boat",
    )

    path = os.path.join(SAVE_DIR, "dqn_boat_final")
    _model.save(path)
    print(f"Done. Model saved → {path}")
    _env.close()


if __name__ == "__main__":
    main()