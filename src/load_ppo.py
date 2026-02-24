# Set this to the path of your .zip file to resume. Set to None to start fresh.
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
TOTAL_TIMESTEPS = 30_000
CHECKPOINT_FREQ = 2_500
# Set this to the path of your .zip file to resume. Set to None to start fresh.
LOAD_PATH = os.path.join("./models/ppo_20260224_023855\ppo_boat_interrupted.zip")


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


def main():
    global _env, _model

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Initializing environment...")
    _env = MalmoBoatEnv(
        mission_xml_path  = ENV_CONFIG["mission_xml_path"],
        millisec_per_tick = ENV_CONFIG["millisec_per_tick"],
        num_tracks        = ENV_CONFIG["num_tracks"],
    )

    if LOAD_PATH and os.path.exists(LOAD_PATH):
        print(f"Resuming training from: {LOAD_PATH}")
        _model = PPO.load(LOAD_PATH, env=_env)
        # Ensure the logger points to your new run directory
        _model.set_logger(None) 
    else:
        print("Building fresh PPO model...")
        _model = PPO(env=_env, **PPO_CONFIG)

    print(f"Starting PPO training...")
    _model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        reset_num_timesteps = False if LOAD_PATH else True,
        callback = CheckpointCallback(
            save_freq   = CHECKPOINT_FREQ,
            save_path   = SAVE_DIR,
            name_prefix = "ppo_boat",
            verbose     = 1,
        ),
        tb_log_name = "ppo_boat",
    )

    path = os.path.join(SAVE_DIR, "ppo_boat_final")
    _model.save(path)
    _env.close()

if __name__ == "__main__":
    main()