from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import DummyVecEnv
from malmo_boat_env import MalmoBoatEnv

# Wrap env
env = DummyVecEnv([lambda: MalmoBoatEnv()])

# Callback to log rewards and actions
class RewardLogger(BaseCallback):
    def __init__(self, verbose=1):
        super(RewardLogger, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            reward = self.locals.get('rewards')
            action = self.locals.get('actions')
            print(f"Step: {self.n_calls}, action: {action}, reward: {reward}")
        return True

# Create PPO2 model with TensorBoard logging
model = PPO2(
    MlpPolicy,
    env,
    verbose=1,
    tensorboard_log="./ppo_boat_tensorboard/",
    learning_rate=3e-4,
    n_steps=1024,
    gamma=0.99,
    lam=0.95,
    ent_coef=0.01
)

# Train a short run for monitoring
model.learn(total_timesteps=20000, callback=RewardLogger())
