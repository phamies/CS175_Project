from stable_baselines import PPO2
from malmo_boat_env import MalmoBoatEnv

env = MalmoBoatEnv()
model = PPO2.load("boat_racer_model")

obs = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()
