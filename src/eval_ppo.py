# eval_ppo.py
import time
from stable_baselines import PPO2
from malmo_boat_env import MalmoBoatEnv

env = MalmoBoatEnv()  # Make sure it renders visuals in Malmo
model = PPO2.load("ppo_iceboat", env=env)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    time.sleep(0.05)  # Slow it down so you can watch

env.close()
