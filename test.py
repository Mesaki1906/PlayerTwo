from stable_baselines3 import DQN
from bob import PongEnv
env = PongEnv()
model = DQN.load("bob_dqn_v5")

wins = 0
episodes = 100

for episode in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if reward > 0:
        wins += 1

print(f"Bob won {wins}/{episodes} rounds")
print(f"Win rate: {wins}%")