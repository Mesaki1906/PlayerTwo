from stable_baselines3 import DQN
from bob import PongEnv

env = PongEnv()

model = DQN(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=150000)
model.save("bob_dqn_v5")