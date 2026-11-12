import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PongEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.width = 480
        self.height = 800
        self.paddle_width = 100
        self.ball_size = 12
        self.bob_speed = 6
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, -5, -6, -5, 0, 0], dtype=np.float32),
            high=np.array([468, 788, 6, 5, 380, 468], dtype=np.float32),
            dtype=np.float32
        )
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ball_x = self.np_random.integers(50, self.width - self.ball_size - 50)
        self.ball_y = (self.height - self.ball_size) // 2
        self.bob_x = self.np_random.integers(0, self.width - self.paddle_width)
        self.ball_speed_x = self.np_random.choice([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
        self.ball_speed_y = -5
        return self._get_observation(), {}
    def _get_observation(self):
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_speed_x,
            self.ball_speed_y,
            self.bob_x,
            self._get_landing_x()
        ], dtype=np.float32)
    def _get_landing_x(self):
        predicted_x = self.ball_x
        predicted_speed_x = self.ball_speed_x
        frames = int(np.ceil(
            (self.ball_y - 40) / abs(self.ball_speed_y)
        ))
        for _ in range(max(0,frames)):
            predicted_x += predicted_speed_x
            if predicted_x <= 0:
                predicted_x = 0
                predicted_speed_x *= -1
            elif predicted_x >= self.width - self.ball_size:
                predicted_x = self.width - self.ball_size
                predicted_speed_x *= -1
        return predicted_x
    def step(self, action):
        landing_x = self._get_landing_x()
        old_distance = abs(
            (landing_x + self.ball_size / 2)
            - (self.bob_x + self.paddle_width / 2)
        )
        if action == 0:
            self.bob_x -= self.bob_speed
        elif action == 2:
            self.bob_x += self.bob_speed

        self.bob_x = max(0, min(self.width - self.paddle_width, self.bob_x))
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y
        if self.ball_x <= 0 or self.ball_x >= self.width - self.ball_size:
            self.ball_speed_x *= -1

        new_distance = abs(
            (landing_x + self.ball_size / 2)
            - (self.bob_x + self.paddle_width / 2)
        )

        reward = (old_distance - new_distance) * 0.01
        terminated = False

        if (
            self.ball_y <= 40
            and self.ball_x + self.ball_size >= self.bob_x
            and self.ball_x <= self.bob_x + self.paddle_width
            and self.ball_speed_y < 0
        ):
            reward += 1
            terminated = True

        elif self.ball_y <= 40 and self.ball_speed_y < 0:
            reward -= 1
            terminated = True

        return self._get_observation(), reward, terminated, False, {}