import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32) 

        # Initial and target states
        self.initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        self.target_state = np.array([0, 0, 1.5, 0, 0, 0, 10, 0, 1], dtype=np.float32)

    def reset(self):
        self.state = self.initial_state.copy()
        return self.state

    def step(self, action):
        # Apply action to the drone and update the state
        A = np.eye(9) 
        B = np.eye(9, 4)  
        next_state = A @ self.state + B @ action
        reward = -np.linalg.norm(next_state - self.target_state)  
        done = np.linalg.norm(next_state - self.target_state) < 0.1 
        self.state = next_state
        return next_state, reward, done, {}

env = DroneEnv()
print("Initial state:", env.reset())

# Wrap the environment
env = DummyVecEnv([lambda: env])

# Define and train the PPO model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500000)

# Save the model
model.save("ppo_drone")

# Load and test the model
model = PPO.load("ppo_drone")

# Test the trained model
obs = env.reset()
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
print("Final state:", obs)
