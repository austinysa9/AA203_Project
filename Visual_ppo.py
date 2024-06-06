import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        # Observation space: [x_drone, x_tip, z_tip]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        # Action space: [x_drone]
        self.action_space = spaces.Box(low=-3, high=3, shape=(1,), dtype=np.float32)
        
        self.initial_state = np.array([0, 0, 0.5], dtype=np.float32)
        self.target_state = np.array([10, 10, 0.5], dtype=np.float32)
        
        # Define your system dynamics
        self.A = np.eye(3)
        self.B = np.zeros((3, 1))
        self.B[0, 0] = 1.0
        # self.A = np.array([[3.3997, -3.9060, 0.5256],
        #                    [0.8523, -1.3069, 0.2653],
        #                    [-2.0696, 2.1728, 0.9466]])
        # self.B = np.array([[-1.3964],
        #                    [-0.3582],
        #                    [0.6523]])

    def reset(self):
        self.state = self.initial_state.copy()
        return self.state

    def step(self, action):
        # Clip action to ensure it is within the defined action space
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        next_state = self.A @ self.state + self.B @ action
        reward = -np.linalg.norm(next_state - self.target_state)
        done = np.linalg.norm(next_state - self.target_state) < 0.001
        self.state = next_state
        return next_state, reward, done, {}

# Create environment
env = DroneEnv()
original_env = env  # Save a reference to the original environment

# Wrap the environment
env = DummyVecEnv([lambda: env])

# Train a new model
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=270000)

# Save the trained model
model.save("ppo_drone_3d")

# Test the trained model and collect the path and rewards
obs = env.reset()
path = [obs[0]]  # Store initial observation
rewards_list = []  # Store rewards at each step
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    path.append(obs[0])  # Store observation at each step
    rewards_list.append(rewards[0])  # Store reward at each step
    print('reward:', rewards)
    if dones:
        break

print("Final state:", obs)

# Convert path to a numpy array for easy slicing
path = np.array(path)

# Plot the path in 2D
plt.figure()
plt.plot(path[:, 1], path[:, 2], label='Drone tip Path')
plt.scatter(original_env.initial_state[1], original_env.initial_state[2], color='green', label='Initial Position', s=100)
plt.scatter(original_env.target_state[1], original_env.target_state[2], color='red', label='Target Position', s=100)
plt.xlabel('X Position')
plt.ylabel('Z Position')
plt.title('Drone tip Path from Initial to Target Position')
plt.legend()
plt.show()

# Plot rewards over time steps
plt.figure()
plt.plot(rewards_list)
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.title('Reward over Time Steps')
plt.show()
