import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32) 

        # Initial and target states
        self.initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        self.target_state = np.array([10, 0, 1.5, 0, 0, 0, 10, 0, 1], dtype=np.float32)

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
original_env = env  # Save a reference to the original environment
print("Initial state:", env.reset())

# Wrap the environment
env = DummyVecEnv([lambda: env])

model = PPO.load("ppo_drone2")

# Test the trained model and collect the path
obs = env.reset()
path = [obs[0]]  # Store initial observation
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    path.append(obs[0])  # Store observation at each step
    if dones:
        break

print("Final state:", obs)

# Convert path to a numpy array for easy slicing
path = np.array(path)

# Plot the path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract positions
x = path[:, 0]
y = path[:, 1]
z = path[:, 2]
x_tip = path[:, 6]
y_tip = path[:, 7]
z_tip = path[:, 8]
print(path)

# Plot the drone's center path
ax.plot(x, y, z, label='Drone Center Path')

# # Plot the bucket's path
# ax.plot(x_tip, y_tip, z_tip, label='Bucket Path', linestyle='--')

# Mark the initial and target positions
ax.scatter([original_env.initial_state[0]], [original_env.initial_state[1]], [original_env.initial_state[2]], color='green', label='Initial Position', s=100)
ax.scatter([original_env.target_state[0]], [original_env.target_state[1]], [original_env.target_state[2]], color='red', label='Target Position', s=100)

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Path from Initial to Target Position')
ax.legend()

plt.show()
