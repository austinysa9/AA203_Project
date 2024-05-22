import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from scipy.io import loadmat

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32) 

        # Initial and target states
        self.vine = np.array(loadmat('Model_AB_1.mat')['vine'])[0, 0]
        self.initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5 - self.vine], dtype=np.float32)
        self.state = self.initial_state.copy()
        self.goal = np.array([10, 0, 1.5, 0, 0, 0, 10, 0, 1.5 - self.vine], dtype=np.float32)

    def reset(self):
        self.state = self.initial_state.copy()
        return self.state
    
    def reward_function(self, current_position, target_position):
        distance_current_to_target = np.linalg.norm(current_position[:3] - target_position[:3])
        distance_next_to_target = np.linalg.norm(self.state[:3] - target_position[:3])

        reward = distance_current_to_target - distance_next_to_target

        if distance_next_to_target <= distance_current_to_target:
            reward += 100

        if distance_next_to_target > distance_current_to_target:
            reward -= 5

        return reward

    def step(self, action):
        # Load the data
        AB = np.array(loadmat('Model_AB_1.mat')['AB'])
        A = AB[:, 0:9]
        B = AB[:, 9:12]
        # Change A and B to float32
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        next_state = A @ self.state + B @ action
        
        reward = self.reward_function(self.state, self.goal)
        
        self.state = next_state.astype(np.float32)
        terminated = bool(np.linalg.norm(self.state[:3] - self.goal[:3]) < 0.1)
        truncated = False  # No specific truncation condition
        return self.state, reward, terminated, truncated, {}
    

env = DroneEnv()
original_env = env  # Save a reference to the original environment
#print("Initial state:", env.reset())

# Wrap the environment
env = DummyVecEnv([lambda: env])

model = DDPG.load("ddpg_drone_pendulum")

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
#print(path)

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
