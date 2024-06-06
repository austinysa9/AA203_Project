import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from scipy.io import loadmat


AB = np.array(loadmat('Model_AB_1_new.mat')['AB'])
vine = np.array(loadmat('Model_AB_1_new.mat')['vine'])[0, 0]
A = AB[:, 0:9]
B = AB[:, 9:12]

# A = np.eye(9)
# B = np.zeros((9, 3))

# # Fill in the diagonal elements to create identity matrices in 3x3 blocks
# for i in range(3):
#     B[3*i:3*(i+1), i] = 1

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32) 

        # Initial and target states
        self.initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine], dtype=np.float32)
        self.target_state = np.array([10, 0, 1.5, 0, 0, 0, 10, 0, 1.5-vine], dtype=np.float32)

    def reset(self):
        self.state = self.initial_state.copy()
        return self.state

    def step(self, action):
        # Apply action to the drone and update the state 
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

# # Load the models
# model1 = PPO.load("ppo_drone_1")
# model2 = PPO.load("ppo_drone_2")  # Replace with your second model path

# # Function to collect the path
# def collect_path(model, env):
#     obs = env.reset()
#     path = [obs[0]]  # Store initial observation
#     for _ in range(100):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         path.append(obs[0])  # Store observation at each step
#         if dones:
#             break
#     return np.array(path)

# # Collect paths
# path1 = collect_path(model1, env)
# path2 = collect_path(model2, env)

# # Plot the paths
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Extract positions for the first path
# x1 = path1[:, 0]
# y1 = path1[:, 1]
# z1 = path1[:, 2]
# x1_tip = path1[:, 6]
# y1_tip = path1[:, 7]
# z1_tip = path1[:, 8]

# # Extract positions for the second path
# x2 = path2[:, 0]
# y2 = path2[:, 1]
# z2 = path2[:, 2]
# x2_tip = path2[:, 6]
# y2_tip = path2[:, 7]
# z2_tip = path2[:, 8]

# # Plot the drone's center path for both models
# ax.plot(x1, y1, z1, label='Drone 1 Center Path')
# ax.plot(x2, y2, z2, label='Drone 2 Center Path')

# # Plot the bucket's path for both models
# ax.plot(x1_tip, y1_tip, z1_tip, label='Drone 1 Bucket Path', linestyle='--')
# ax.plot(x2_tip, y2_tip, z2_tip, label='Drone 2 Bucket Path', linestyle='--')

# # Mark the initial and target positions (assuming the same for both models)
# ax.scatter([original_env.initial_state[0]], [original_env.initial_state[1]], [original_env.initial_state[2]], color='green', label='Initial Position', s=100)
# ax.scatter([original_env.target_state[0]], [original_env.target_state[1]], [original_env.target_state[2]], color='red', label='Target Position', s=100)

# # Set labels and title
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Z Position')
# ax.set_title('Comparative Drone Paths from Initial to Target Position')
# ax.legend()

# plt.show()


model = PPO.load("ppo_drone")

# Test the trained model and collect the path
obs = env.reset()
path = [obs[0]]  # Store initial observation
for i in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    path.append(obs[0])  # Store observation at each step
    if dones:
        break


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


# Plot the drone's center path
ax.plot(x, y, z, label='Drone Center Path')

# Plot the bucket's path
ax.plot(x_tip, y_tip, z_tip, label='Bucket Path', linestyle='--')

# Mark the initial and target positions
ax.scatter([original_env.initial_state[0]], [original_env.initial_state[1]], [original_env.initial_state[2]], color='green', label='Initial Position', s=100)
ax.scatter([original_env.target_state[0]], [original_env.target_state[1]], [original_env.target_state[2]], color='red', label='Target Position', s=100)


ax.scatter([0], [0], [1.5])
ax.scatter([10], [0], [1.5])       
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Path from Initial to Target Position')
ax.legend()

plt.show()
