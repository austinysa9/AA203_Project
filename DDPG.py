import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from scipy.io import loadmat

class DronePendulumEnv(gym.Env):
    def __init__(self):
        super(DronePendulumEnv, self).__init__()
        # Action space: [dx, dy, dz] position changes for the drone
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # Observation space: [x, y, z, rotation_x, rotation_y, rotation_z, x_tip, y_tip, z_tip]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        # Initial state
        vine = np.array(loadmat('Model_AB_1.mat')['vine'])[0, 0]
        self.initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5 - vine], dtype=np.float32)
        self.state = self.initial_state.copy()
        self.goal = np.array([10, 10, 1.5, 0, 0, 0, 10, 10, 1.5 - vine], dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.initial_state.copy().astype(np.float32)
        return self.state,{}
    
    def step(self, action):
        # Load the data
        AB = np.array(loadmat('Model_AB_1.mat')['AB'])
        A = AB[:, 0:9]
        B = AB[:, 9:12]
        # Change A and B to float32
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        next_state = A @ self.state + B @ action
        
        reward = -float(np.linalg.norm(next_state[:3] - self.goal[:3]))  # Ensure reward is a Python float
        
        self.state = next_state.astype(np.float32)
        terminated = bool(np.linalg.norm(self.state[:3] - self.goal[:3]) < 0.1)
        truncated = False  # No specific truncation condition
        return self.state, reward, terminated, truncated, {}
    
    # def reward_function(self, current_position, target_position):
    #     distance_current_to_target = np.linalg.norm(current_position[:3] - target_position[:3])
    #     distance_next_to_target = np.linalg.norm(self.state[:3] - target_position[:3])

    #     reward = distance_current_to_target - distance_next_to_target

    #     if distance_next_to_target <= distance_current_to_target:
    #         reward += 100

    #     if distance_next_to_target > distance_current_to_target:
    #         reward -= 5

    #     return reward
    
    def render(self, mode='human'):
        pass

# Create a custom callback
class ProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Step: {self.n_calls}, Time Step: {self.num_timesteps}")
        return True

# Create the custom environment
env = DronePendulumEnv()
original_env = env  # Save a reference to the original environment

# Check the environment
check_env(env)

# Wrap the environment with Monitor and DummyVecEnv
env = Monitor(env)
env = DummyVecEnv([lambda: env])

# Create the DDPG model
model = DDPG('MlpPolicy', env, verbose=1)

# Train the model with the custom callback
model.learn(total_timesteps=1000, callback=ProgressCallback(check_freq=100))

# Save the model
model.save("ddpg_drone_pendulum")

# Load the model
model = DDPG.load("ddpg_drone_pendulum")

# Test the trained model
obs = env.reset()  # Adjusted for vectorized environment
path = [obs[0]]  # Ensure to copy to avoid referencing issues
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    path.append(obs[0].copy())  # Ensure to copy and extract the first element
    if dones[0]:  # Extracting the first element
        break

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

# Print the final state
initial_state = original_env.initial_state[0:3]
final_state = original_env.goal[6:9]


print("Actual Final State:", path[-1])

# Plot the initial and final states
ax.scatter(*initial_state, color='red', label='Initial State')
ax.scatter(*final_state, color='green', label='Final State')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('Drone Path from Initial to Target Position')
ax.legend()

plt.show()


