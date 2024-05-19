import numpy as np
from scipy.io import loadmat
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

# Custom environment class
class DroneEnv(gym.Env):
    def __init__(self, A, B, initial_state, target_state):
        super(DroneEnv, self).__init__()
        
        self.A = A
        self.B = B
        self.initial_state = initial_state
        self.target_state = target_state
        self.state = np.copy(initial_state)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(B.shape[1],), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(A.shape[0],), dtype=np.float32)

    def step(self, action):
        # Ensure action has correct shape
        action = np.reshape(action, (self.B.shape[1], 1))
        
        # Calculate the next state using the state-space model
        next_state = self.A @ self.state.reshape(-1, 1) + self.B @ action
        self.state = next_state.flatten()
        
        # Calculate the reward (negative distance to the target state)
        reward = -np.linalg.norm(self.state - self.target_state)
        
        # Check if the episode is done
        done = np.allclose(self.state, self.target_state, atol=0.1)
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.copy(self.initial_state)
        return self.state

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

# Load the data
AB = np.array(loadmat('Model_AB_1.mat')['AB'])
vine = np.array(loadmat('Model_AB_1.mat')['vine'])[0, 0]
A = AB[:, 0:9]
B = AB[:, 9:12]

initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine], dtype=np.float32)
target_state = np.array([10, 0, 1.5, 0, 0, 0, 10, 0, 1.5-vine], dtype=np.float32)

# Create the environment
env = make_vec_env(lambda: DroneEnv(A, B, initial_state, target_state), n_envs=1)

# Verify CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Create the PPO model with GPU support
model = PPO('MlpPolicy', env, verbose=1, device='cuda' if torch.cuda.is_available() else 'cpu')

# Train the model
model.learn(total_timesteps=300000)

# Save the model
model.save("ppo_drone")

# Load the model
model = PPO.load("ppo_drone")

# Test the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
