import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
from tqdm import tqdm

class DiscretizedDroneEnv(gym.Env):
    def __init__(self, n_bins=5):
        super(DiscretizedDroneEnv, self).__init__()
        self.n_bins = n_bins

        # Action space: discretized [dx, dy, dz] position changes for the drone
        self.action_space = spaces.Discrete(n_bins**3)

        # Observation space: discretized [x, y, z, rotation_x, rotation_y, rotation_z, x_tip, y_tip, z_tip]
        self.observation_space = spaces.Discrete(n_bins**9)
        
        # Initial state
        vine = np.array(loadmat('Model_AB_1.mat')['vine'])[0, 0]
        self.initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5 - vine], dtype=np.float32)
        self.state = self.initial_state.copy()
        self.goal = np.array([10, 10, 1.5, 0, 0, 0, 10, 10, 1.5 - vine], dtype=np.float32)
        
        # Load model matrices
        AB = np.array(loadmat('Model_AB_1.mat')['AB'])
        self.A = AB[:, 0:9].astype(np.float32)
        self.B = AB[:, 9:12].astype(np.float32)
        
        # Define state and action bins
        self.state_bins = [np.linspace(-10, 10, self.n_bins) for _ in range(9)]
        self.action_bins = [np.linspace(-1, 1, self.n_bins) for _ in range(3)]

    def discretize_state(self, state):
        """Discretizes a continuous state."""
        state_idx = [np.digitize(s, bins) - 1 for s, bins in zip(state, self.state_bins)]
        return tuple(state_idx)

    def discretize_action(self, action):
        """Discretizes a continuous action."""
        action_idx = [np.digitize(a, bins) - 1 for a, bins in zip(action, self.action_bins)]
        return tuple(action_idx)

    def continuous_action(self, action_idx):
        """Converts a discrete action index to a continuous action."""
        return np.array([self.action_bins[i][idx] for i, idx in enumerate(action_idx)], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.initial_state.copy().astype(np.float32)
        return self.discretize_state(self.state), {}
    
    def step(self, action_idx):
        action = self.continuous_action(np.unravel_index(action_idx, (self.n_bins,) * 3))
        next_state = self.A @ self.state + self.B @ action
        
        reward = -np.linalg.norm(next_state[:3] - self.goal[:3])
        
        self.state = next_state.astype(np.float32)
        terminated = bool(np.linalg.norm(self.state[:3] - self.goal[:3]) < 0.1)
        truncated = False  
        return self.discretize_state(self.state), reward, terminated, truncated, {}

# Q-learning implementation
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, n_bins=5):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(float)
        self.n_bins = n_bins

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_actions = [self.q_table[state + (a,)] for a in range(self.env.action_space.n)]
            return np.argmax(state_actions)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax([self.q_table[next_state + (a,)] for a in range(self.env.action_space.n)])
        td_target = reward + self.gamma * self.q_table[next_state + (best_next_action,)]
        td_error = td_target - self.q_table[state + (action,)]
        self.q_table[state + (action,)] += self.alpha * td_error

    def train(self, n_episodes=500):
        for episode in tqdm(range(n_episodes), desc="Training Progress"):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.update_q_value(state, action, reward, next_state)
                state = next_state
                done = terminated or truncated

    def test(self, env, n_steps=100):
        state, _ = env.reset()
        path = [env.state]
        for _ in range(n_steps):
            action = self.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            path.append(env.state)
            state = next_state
            if terminated or truncated:
                break
        return path

# Initialize environment and agent
env = DiscretizedDroneEnv(n_bins=5)
agent = QLearningAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1, n_bins=5)

# Train the agent
agent.train(n_episodes=500)

# Test the agent
path = agent.test(env, n_steps=100)

# Convert path to numpy array for plotting
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
initial_state = env.initial_state[0:3]
final_state = env.goal[0:3]

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
