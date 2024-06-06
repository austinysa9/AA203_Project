import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict, deque
import random
import scipy.io
from itertools import product
from scipy.io import loadmat
import time

AB = np.array(loadmat('Model_AB.mat')['AB'])
vine = np.array(loadmat('Model_AB.mat')['vine'])[0, 0]

class Environment:
    def __init__(self, target, AB, vine):
        self.target = np.array(target)
        self.AB = np.array(AB)  # Dynamics matrix
        self.state_history = deque(maxlen=5)
        self.reset(vine)

    def reset(self, vine):
        initial_states = [
            np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine]),  # t = 1
            np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine]),  # t = 2
            np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine]),  # t = 3
            np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine]),  # t = 4
            np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine])   # t = 5
        ]
        self.state_history.clear()
        for state in initial_states:
            self.state_history.appendleft(state)
        self.trajectory = [self.state_history[0][-3:].copy()]  # Start trajectory plotting from last of initial states
        return self.state_history[0]

    def step(self, action):
        next_state = self.transition(action)
        self.trajectory.append(next_state.copy())
        reward = self.reward_function(next_state)
        done = np.linalg.norm(next_state[-3:] - self.target) < 0.1
        self.state_history.appendleft(next_state)
        return next_state, reward, done

    def transition(self, action):
        extended_state_action = np.hstack([state for state in self.state_history] + [action])
        next_state = np.dot(self.AB, extended_state_action)
        return next_state

    def reward_function(self, state):
        distance_to_target = np.linalg.norm(state[-3:] - self.target)
        return -distance_to_target

    def plot_trajectory(self):
        traj = np.array(self.trajectory)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker='o')
        ax.scatter(*self.target, color='red', label='Target', s=100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

class Agent:
    def __init__(self, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, alpha=0.1, gamma=0.9):
        self.Q = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

    def compute_actions(self, state):
        x, y, z = state[:3]
        # actions_x = np.linspace(-10 + x, 10 + x, num=int((10 - (-10)) / 0.5) + 1)
        # actions_y = np.linspace(-10 + y, 10 + y, num=int((10 - (-10)) / 0.5) + 1)
        actions_x = np.linspace(-2 + x, 2 + x, num=int((2 - (-2)) / 0.5) + 1)
        actions_y = np.linspace(0, 0, 1)
        actions_z = np.linspace(0, 0, 1)
        return [np.array([ax, ay, az]) for ax, ay, az in product(actions_x, actions_y, actions_z)]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            actions = self.compute_actions(state)
            return random.choice(actions)
        else:
            state_key = tuple(state)
            if state_key in self.Q and self.Q[state_key]:
                return max(self.Q[state_key], key=self.Q[state_key].get, default=random.choice(self.compute_actions(state)))
            else:
                return random.choice(self.compute_actions(state))

    def learn(self, state, action, reward, next_state):
        current_q = self.Q[tuple(state)][tuple(action)]
        next_max_q = max(self.Q[tuple(next_state)].values(), default=0)
        self.Q[tuple(state)][tuple(action)] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train(agent, environment, episodes, save_path='control_inputs.mat'):
    control_inputs = []
    start_time = time.time()
    for episode in range(episodes):
        print(f"Episode {episode+1}/{episodes}")
        total_reward = 0
        state = environment.reset(vine)
        done = False
        while not done:
            action = agent.choose_action(state)
            control_inputs.append(action.tolist())
            next_state, reward, done = environment.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.update_epsilon()
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    total_runtime = time.time() - start_time
    print(f"Total training time: {total_runtime:.2f} seconds")
    
    scipy.io.savemat(save_path, {'control_inputs': control_inputs})
    environment.plot_trajectory()

# Setup and run training
target_position = [1.5, 0, 1.5-vine]
start_time = time.time()
environment = Environment(target_position, AB, vine)
agent = Agent()
end_time = time.time()  # Record the end time
total_runtime = end_time - start_time  # Calculate total runtime
train(agent, environment, 100, 'control_inputs.mat')
