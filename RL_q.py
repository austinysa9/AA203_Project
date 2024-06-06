import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict, deque
import random
from itertools import product
from scipy.io import loadmat
import time

# Load the data from the mat file
AB = np.array(loadmat('Model_AB.mat')['AB'])
vine = np.array(loadmat('Model_AB.mat')['vine'])[0, 0]

class Environment:
    def __init__(self, target, AB):
        self.target = np.array(target)
        self.AB = np.array(AB)  # Combined dynamics matrix (9 x 48)
        self.state_history = deque(maxlen=5)  # Store the last 5 states
        self.reset()

    def reset(self):
        initial_states = [
            np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-vine]) for _ in range(5)
        ]
        self.state_history.clear()
        for state in initial_states:
            self.state_history.appendleft(state)
        self.trajectory = [self.state_history[0][-3:].copy()]
        return self.state_history[0]

    def step(self, action):
        next_state = self.transition(action)
        self.trajectory.append(next_state[-3:].copy())
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
    def __init__(self, actions):
        self.Q = defaultdict(lambda: defaultdict(float))
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.9
        self.actions = actions

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[tuple(state)], key=self.Q[tuple(state)].get, default=random.choice(self.actions))

    def learn(self, state, action, reward, next_state):
        current_q = self.Q[tuple(state)][tuple(action)]
        next_max_q = max(self.Q[tuple(next_state)].values(), default=0)
        self.Q[tuple(state)][tuple(action)] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def print_q_values(self, state=None):
        if state is not None:
            print(f"Q-values for the state {state}:")
            for action, q_value in self.Q[tuple(state)].items():
                print(f"Action: {action}, Q-value: {q_value}")
        else:
            print("Top Q-values across all states:")
            top_q_values = sorted(((state, action, q) for state, actions in self.Q.items() for action, q in actions.items()), key=lambda x: x[2], reverse=True) # [:10]
            for state, action, q_value in top_q_values:
                print(f"State: {state}, Action: {action}, Q-value: {q_value}")

def train(agent, environment, episodes):
    for episode in range(episodes):
        total_reward = 0
        state = environment.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        agent.update_epsilon()
        print(f"Episode {episode+1}: Total Reward = {total_reward}")
    agent.print_q_values()
    environment.plot_trajectory()

# Define possible actions
action_range = np.linspace(-10, 10, num=int((10 - (-10)) / 0.1) + 1)
actions = [np.array([x, y, z]) for x, y, z in product(action_range, repeat=3)]

# Initialize environment and agent
target_position = [2, 0, 1.5-vine]
environment = Environment(target_position, AB)
agent = Agent(actions)

start_time = time.time()
# Train the agent
train(agent, environment, 100)
end_time = time.time()
total_runtime = end_time - start_time
print(f"Total training time: {total_runtime:.2f} seconds")

