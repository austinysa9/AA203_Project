import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Define the dynamic model
A = np.eye(9)  # Assuming identity matrix for simplicity; replace with actual A matrix
B = np.eye(9, 3)  # Assuming identity matrix for simplicity; replace with actual B matrix

class DroneState(object):
    def __init__(self):
        self.state = np.zeros(9)  # 9 states initialized to 0
        self.done = False
        self.Q = dict()
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.alpha = 0.8
        self.gamma = 0.9
        self.trajectory = []
        self.initial_state = np.array([0, 0, 1.5, 0, 0, 0, 0, 0, 1.5-0.6], dtype=np.float32)
        self.target_state = np.array([10, 0, 1.5, 0, 0, 0, 10, 0, 1.5-0.6], dtype=np.float32)
        self.reset()

    def build_state(self, state):
        state = '_'.join(map(str, map(int, np.round(state[:2]))))  # Using only x and y for state representation
        return state

    def get_maxQ(self, state):
        maxQ = -10000000
        for action in self.Q[state]:
            if self.Q[state][action] > maxQ:
                maxQ = self.Q[state][action]
        return maxQ 

    def createQ(self, state):
        if state not in self.Q.keys():
            self.Q[state] = self.Q.get(state, {'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0})
        return

    def choose_action(self, state):
        valid_actions = ['0', '1', '2', '3']
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
        else:
            actions = []
            maxQ = self.get_maxQ(state)
            for action in self.Q[state]:
                if self.Q[state][action] == float(maxQ):
                    actions.append(action)
            action = random.choice(actions)
        return action

    def learn(self, state, action, reward, next_state):
        maxQ_next_state = self.get_maxQ(next_state)
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (self.gamma * (reward + maxQ_next_state))
        return

    def reset(self):
        self.state = self.initial_state.copy()
        self.done = False
        self.trajectory = []  # Reset the trajectory for the new episode
        return self.state

    def step(self, action):
        u = np.zeros(3)
        if action == '0':
            u = [1, 0, 0]
        elif action == '1':
            u = [-1, 0, 0]
        elif action == '2':
            u = [0, 1, 0]
        elif action == '3':
            u = [0, -1, 0]

        u = np.array(u).reshape(3, 1)
        self.state = np.dot(A, self.state) + np.dot(B, u).flatten()

        reward = self.reward_function(self.state, self.target_state)
        if (self.state[0] < -0.05 or self.state[0] > 10.05 or self.state[1] < -0.05 or self.state[1] > 10.05):
            self.done = True

        # Append the current position to the trajectory
        self.trajectory.append(self.state[:3])

        return self.state, reward, self.done

    def reward_function(self, current_position, target_position):
        distance_current_to_target = np.linalg.norm(current_position[:3] - target_position[:3])
        distance_next_to_target = np.linalg.norm(self.state[:3] - target_position[:3])

        reward = distance_current_to_target - distance_next_to_target

        if distance_next_to_target <= distance_current_to_target:
            reward += 100

        if distance_next_to_target > distance_current_to_target:
            reward -= 5

        return reward

env = DroneState()

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    while not env.done:
        state = env.build_state(state)
        env.createQ(state)
        action = env.choose_action(state)
        env.epsilon = env.epsilon_min + (env.epsilon_max - env.epsilon_min) * (math.exp(-0.01 * episode))

        next_state, reward, done = env.step(action)
        total_reward += reward

        next_state_temp = env.build_state(next_state)
        env.createQ(next_state_temp)
        env.learn(state, action, reward, next_state_temp)
        state = next_state

    print(f"Reward in episode {episode}: {total_reward}")

# Extract Q values for visualization
grid_size = 11  # Adjust the grid size if necessary
q_values = np.zeros((grid_size, grid_size))  # Assuming the grid is 11x11 for simplicity

for state in env.Q:
    x, y = map(int, state.split('_'))
    if 0 <= x < grid_size and 0 <= y < grid_size:  # Ensure indices are within bounds
        maxQ = env.get_maxQ(state)
        q_values[x, y] = maxQ

# Visualization as a heat map
plt.figure(figsize=(10, 8))
plt.imshow(q_values, cmap='hot', interpolation='nearest')
plt.colorbar(label='Max Q value')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Q Table Heat Map')
plt.show()

# Plotting the drone trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 11])
ax.set_ylim([-1, 11])
ax.set_zlim([0, 2])

trajectory = np.array(env.trajectory)
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], marker='o')

plt.xlabel('X position')
plt.ylabel('Y position')
ax.set_zlabel('Z position')
plt.title('Drone Trajectory')
plt.show()
