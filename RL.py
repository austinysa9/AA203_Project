import numpy as np
import random
import math
import Simulation as sim
import matplotlib.pyplot as plt
from collections import defaultdict


# Class for drone and drone vine state
class DroneState(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.x_tip = 0.0
        self.y_tip = 0.0
        self.z_tip = 0.0
        self.drone_state = [self.x, self.y, self.z, self.rot_x, self.rot_y, self.rot_z,self.x_tip, self.y_tip, self.z_tip]
        self.done = False
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.alpha = 0.8 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.actions = [self.generate_action() for _ in range(5)]  # Generate a list of 10 actions
        self.position = np.array([0.0, 0.0, 1.0 ])
        self.target = np.array([5.0, 0.0, 4.5])
        self.best_policy = {}
        self.path = [self.position.copy()] # Store the path of the drone

    # Create the Q-table
    def create_Q_table(self, state):
        state_key = tuple(state)  
        if state_key not in self.Q:
            self.Q[state_key] = {tuple(action): 0.0 for action in self.actions}


    def get_maxQ(self, state):
        state_key = tuple(state)  
        maxQ = -100000
        for action in self.Q[state_key]:
            if self.Q[state_key][action] > maxQ:
                maxQ = self.Q[state_key][action]
        return maxQ
    
    # Generate action of the drone: it will move in unit length based on the random theta value
    def generate_action(self):
        theta = random.uniform(0, 2*math.pi)
        ux = 0.1 * math.cos(theta)
        uz = 0.1 * math.sin(theta)
        return [ux, 0, uz]

    # Define the reward function
    def reward_function(self, current_position, next_position):
        distance_current_to_target = np.linalg.norm(current_position - self.target)
        distance_next_to_target = np.linalg.norm(next_position - self.target)
        
        reward = distance_current_to_target - distance_next_to_target
        
        if distance_next_to_target <= distance_current_to_target:  
            reward += 100  
        
        if distance_next_to_target > distance_current_to_target:
            reward -= 5  
        
        return reward
    
    # # Generate the next state of the drone using sim.calc
    # def generate_next_state(self, action):
    #     u = np.array(action).reshape(3, 1)
    #     x = sim.calc(u)
    #     next_state = x[:, -1]
    #     return next_state

    def generate_next_state(self, action):
        next_state = self.position + action
        return next_state

    # Update the Q-table
    def update_Q_table(self, state, action, reward, next_state):
        state_key = tuple(state)  # Convert state to a tuple
        next_state_key = tuple(next_state)  # Convert next_state to a tuple
        action_key = tuple(action)  # Ensure action is a tuple (should be by default if coming from generate_action)
        self.Q[state_key][action_key] += self.alpha * (reward + self.gamma * self.get_maxQ(next_state_key) - self.Q[state_key][action_key])

    # Choose the action based on epsilon-greedy policy
    def choose_action(self, state):
        state_key = tuple(state)  # Convert state to a tuple
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.Q[state_key], key=self.Q[state_key].get, default=random.choice(self.actions))

    # Use Q-learning to train the drone to get best policy from start to target position and store the policy
    def train(self, start, target, episodes=3):
        for _ in range(episodes):
            state = np.array(start)
            self.create_Q_table(state)
            while not self.done:
                action = self.choose_action(state)
                next_state = self.generate_next_state(action)
                reward = self.reward_function(state, next_state)
                self.update_Q_table(state, action, reward, next_state)
                state = next_state
                if np.linalg.norm(state - np.array(target)) < 0.1:
                    self.done = True
                self.epsilon = max(self.epsilon_min, self.epsilon - (self.epsilon_max - self.epsilon_min) * 0.01)
                print("Q_table", self.Q)
        self.extract_best_policy()

    # Extract the best policy from the Q-table        
    def extract_best_policy(self):
        for state in self.Q:
            self.best_policy[state] = max(self.Q[state], key=self.Q[state].get)

    # Use the policy to move the drone from start to target position
    def move_using_policy(self, start, target):
        state = np.array(start)
        while np.linalg.norm(state - np.array(target)) >= 0.1:
            action = self.best_policy.get(tuple(state), random.choice(self.actions))  # Use the best policy or random action if state not in policy
            state = self.generate_next_state(action)
            self.path.append(state.copy())

    # Plot the path of the drone
    def plot_path(self):
        x = [point[0] for point in self.path]
        y = [point[1] for point in self.path]
        z = [point[2] for point in self.path]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

position = np.array([0.0, 0.0, 1.0 ])  # Assuming drone starts at origin
target = np.array([5.0, 0.0, 4.5])   # Target position on X axis
drone = DroneState()
drone.train(position, target)
drone.move_using_policy(position, target)