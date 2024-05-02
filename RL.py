import numpy as np
import random
import math
import Animation
import Simulation as sim


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
        self.Q = dict()
        self.epsilon = 1.0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.01
        self.alpha = 0.8 # Learning rate
        self.gamma = 0.9 # Discount factor

    def build_state(self, state):

        state = str(int(round(state[0])))+'_'+str(int(round(state[1])))
            
        return state

    def get_maxQ(self, state):
        maxQ = -10000000
        for action in self.Q[state]:
            if self.Q[state][action] > maxQ:
                maxQ = self.Q[state][action]

        return maxQ