import gym 
from pytorch_DQN import Agent 
# from utils import plotLearning
import numpy as np 
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import gym
from gym import Env
from gym.spaces import Discrete, Box
import os
from stable_baselines3 import DQN
import random



if __name__=="__main__":
    # env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    A_tank=10 #m^2
    qin=10 #m^3/min
    qout=20
    A_out_pipe=1
    g=9.8
    def height_model(x, t, action=0):
        h = x[0]
    #     dhdt = A_tank*(qin - action*A_out_pipe*(np.sqrt(2*g*h)))
        dhdt = (qin-action*qout)/A_tank
        return dhdt

    # class tank_env(Env):
    #     def __init__(self):
    #         self.action_space = Discrete(2)
    #         self.observation_space = Box(0, 100, shape=(1,))
    #         self.max_length = 1000
    #         self.state = 10 + random.randint(-2,2)
    #     def step(self, action):
    #         tn = np.linspace(0, 1, 2)
    #         sol = odeint(height_model, self.state, tn, args=(action,))
    #         self.state = sol[-1]
    #         self.max_length-=1
    #         reward=0
    #         if self.state<=0 or self.state>=100:
    #             done=True
    #         elif self.max_length<=0:
    #             done=True
    #         elif self.state<12 and self.state>8:
    #             done=False
    #             reward = + 1
    #         else:
    #             done=False
    #             reward = 0
    #         info={}
    #         return self.state, reward, done, info
            
    #     def render(self):
    #         pass
    #     def reset(self):
    #         self.state = np.array([10 + random.randint(-2,2)]).astype(float)
    #         self.max_length = 1000
    #         return self.state

    class tank_env(Env):
        def __init__(self, set_point):
            self.action_space = Discrete(2)
            self.observation_space = Box(0, 100, shape=(1,))
            self.max_length = 1000
            self.set_point = set_point
            self.state = np.array([10+random.randint(-2,2), self.set_point])
            
        def step(self, action):
            tn = np.linspace(0, 1, 2)
            sol = odeint(height_model, self.state[0], tn, args=(action,))
            self.state = np.array([sol[-1], self.set_point]).astype(dtype=float)
            self.max_length-=1
            reward=0
            if self.state[0]<=0 or self.state[0]>=100:
                done=True
            elif self.max_length<=0:
                done=True
            elif self.state[0]<(self.set_point+2) and self.state[0]>(self.set_point-2):
                done=False
                reward = + 1
            else:
                done=False
                reward = 0
            info={}
            return self.state, reward, done, info
            
        def render(self):
            pass
        def reset(self):
            self.state = np.array([10 + random.randint(-2,2), self.set_point]).astype(dtype=float)
            self.max_length = 1000
            return self.state

    env=tank_env(set_point=10.0)


    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=32, n_actions=2, eps_end=0.01, input_dims=[2], lr=0.001)
    scores, eps_history = [], []
    n_games = 100

    for i in range(n_games):
        score = 0
        done = False 
        observation = env.reset()
        
        while not done:
            env.render()
            action = agent.choose_action(observation) 
            observation_, reward, done, info = env.step(action)
            score+=reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print(f'episode{i}, scores{score}, average_score{avg_score}, epsilon{agent.epsilon}')
    
    x = [i+1 for i in range(n_games)]
