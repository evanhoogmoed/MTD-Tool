import gym
from gym import Env
from gym.spaces import Discrete, Box

import numpy as np
import random
import os
import networkx as nx
import pandas as pd 
import time
pd.options.mode.chained_assignment = None

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

class GLOBAL:
    LINK_WEIGHT_SET = []

class FlowEnv(Env):
    def __init__(self):
        self.network = self.EsnetGraph()
        self.flow = nx.shortest_path(self.network,source=2,target=10,weight='weight')

        self.action_space = Discrete(15)  #number of links to change
        self.observation_space = Box(low=0, high=100, shape=(15,15)) #to_numpy_array(network) adjacency matrix
        self.state = 40 + random.randint(-3,3) #initial flow rate should be 40
        self.episode_length = 60 #episode length

    def step(self, action):
        #randomize the link weights
        self.network = self.randomize_link_weights(action,self.network)

        self.state = self.get_flow_weight(self.network) 
        self.episode_length -= 1

        if self.state >= 39 and self.state <= 41:
            reward = 1
        else:
            reward = -1
        
        if self.episode_length <= 0:
            done = True
        else:
            done = False
        
        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = np.array(40, dtype=np.int32)
        self.episode_length = 60
        return self.state

    def render(self):
        pass

    def EsnetGraph(self):
        connections_df = pd.read_csv('connections2.csv')

        #convert weight which is in meters to light miliseconds in column wieght and round to nearest 5 to simplify the problem
        connections_df['weight'] = connections_df['weight'].apply(lambda x: round(x/2.9979e8*10000/5)*5)

        #get the full range of weights in the network
        GLOBAL.LINK_WEIGHT_SET = connections_df['weight'].unique()

        #create dictionary so each unique node name is paired with unique int ex. 'SEAT':0
        node_names = connections_df['source'].unique()
        node_dict = {}
        id = 0
        for node in node_names:
            node_dict[node] = id
            id += 1

        #replaces string node names with their ids for use in RL model
        for index in range(len(connections_df['source'])):
            connections_df['source'][index] = node_dict.get(connections_df['source'][index])
            connections_df['target'][index] = node_dict.get(connections_df['target'][index])

        esNet = nx.Graph()

        #add nodes to graph
        for node in connections_df['source']:
            esNet.add_node(node)
        
        #add edges to graph
        for index in range(len(connections_df['source'])):
            esNet.add_edge(connections_df['source'][index],connections_df['target'][index],weight=connections_df['weight'][index])
        
        return esNet
    
    def get_flow_weight(self,network):
        #get weight of path of net.net.flow
        path = self.flow
        total_weight = 0
        for i in range(0,len(path)-1):
            total_weight += network[path[i]][path[i+1]]['weight']
        return total_weight
    
    def randomize_link_weights(self,num_of_links,network):
        net_copy = network.copy()

        for i in range(num_of_links):
            random_link = random.choice(list(net_copy.edges))
            net_copy.edges[random_link]['weight'] = random.choice(GLOBAL.LINK_WEIGHT_SET)
            
            #check if link weight changed
            #print(network.edges[random_link]['weight'],net_copy.edges[random_link]['weight'])

        return net_copy
    


env = FlowEnv()

"""episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))"""

log_path = os.path.join('Training', 'Logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=1000000)
name = "ppo_15edge_flow_1M_" + time.strftime("%m%d-%H%M")
model.save(name)
