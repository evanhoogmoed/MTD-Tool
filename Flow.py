import gym
from gym import Env
from gym.spaces import Discrete, Box

import numpy as np
import random
import os
import networkx as nx
import pandas as pd 
import time
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

class GLOBAL:
    LINK_WEIGHT_SET = []
    LINK_DICT = {}
    

class FlowEnv(Env):
    def __init__(self):
        self.network = self.EsnetGraph()
        self.flow = nx.shortest_path(self.network,source=2,target=8,weight='weight')


        self.action_space = Box(low=0,high=17,shape=(14,))  #each number corresponds to a link to change [9 7 11]
        self.observation_space = Box(low=0, high=100, shape=(15,15)) #to_numpy_array(network) adjacency matrix
        self.state = 70 + random.randint(-3,3) #initial flow rate should be 70
        self.episode_length = 60 #episode length

    def step(self, action):
        #randomize the link weights
        self.network = self.randomize_link_weights(action,self.network)
        obs = nx.to_numpy_array(self.network, dtype=np.float32)
        self.state = self.get_flow_weight(self.network) 
        self.episode_length -= 1

        #if flow rate is 70, reward is 100 since it is our optimal flow rate
        if self.state == 70:
            reward = 1000

        #else for every 1 unit away from 70 subtract 10 from the reward
        else:
            reward = 100 - abs(70-self.state)


        
        if self.episode_length <= 0:
            done = True
        else:
            done = False
        
        info = {}

        return obs, float(reward), done, info

    def reset(self):
        self.state = np.array(70 + random.randint(-3,3))
        self.episode_length = 60
        network = self.EsnetGraph()
        obs = nx.to_numpy_array(network, dtype=np.float32)

        return obs

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
        
        #for every pair in esNet.edges() add to dictionary with key as int and value as tuple
        for index in range(len(esNet.edges())):
            GLOBAL.LINK_DICT[index] = list(esNet.edges())[index]


        return esNet
    
    def get_flow_weight(self,network):
        #get weight of path of net.net.flow
        path = nx.shortest_path(self.network,source=2,target=8,weight='weight')

        total_weight = 0
        for i in range(0,len(path)-1):
            total_weight += network[path[i]][path[i+1]]['weight']
    
        return total_weight
    
    def randomize_link_weights(self,action,network):
        net_copy = network.copy()


        for i in action:
            #round i to nearest int
            i = int(round(i))
            #look up i in dictionary to get the link
            change_link = GLOBAL.LINK_DICT[i]
            #change the weight of the link
            net_copy.edges[change_link]['weight'] = random.choice(GLOBAL.LINK_WEIGHT_SET)
            
            #check if link weight changed
            #print(network.edges[change_link]['weight'],net_copy.edges[change_link]['weight'])

        return net_copy
    


env = FlowEnv()
#check_env(env, warn=True)

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
env = Monitor(env, log_path, allow_early_resets=True)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)



model = PPO('MlpPolicy', env, verbose=1, ent_coef=0.001, tensorboard_log=log_path)
model.learn(total_timesteps=100000)
name = "ppo_14edge_flow_001entropy_100k_" + time.strftime("%m%d-%H%M")
model.save(name)

"""nx.draw(env.network, with_labels=True)
plt.show()

for u, v, w in env.network.edges(data='weight'):
    print(f"Edge ({u}, {v}) has weight: {w}")"""
