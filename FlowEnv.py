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
    LINK_WEIGHT_SET = [10,20,30,40,50]

class BcEnv(Env):
    def __init__(self):
        self.network = self.EsnetGraph()
        
        #assign a special flow
        self.flow = nx.shortest_path(self.network,2,8)

        #get the weight of the flow
        self.flow_weight = 0
        for i in range(len(self.flow)-1):
            self.flow_weight += self.network[self.flow[i]][self.flow[i+1]]['weight']
        

        self.action_space = Discrete(10)  #how much to increase the weight of the link by 0-10
        self.observation_space = Box(low=0, high=50, shape=(15,15)) #adjacency matrix of the network with weights

        self.state = self.bc_links() #top 5 betweenness centrality links
        self.episode_length = 60 

    def step(self, action):
        prev_bc_links = self.state
        
        self.network = self.randomize_link_weights(action,self.network)

        #get weight of self.flow in new network
        new_flow_weight = 0
        for i in range(len(self.flow)-1):
            new_flow_weight += self.network[self.flow[i]][self.flow[i+1]]['weight']
        
        

        #get top bc links from new network 
        self.state = self.bc_links()

        self.episode_length -= 1

        reward = 0
        #for every link in self.state that is not in prev_bc_links, reward += 1
        for li in self.state:
            if li not in prev_bc_links:
                reward += 1
        
        #if all links are the same penalize heavily
        if reward == 0:
            reward = -100

        #if all links are different reward heavily
        if reward == 5:
            reward = 100

        #have a higher reward for minimal link weight changes and a lower reward for higher link weight changes
        reward -= action * 10 

        #if the flow weight has not changed reward heavily
        if new_flow_weight == self.flow_weight:
            reward += 100
        
        #for every point away from the flow weight reward -= 1
        reward -= abs(new_flow_weight - self.flow_weight)
            
        
        if self.episode_length <= 0:
            done = True
        else:
            done = False
        
        info = {}

        #get adjacency matrix of new network to set as observation
        obs = nx.to_numpy_array(self.network, dtype=np.float32)

        return obs, reward, done, info

    def reset(self):
        #reset network to original state
        self.network = self.EsnetGraph()
        self.state = self.bc_links()
        obs = nx.to_numpy_array(self.network, dtype=np.float32)
        self.episode_length = 60
        return obs

    def render(self):
        pass

    #construct the network using ESNET data
    def EsnetGraph(self):
        #connections2 is a reduced set of the ESNET network
        connections_df = pd.read_csv('connections2.csv')

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
            esNet.add_edge(connections_df['source'][index],connections_df['target'][index],weight=random.choice(GLOBAL.LINK_WEIGHT_SET))
        
        return esNet
    
    def bc_links(self):
        #get top 5 betweenness centrality links
        bc_links = nx.edge_betweenness_centrality(self.network,weight='weight')
        bc_links = sorted(bc_links.items(), key=lambda x: x[1], reverse=True)[:5]
        
        #write bc_links as an array of arrays with format [source,target]
        for i in range(len(bc_links)):
            bc_links[i] = [bc_links[i][0][0],bc_links[i][0][1]]

        return bc_links
    
    def randomize_link_weights(self,action,network):
        net_copy = network.copy()
        
        #get top bc links
        bc_links = self.bc_links()

        #find first link in bc_links whose weight is < 50:
        for link in bc_links:
            if network.edges[link]['weight'] < 50:
                random_link = link
                break
            else:
                random_link = bc_links[0]

        #Increase random_link weight by action
        net_copy.edges[random_link]['weight'] += action

        #do not allow link weight to exceed maximum link weight of org network
        if net_copy.edges[random_link]['weight'] > 50:
            net_copy.edges[random_link]['weight'] = 50



        return net_copy



def main():
    env = BcEnv()

    #used to ensure gym enviornment is compatible with stable baselines
    check_env(env, warn=True)

    #Uncomment the following section to test changes to the enviornment prior to training    
    """
    episodes = 5
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
    model = PPO('MlpPolicy', env,ent_coef=.01 ,verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=200000)
    name = "ppo_200k_entropy01"
    model.save(name)


if __name__ == '__main__':
    main()



