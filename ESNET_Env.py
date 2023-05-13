import networkx as nx
import simpy
import numpy as np
import pandas as pd
import random
import heapq
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

class GLOBAL: 
  LINK_WEIGHT_SET = []


#packet class to store packet information    
class Packet:
    def __init__(self, id, src, dst, delay, path):
        self.id = id
        self.src = src 
        self.dst = dst 
        self.delay = delay
        self.path = path


class NetworkGraph(object):
    def __init__(self):
        self.net = None
        self.nodeSets = None
        self.flow = None

        self.init_graph()
    
    def init_graph(self):
        self.net = self.EsnetGraph()
        self.nodeSets = self.SourceAndDestNode()
        #find the path between nodes 2 and 10
        self.flow = nx.shortest_path(self.net,source=2,target=10,weight='weight')

        

  
    #Constructs a networkx graph of 15 nodes from a sample of the ESNET network
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

    #hardcoded source and destination nodes, made into a function for future use 
    def SourceAndDestNode(self):
        src_nodes = [2,7,8,12]
        dst_nodes = [5,10,11,13]
        
        return src_nodes,dst_nodes

    #creates a set number of packets with random source and destination nodes and calculates paths
    def create_packets(self,network):
        packet_seed = np.random.default_rng(2021)
        packet_id = 1  
        packets = []

        for i in range(0,500):
            new_packet = Packet(packet_id,0,0,0,[])
            new_packet.src = packet_seed.choice(self.nodeSets[0])
            new_packet.dst = packet_seed.choice(self.nodeSets[1])

            path = nx.dijkstra_path(network,new_packet.src,new_packet.dst) 
            new_packet.path = path

            #add up all weights of the links in the path
            for i in range(0,len(path)-1):
                new_packet.delay += self.net[path[i]][path[i+1]]['weight']

            packet_id += 1
            packets.append(new_packet)

        return packets


    #randomizes the number of links specified in num_of_links
    def randomize_link_weights(self,num_of_links,network):
        net_copy = network.copy()

        for i in range(num_of_links):
            random_link = random.choice(list(net_copy.edges))
            net_copy.edges[random_link]['weight'] = random.choice(GLOBAL.LINK_WEIGHT_SET)
            
            #check if link weight changed
            #print(network.edges[random_link]['weight'],net_copy.edges[random_link]['weight'])

        return net_copy


    #state is all the links and their betweenness centrality
    def get_state(self,network):
        edge_bc_dict = nx.edge_betweenness_centrality(network,weight='weight')

        #change the dictionary into an array with format [src,dst,bc]
        state = []
        for key,value in edge_bc_dict.items():
            state.append([key[0],key[1],value])

        
        return state
    
    #calls the packet creator and collects the delays of all packets
    def delay(self,network):
        packets = self.create_packets(network)

        packet_delay = []
        for packet in packets:
            packet_delay.append(packet.delay)

        return packet_delay


    def GenBoxPlot(self,org_delay,rand_delay,graphname):
        fig = plt.figure()
        fig.suptitle('Packet Delay Distribution')
        plt.boxplot([org_delay,rand_delay])
        plt.xticks([1,2],['org',graphname])
        plt.ylim([0,300])
        plt.grid()
        plt.show()

        
       

#network = NetworkGraph()

#randomize_edges = network.randomize_link_weights(5,network.net)
#print(GLOBAL.LINK_WEIGHT_SET)
#get state of network 
#state = network.get_state(network.net)
#print(state)

"""#randomize network
randomize_edges = network.randomize_link_weights(5,network.net)
#get state of randomized network
rand_state = network.get_state(5,randomize_edges)
print(rand_state)"""


"""org_delay = network.delay(network.net)
randomize_edges = network.randomize_link_weights(5,network.net)
rand_delay = network.delay(randomize_edges)

network.GenBoxPlot(org_delay,rand_delay,'randomize_edges')"""
