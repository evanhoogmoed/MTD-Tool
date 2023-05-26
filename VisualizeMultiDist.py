import matplotlib.pyplot as plt
from BcEnv import *


class Packet:
    def __init__(self, id, src, dst, delay, path):
        self.id = id
        self.src = src 
        self.dst = dst 
        self.delay = delay
        self.path = path

def GenPackets():
    packets = []
    for i in range(100):
        src = random.randint(0,14)
        dst = random.randint(0,14)
        while dst == src:
            dst = random.randint(0,14)

        packets.append(Packet(i,src,dst,0,[]))
    return packets

def arbitrary_randomization(network):
    #randomize all links
    for link in network.edges:
        network.edges[link]['weight'] = random.randint(0,50)
    return network
    
def bc_links(network):
    #get top 5 betweenness centrality links
    bc_links = nx.edge_betweenness_centrality(network,weight='weight')
    bc_links = sorted(bc_links.items(), key=lambda x: x[1], reverse=True)[:5]
    
    #write bc_links as an array of arrays with format [source,target]
    for i in range(len(bc_links)):
        bc_links[i] = [bc_links[i][0][0],bc_links[i][0][1]]

    return bc_links 

def Get_delay(network,packets):
    all_delays = []
    for packet in packets:
        path = nx.shortest_path(network,packet.src,packet.dst)
        delay = 0
        for i in range(len(path)-1):
            delay += network[path[i]][path[i+1]]['weight']
        all_delays.append(delay)
    return all_delays

def GenBoxPlot(org_delay,rand_delay,PPO_delay):
    fig = plt.figure()
    fig.suptitle('Packet Delay Distribution')
    plt.boxplot([org_delay,rand_delay,PPO_delay])
    plt.xticks([1,2,3],['Org','Random','PPO'])
    plt.ylim([-5,300])
    plt.grid()
    plt.show()

def main():
    #Load the model
    model = PPO.load("models/ppo_500k_entropy01")
    edge_env = BcEnv()


    #begin with the orginal network
    org_network = BcEnv().network

    #create packets
    packets = GenPackets()

    #get delay of packets on orginal network
    org_delay = Get_delay(org_network,packets)

    #get the average delay of the orginal network
    print("org delay average: ",sum(org_delay)/len(org_delay))


    #get top5 bc links
    bc_links_org = bc_links(org_network)
    print("org bc_links: ",bc_links_org)
    
    #randomize one link aritrarily
    random_delay = []
    bc_links_rand = []
    for i in range(50):
        rand_network = arbitrary_randomization(org_network)
        rand_delay = Get_delay(rand_network,packets)
        random_delay.append(sum(rand_delay)/len(rand_delay))

        #count how many bc links match the orginal network
        count = 0
        for link in bc_links_org:
            if link in bc_links(rand_network):
                count += 1
        bc_links_rand.append(count)

    

    #get top5 bc links
    #bc_links_rand = bc_links(rand_network)
    #print("rand bc_links: ",bc_links_rand)

    #get the best network graph from model
    obs = edge_env.reset()
    PPO_delays = []
    PPO_count = []
    for i in range(50):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = edge_env.step(action)
    
   
        #convert obs to networkx graph
        PPO_network = nx.from_numpy_matrix(obs)
        PPO_delay = Get_delay(PPO_network,packets)
        PPO_delays.append(sum(PPO_delay)/len(PPO_delay))

        #count how many bc links match the orginal network
        count = 0
        for link in bc_links_org:
            if link in bc_links(PPO_network):
                count += 1
        PPO_count.append(count)

    #get top5 bc links
    #bc_links_ppo = bc_links(PPO_network)
    #print("PPO bc_links: ",bc_links_ppo)

    GenBoxPlot(org_delay,rand_delay,PPO_delay)


if __name__ == "__main__":
    main()
