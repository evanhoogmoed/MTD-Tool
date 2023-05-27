import matplotlib.pyplot as plt
from BcEnv import *
import statistics


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
    #randomize one link aritrarily
    link = random.choice(list(network.edges))
    network.edges[link]['weight'] = random.randint(10,50)
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

    """# Plot the boxplot
    ax.boxplot(data)

    # Plot the horizontal line
    ax.axhline(y=line_value, color='r', linestyle='--')

    # Customize the plot
    ax.set_xticklabels(['Box 1', 'Box 2', 'Box 3'])
    ax.set_xlabel('Box')
    ax.set_ylabel('Value')
    ax.set_title('Boxplot with Horizontal Line')
    ax.legend(['Horizontal Line'])

    # Show the plot
    plt.show()"""

def main():
    #Load the model
    model = PPO.load("ppo_200k_link1")
    edge_env = BcEnv()



    #begin with the orginal network
    org_network = BcEnv().network

    flow =  nx.shortest_path(org_network,2,8)

    #sum weight of flow path
    org_flow_sum = 0
    for i in range(len(flow)-1):
        org_flow_sum += org_network[flow[i]][flow[i+1]]['weight']
    print("flow weight sum: ",org_flow_sum)



    #create packets
    packets = GenPackets()

    #get delay of packets on orginal network
    org_delay = Get_delay(org_network,packets)



    #get top5 bc links
    bc_links_org = bc_links(org_network)
    print("org bc_links: ",bc_links_org)
    
    #randomize one link aritrarily
    random_delay = []
    bc_links_rand = []
    flow_rand = []
    for i in range(100):
        rand_network = arbitrary_randomization(org_network)
        rand_delay = Get_delay(rand_network,packets)
        random_delay.append(sum(rand_delay)/len(rand_delay))

        #count how many bc links match the orginal network
        count = 0
        for link in bc_links_org:
            if link not in bc_links(rand_network):
                count += 1
        bc_links_rand.append(count)

        #get flow weight
        flow =  nx.shortest_path(rand_network,2,8)
        flow_sum = 0
        for i in range(len(flow)-1):
            flow_sum += rand_network[flow[i]][flow[i+1]]['weight']
        flow_rand.append(flow_sum)

    print("rand bc_links: ",bc_links_rand)
    
    #get average of flows
    rand_flow_avg = sum(flow_rand)/len(flow_rand)
    print("rand flow avg: ",rand_flow_avg)
    #print difference from average and flow
    print("rand flow avg diff: ",org_flow_sum - rand_flow_avg)



    print("rand bc_links Mode: ",statistics.mode(bc_links_rand))
    print("rand bc_links Avg: ",sum(bc_links_rand)/len(bc_links_rand))
  

    #get the best network graph from model
    obs = edge_env.reset()
    PPO_delays = []
    PPO_count = []
    PPO_flow = []
    for i in range(100):
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
            if link not in bc_links(PPO_network):
                count += 1
        PPO_count.append(count)

        #get flow weight
        flow =  nx.shortest_path(PPO_network,2,8)
        flow_sum = 0
        for i in range(len(flow)-1):
            flow_sum += PPO_network[flow[i]][flow[i+1]]['weight']
        PPO_flow.append(flow_sum)

    #get average of flows
    PPO_flow_avg = sum(PPO_flow)/len(PPO_flow)
    #print difference from average and flow
    print("PPO flow avg: ",PPO_flow_avg)
    print("PPO flow avg diff: ",org_flow_sum - PPO_flow_avg)
    print("PPO bc_links: ",PPO_count)

    #print the mode
    print("PPO bc_links Mode: ",statistics.mode(PPO_count))
    print("PPO bc_links Avg: ",sum(PPO_count)/len(PPO_count))

    GenBoxPlot(org_delay,rand_delay,PPO_delay)


if __name__ == "__main__":
    main()
