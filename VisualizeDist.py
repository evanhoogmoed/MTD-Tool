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

def Get_delay(network,packets):
    all_delays = []
    for packet in packets:
        path = nx.shortest_path(network,packet.src,packet.dst)
        delay = 0
        for i in range(len(path)-1):
            delay += network[path[i]][path[i+1]]['weight']
        all_delays.append(delay)
    return all_delays

def GenBoxPlot(org_delay,rand_delay,A2C_delay):
    fig = plt.figure()
    fig.suptitle('Packet Delay Distribution')
    plt.boxplot([org_delay,rand_delay,A2C_delay])
    plt.xticks([1,2,3],['Org','Random','A2C'])
    plt.ylim([-5,300])
    plt.grid()
    plt.show()

def main():
    #Load the model
    model = PPO.load("ppo_500k_entropy01")
    edge_env = BcEnv()


    #begin with the orginal network
    org_network = BcEnv().network

    #create packets
    packets = GenPackets()

    #get delay of packets on orginal network
    org_delay = Get_delay(org_network,packets)
    print(org_delay)
"""
    org_delay = edge_env.net.delay(edge_env.net.nodeSets,edge_env.net.net)

    #create random delay
    print("Creating random delay")
    random_delay = edge_env.net.delay(edge_env.net.nodeSets,edge_env.net.randomize_link_weights(2,edge_env.net.net))

    obs = edge_env.reset()
    delay = []
    random = []
    print("Creating A2C delay")
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = edge_env.step(action)
        delay.append(info['delay'])
        #random.append(info['random_delay'])

    #flatten delay
    delay = np.array(delay, dtype=object)
    if delay.any():
        delay = np.hstack(delay)
    else:
        delay = delay.flatten()

    #flatten random
    random = np.array(random, dtype=object)
    if random.any():
        random = np.hstack(random)
    else:
        random = random.flatten()

    org_delay = np.array(org_delay, dtype=object)
    
    print("Creating boxplot")
    #create boxplot of delay
    GenBoxPlot(org_delay,random_delay,delay)"""


if __name__ == "__main__":
    main()