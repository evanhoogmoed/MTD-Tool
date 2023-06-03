import matplotlib.pyplot as plt
from BcEnv import *
import statistics



def arbitrary_randomization(network):
    #randomize one link aritrarily
    link = random.choice(list(network.edges))
    network.edges[link]['weight'] += random.randint(0,10)
    return network
    
def bc_links(network):
    #get top 5 betweenness centrality links
    bc_links = nx.edge_betweenness_centrality(network,weight='weight')
    bc_links = sorted(bc_links.items(), key=lambda x: x[1], reverse=True)[:5]
    
    #write bc_links as an array of arrays with format [source,target]
    for i in range(len(bc_links)):
        bc_links[i] = [bc_links[i][0][0],bc_links[i][0][1]]

    return bc_links 

    

def main():
    #Load the model
    model = PPO.load("ppo_500k_link1_bc")
    edge_env = BcEnv()
    #begin with the orginal network
    org_network = BcEnv().network

    #get top5 bc links
    bc_links_org = bc_links(org_network)
    print("org bc_links: ",bc_links_org)
    
    #randomize one link aritrarily
    random_delay = []
    bc_links_rand = []
    flow_rand = []
    for i in range(500):
        rand_network = arbitrary_randomization(org_network)

        #count how many bc links match the orginal network
        count = 0
        for link in bc_links_org:
            if link not in bc_links(rand_network):
                count += 1
        bc_links_rand.append(count)



    print("rand bc_links: ",bc_links_rand)
    

    
     #count the number of zeros, ones, twos, etc in bc_links_rand and store in dictionary
    bc_links_rand_dict = {0:0,1:0,2:0,3:0,4:0,5:0}
    for i in range(len(bc_links_rand)):
        if bc_links_rand[i] in bc_links_rand_dict:
            bc_links_rand_dict[bc_links_rand[i]] += 1
    
    print(bc_links_rand_dict)

  

    #PPO model
    PPO_count = []
    for i in range(500):
        obs = edge_env.reset()
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = edge_env.step(action)
    
   
        #convert obs to networkx graph
 
        PPO_network = nx.from_numpy_matrix(obs)
        #print(PPO_network.edges.data('weight'))

        #count how many bc links match the orginal network
        count = 0
        for link in bc_links_org:
            if link not in bc_links(PPO_network):
                count += 1
        PPO_count.append(count)


    #count the number of zeros, ones, twos, etc in PPO_count and store in dictionary
    PPO_count_dict = {0:0,1:0,2:0,3:0,4:0,5:0}
    for i in range(len(PPO_count)):
        if PPO_count[i] in PPO_count_dict:
            PPO_count_dict[PPO_count[i]] += 1
    
    print("PPO_count_dict: ",PPO_count_dict)


    X = np.arange(len(bc_links_rand_dict))
    ax = plt.subplot(111)
    ax.bar(X, bc_links_rand_dict.values(), width=0.2, color='#0F4392', align='center')
    ax.bar(X-0.2, PPO_count_dict.values(), width=0.2, color='#FF4949', align='center')
    ax.legend(('Random','PPO'))
    ax.set_ylabel('Number of Networks') 
    ax.set_xlabel('Number of BC links Changed')
    plt.xticks(X, bc_links_rand_dict.keys())
    plt.title("Change in BC links across 500 networks")
    plt.show()

if __name__ == "__main__":
    main()
