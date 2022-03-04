import numpy as np
import matplotlib.pyplot as plt 
from IPython.display import Math, Latex # for latex equations
from IPython.core.display import Image # for displaying images
from scipy import stats
from scipy import optimize
import networkx as nx
import pandas as pd
from pyvis.network import Network
import collections
import itertools
import altair as alt


# Load data from the respective csv and construct a static graph/network of plant tissue with the state 'state'

def create_tissue(csv_path,edge_weight):
    
    csv_df = pd.read_csv(csv_path) 
    
    g = nx.Graph()
    node_list = list(sorted(set(csv_df.iloc[:,0])))
    g.add_nodes_from(node_list)
    for i in range(len(csv_df)):
        g.add_edge(csv_df.iloc[i,0],csv_df.iloc[i,1])
    initial_state = np.zeros(len(g.nodes()), dtype = np.int8)
    assert len(initial_state) == len(g.nodes())
    nx.set_node_attributes(g, dict(zip(g.nodes(), initial_state)), name="state") 
    if edge_weight == True:
        weights = [csv_df.iloc[i,2] for i in range(len(csv_df))]
        nx.set_edge_attributes(g, dict(zip(g.edges(), weights)), "weight")
    
    return g

def plt_tissue(g, edge_weight, save_with_name):
    plt.figure(figsize = (10,10))
    pos = nx.kamada_kawai_layout(g)
    color_list = []
    for x in g.nodes():
        if g.nodes[x]["state"] == 0:
            color_list.append("black")
        if g.nodes[x]["state"] == 1:
            color_list.append("white")
            
    nx.draw_networkx_nodes(g,pos,node_color= color_list, edgecolors="black")
    if edge_weight == False:
        nx.draw_networkx_edges(g,pos,edge_color = 'blue')
        
    if edge_weight == True:
        all_weights = []
        for (node1,node2,data) in g.edges(data=True):
            all_weights.append(data["weight"])
        unique_weights = list(set(all_weights))

        for w in unique_weights:
            weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in g.edges(data=True) if edge_attr['weight']==w]
            nx.draw_networkx_edges(g,pos,edge_color = 'blue',edgelist=weighted_edges,width=0.3*w)
    if save_with_name != None:
        plt.savefig(save_with_name)
    
    plt.show()

# Update the network; basically the state of the network according to the rule-3 (r3) which corresponds with Doug's model-3 

def update_rule3(g, temp, noise_cold, noise_warm): 
    for x in list(g.nodes()):
        if temp == 0 and g.nodes[x]["state"] == 0:
            c1 = np.random.choice(['noise','no_noise'], p = [noise_cold,1-noise_cold])
            if c1 == 'noise':
                g.nodes[x]["state"] = 1
            if c1 == 'no_noise':
                neighbor_states = np.array([g.nodes[y]["state"] for y in list(g.neighbors(x))])
                if np.sum(neighbor_states)/len(neighbor_states) >= 0.5:
                    g.nodes[x]["state"] = 1

        if temp == 1 and g.nodes[x]["state"] == 1:
            c2 = np.random.choice(['noise','no_noise'], p = [noise_warm,1-noise_warm])
            if c2 == 'noise':
                g.nodes[x]["state"] = 0
                
    return g          
    
def update_spontaneous(g, jump_state):
    if jump_state == "default":
        jump = np.zeros((len(g.nodes()),),dtype = int)
    else:
        assert len(g.nodes()) == len(jump_state)
        jump = jump_state
    for i in range(len(g.nodes())):
        g.nodes[list(g.nodes())[i]]["state"] = jump[i]
        


def trajectory(g, temp_sch, noise_cold, noise_warm): # temp schedule is the list of cold (0), warm(1) schedules 
    
    temp_array = np.array([],dtype = int)         
    for i in range(len(temp_sch)):
        if int(i/2)*2  == i: 
            to_append = np.zeros(temp_sch[i], dtype = int)
        else:
            to_append = np.ones(temp_sch[i], dtype = int)
        temp_array = np.append(temp_array, to_append)
        
    
    trajectory = np.empty([len(temp_array)+1 , len(g.nodes())],dtype = int)
    g_0 = g
    trajectory[0] = np.array([g_0.nodes[j]["state"] for j in g_0.nodes()])
    for k in range(len(temp_array)):
        g_0 = update_rule3(g_0, temp_array[k], noise_cold, noise_warm)
        trajectory[k+1] = np.array([g_0.nodes[j]["state"] for j in g_0.nodes()])
        
    time_array = np.arange(len(temp_array)+1)
    avg_exp = []
    for i in range(len(trajectory)):
        avg_exp.append(np.sum(trajectory[i])*100/len(trajectory[0]))
    expression_level = np.array(avg_exp)
    
    trajectory_df = pd.DataFrame(trajectory, columns = ['node{}'.format(x) for x in g.nodes()])
    trajectory_df.insert(0, "Time_step", time_array)
    trajectory_df.insert(1, "expression_level", expression_level)
    return trajectory_df
    

def ensemble(g, temp_sch, noise_cold, noise_warm, ensemble_size):
    ensemble_data = trajectory(g, temp_sch, noise_cold, noise_warm).iloc[:,[0,1]]
    ensemble_data.rename(columns={"FLC_off_level":"sim_1"} ,inplace=True)
    for i in range(ensemble_size-1):
        update_spontaneous(g, "default")
        traj = trajectory(g, temp_sch, noise_cold, noise_warm).iloc[:,1]
        ensemble_data.insert(i+2,"sim_{}".format(i+2), traj)
    
    ensemble_data['mean'] = ensemble_data.iloc[:,1:ensemble_size].mean(axis=1)
    ensemble_data['std'] = ensemble_data.iloc[:,1:ensemble_size].std(axis=1)
    ensemble_data['upper'] = ensemble_data.iloc[:,ensemble_size+1] + 0.5*ensemble_data.iloc[:,ensemble_size+2]
    ensemble_data['lower'] = ensemble_data.iloc[:,ensemble_size+1] - 0.5*ensemble_data.iloc[:,ensemble_size+2]
    
   
    return ensemble_data
    
def percentFLC_plt(ensemble_data):
    line = alt.Chart(ensemble_data).mark_line().encode(
    x=alt.X('Time_step', title='Time [1 unit = 1 hour]'),
    y=alt.Y('mean', title= '% of cells with FT1 expressed')
    )

    band = alt.Chart(ensemble_data).mark_area(
    opacity=0.4, color='green').encode(
    x=alt.X('Time_step', title='Time [1 unit = 1 hour]'),
    y='lower',
    y2='upper'
    )
    
    return (line + band).interactive() 

def std_plt(ensemble_data):
    line2 = alt.Chart(ensemble_data).mark_line().encode(
    x=alt.X('Time_step', title='Time [1 unit = 1 hour]'),
    y=alt.Y('std', title= 'Standard Deviation')
    )
    
    return line2.interactive()
    

