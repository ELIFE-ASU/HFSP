import numpy as np
import matplotlib.pyplot as plt 
from IPython.display import Math, Latex # for latex equations
from IPython.core.display import Image # for displaying images
from scipy import stats
from scipy import optimize
import networkx as nx
import pandas as pd
import itertools 
import altair as alt # no need


# Load data from the respective csv and construct a static graph/network of plant tissue with the state 'state'

def create_tissue(csv_path,edge_state):
    
    csv_df = pd.read_csv(csv_path) 
    
    g = nx.Graph()
    node_list = list(sorted(set(csv_df.iloc[:,0])))
    g.add_nodes_from(node_list)
    for i in range(len(csv_df)):
        g.add_edge(csv_df.iloc[i,0],csv_df.iloc[i,1])
    initial_node_state = np.zeros(len(g.nodes()), dtype = np.int8)
    assert len(initial_node_state) == len(g.nodes()) 
    nx.set_node_attributes(g, dict(zip(g.nodes(), initial_node_state)), name="state") 
    if edge_state == True:
        initial_edge_state = np.zeros(len(g.edges()), dtype = np.int8)
        nx.set_edge_attributes(g, dict(zip(g.edges(), initial_edge_state)), "edge_state")
    
    return g

def plt_tissue(g, edge_state, save_with_name):
    plt.figure(figsize = (12,12))
    pos = nx.kamada_kawai_layout(g)
    color_list_nodes = []
    for x in g.nodes():
        if g.nodes[x]["state"] == 0:
            color_list_nodes.append("black")
        if g.nodes[x]["state"] == 1:
            color_list_nodes.append("white")
            
    nx.draw_networkx_nodes(g, pos, node_color= color_list_nodes,edgecolors="black")
    if edge_state == False:
        nx.draw_networkx_edges(g,pos,edge_color = 'green', width=2)
        
    if edge_state == True:
        '''        
        all_weights = []
        for (node1,node2,data) in g.edges(data=True):
            all_weights.append(data["weight"])
        unique_weights = list(set(all_weights))

        for w in unique_weights:
            weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in g.edges(data=True) if edge_attr['weight']==w]
            nx.draw_networkx_edges(g,pos,edge_color = 'blue',edgelist=weighted_edges,width=0.3*w)
        '''
        color_list_edges = []
        for x in g.edges():
            if g.edges[x]["edge_state"] == 0:
                color_list_edges.append("red")
            if g.edges[x]["edge_state"] == 1:
                color_list_edges.append("green")
            
        nx.draw_networkx_edges(g,pos,edge_color = color_list_edges, width=2)
        
    if save_with_name != None:
        plt.savefig(save_with_name)
    
    plt.show()

# Update the nodes (gene expression) according the following function;

# rule_code = [0,m]: All edges are assumed to be open (green)
# rule_code = [1,m]: n_11/n > m
# rule_code = [2,m]: n_11/n_1 > m
# rule_code = [3,c]: n_11 > c

def update_rule_nodes(g, temp, p_decay, p_cold, p_warm, rule_code): 
    for x in list(g.nodes()):
        
        if temp == 0 and g.nodes[x]["state"] == 0:
            c1 = np.random.choice([0,1], p = [p_cold,1-p_cold])
            if c1 == 0:
                g.nodes[x]["state"] = 1
            
            if c1 == 1 and rule_code[0] == 0:
                neighbor_states = np.array([g.nodes[y]["state"] for y in list(g.neighbors(x))])
                if np.sum(neighbor_states)/len(neighbor_states) >= rule_code[1]:
                    g.nodes[x]["state"] = 1
            
            if c1 == 1 and rule_code[0] == 3:
                active_edges = [] 
                Q = list(g.edges([x])) 
                for j in range(len(Q)):
                    if g.edges[Q[j]]["edge_state"] == 1:
                        active_edges.append(Q[j]) # This way, we create a list of active edges of a given node 'x'
                nodes_active_edges = [k[1] for k in active_edges] # list of 'nodes' with active edges
                
                active_nodes = []
                P = list(g.neighbors(x)) 
                for i in range(len(P)):
                    if g.nodes[P[i]]["state"] == 1:
                        active_nodes.append(P[i])
                active_nodes_edges = list(set(nodes_active_edges) & set(active_nodes)) # nodes with active edges AND active nodes!
                
                n = len(active_nodes_edges)
                if n >= rule_code[1]:
                    g.nodes[x]["state"] = 1
                    
        if temp == 1 and g.nodes[x]["state"] == 1:
            c2 = np.random.choice([0,1], p = [p_warm,1-p_warm])
            if c2 == 0:
                g.nodes[x]["state"] = 0
                
        if p_decay != 0:        
            if g.nodes[x]["state"] == 1:
                c0 = np.random.choice([0,1], p = [p_decay, 1-p_decay])
                if c0 == 0:
                    g.nodes[x]["state"] = 0
    return g  

def update_rule_edges(g, p_edges, rule_code): 
    # If rule code is 0 then it means edge dynamics is turned OFF
    if rule_code == 0: # decoupled PD and genetics
        for x in list(g.edges()):
            if temp == 0 and g.edges[x]["edge_state"] == 0:
                c0 = np.random.choice(['PD_open', 'PD_closed'], p = [p_edges, 1-p_edges])
                if c0 == 'PD_open':        
                    g.edges[x]["edge_state"] = 1 # Very cool! This should work.

    if rule_code == 1: # Coupled PD and genetics
        active_nodes = [x for x in g.nodes() if g.nodes[x]["state"] == 1]
        potentially_active_edges = list(g.edges(active_nodes))
        for y in potentially_active_edges:
            c1 = np.random.choice(['PD_open', 'PD_closed'], p = [p_edges, 1-p_edges])
            if c1 == 'PD_open':        
                g.edges[y]["edge_state"] = 1
    # if rule_code == 10: rule 0 and 1 are superimposed!
    # if rule_code == 2
    # if rule_code == 20: rule 0 and 2 are superimposed!
        
        
    return g       
    
def update_spontaneous(g, jump_state):
    if jump_state == "default" :
        jump = np.zeros((len(g.nodes()),),dtype = int)
    else:
        assert len(g.nodes()) == len(jump_state)
        jump = jump_state
    for i in range(len(g.nodes())):
        g.nodes[list(g.nodes())[i]]["state"] = jump[i]
        


def trajectory(g, temp_sch, p_decay, p_cold, p_warm, p_edge, rule_code_node, rule_code_edge): # temp schedule is the list of cold (0), warm(1) schedules 
    
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
        g_0 = update_rule_nodes(g_0, temp_array[k], p_decay, p_cold, p_warm, rule_code_node)
        trajectory[k+1] = np.array([g_0.nodes[j]["state"] for j in g_0.nodes()])
        
    time_array = np.arange(len(temp_array)+1)
    avg_exp = []
    for i in range(len(trajectory)):
        avg_exp.append(np.sum(trajectory[i])*100/len(trajectory[0]))
    expression_level = np.array(avg_exp)
    
    trajectory_df = pd.DataFrame(trajectory, columns = ['node{}'.format(x) for x in g.nodes()])
    trajectory_df.insert(0, "time", time_array)
    trajectory_df.insert(1, "expression_level", expression_level)
    return trajectory_df
    

def ensemble(g, temp_sch, p_decay, p_cold, p_warm, p_edge, rule_code_node, rule_code_edge, ensemble_size, jump_state):
    ensemble_data = trajectory(g, temp_sch, p_decay, p_cold, p_warm, p_edge, rule_code_node, rule_code_edge).iloc[:,[0,1]]
    ensemble_data.rename(columns={"expression_level":"sim_1"} ,inplace=True)
    for i in range(ensemble_size-1):
        update_spontaneous(g, jump_state)
        traj = trajectory(g, temp_sch, p_decay, p_cold, p_warm, p_edge, rule_code_node, rule_code_edge).iloc[:,1]
        ensemble_data.insert(i+2,"sim_{}".format(i+2), traj)
    
    ensemble_data['mean'] = ensemble_data.iloc[:,1:ensemble_size].mean(axis=1)
    ensemble_data['std'] = ensemble_data.iloc[:,1:ensemble_size].std(axis=1)
    ensemble_data['upper'] = ensemble_data.iloc[:,ensemble_size+1] + 1.96*ensemble_data.iloc[:,ensemble_size+2]/np.sqrt(ensemble_size)
    ensemble_data['lower'] = ensemble_data.iloc[:,ensemble_size+1] - 1.96*ensemble_data.iloc[:,ensemble_size+2]/np.sqrt(ensemble_size)
    
   
    return ensemble_data
    
def percentGA_plt(ensemble_data, color):
    line = alt.Chart(ensemble_data).mark_line(color = color).encode(
    x=alt.X('time', title='time [1 unit = 1 hour]'),
    y=alt.Y('mean', title= '% of cells with GA20OX-1 expressed')
    )

    band = alt.Chart(ensemble_data).mark_area(
    opacity=0.4, color=color).encode(
    x=alt.X('time', title='time [1 unit = 1 hour]'),
    y='lower',
    y2='upper'
    )
    
    return (line + band).interactive() 

def std_plt(ensemble_data):
    line2 = alt.Chart(ensemble_data).mark_line().encode(
    x=alt.X('time', title='time [1 unit = 1 hour]'),
    y=alt.Y('std', title= 'Standard Deviation')
    )
    
    return line2.interactive()
    
