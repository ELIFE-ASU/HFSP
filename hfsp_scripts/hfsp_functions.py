import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from scipy import optimize
import networkx as nx
import pandas as pd
import itertools 


# Load data from the respective csv (enter in csv_path).
# edge_state is a boolean variable. If it is False, no state-attribute would be assingned to the edges. If it is True, Initial state - 0 will be assingned to all edges.


def create_tissue(csv_path,edge_state):
    
    csv_df = pd.read_csv(csv_path) 
    
    g = nx.Graph()
    node_list = list(set(csv_df.iloc[:,0]))
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
    plt.figure(figsize = (15,15))
    pos = nx.kamada_kawai_layout(g)
    # pos = nx.fruchterman_reingold_layout(g)
    color_list_nodes = []
    for x in g.nodes():
        if g.nodes[x]["state"] == 0:
            color_list_nodes.append("black")
        if g.nodes[x]["state"] == 1:
            color_list_nodes.append("white")
            
    nx.draw_networkx_nodes(g, pos, node_color= color_list_nodes, edgecolors="black", node_size = 40)
    if edge_state == False:
        nx.draw_networkx_edges(g,pos,edge_color = 'green', width=2)
        
    if edge_state == True:
        color_list_edges = []
        for x in g.edges():
            if g.edges[x]["edge_state"] == 0:
                color_list_edges.append("red")
            if g.edges[x]["edge_state"] == 1:
                color_list_edges.append("green")
            
        nx.draw_networkx_edges(g,pos,edge_color = color_list_edges, width=2)
   
    if save_with_name != None:
        plt.savefig(save_with_name, dpi = 600)
    
    plt.show()

# Update the nodes (gene expression) according the following function;
# rule_code = [0,m]: n1/n0 ; All edges are assumed to be open (green)
# rule_code = [1,m]: n_12/n0 > m
# rule_code = [2,m]: n_12/n2 > m


def update_rule_nodes(g, temp, p_decay, p_cold, p_warm, rule_code): 
    for x in list(g.nodes()):
        
        if temp == 0 and g.nodes[x]["state"] == 0:
            c1 = np.random.choice([0,1], p = [p_cold,1-p_cold])
            if c1 == 0:
                g.nodes[x]["state"] = 1
                
            # Code for Majority Rule 0
            if c1 == 1 and rule_code[0] == 0:
                neighbor_states = np.array([g.nodes[y]["state"] for y in list(g.neighbors(x))])
                if np.sum(neighbor_states)/len(neighbor_states) >= rule_code[1]:
                    g.nodes[x]["state"] = 1
                    
            # Code for Majority Rule 1
            if c1 == 1 and rule_code[0] == 1:
                active_edges = [] 
                Q = list(g.edges([x])) 
                for j in range(len(Q)):
                    if g.edges[Q[j]]["edge_state"] == 1:
                        active_edges.append(Q[j]) # This way, we create a list of active edges of a given node 'x'
                N2 = [k[1] for k in active_edges] # list of 'nodes' with active edges

                N0 = list(g.neighbors(x))
                
                N1 = []
                P = list(g.neighbors(x)) 
                for i in range(len(P)):
                    if g.nodes[P[i]]["state"] == 1:
                        N1.append(P[i])
                N12 = list(set(N2) & set(N1)) # nodes with active edges AND active nodes!

                n = len(N12)/len(N0)
                if n >= rule_code[1]:
                    g.nodes[x]["state"] = 1
                    
            # Code for Majority Rule 2           
            if c1 == 1 and rule_code[0] == 2:
                active_edges = [] 
                Q = list(g.edges([x])) 
                for j in range(len(Q)):
                    if g.edges[Q[j]]["edge_state"] == 1:
                        active_edges.append(Q[j]) # This way, we create a list of active edges of a given node 'x'
                N2 = [k[1] for k in active_edges] # list of 'nodes' with active edges
                
                if len(N2) != 0:
                    N1 = []
                    P = list(g.neighbors(x)) 
                    for i in range(len(P)):
                        if g.nodes[P[i]]["state"] == 1:
                            N1.append(P[i])
                    N12 = list(set(N2) & set(N1)) # nodes with active edges AND active nodes!

                    n = len(N12)/len(N2)
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

def update_rule_edges(g, temp, p_edge_cold, p_edge_warm, rule_code): 
    # If rule code is 'None' then it means edge dynamics is turned OFF
    
    if rule_code == 0: # decoupled PD and genetics
        for x in list(g.edges()):
            if temp == 0 and g.edges[x]["edge_state"] == 0:
                c0 = np.random.choice([0,1], p = [p_edge_cold, 1-p_edge_cold])
                if c0 == 0:        
                    g.edges[x]["edge_state"] = 1

        for x in list(g.edges()):
            if temp == 1 and g.edges[x]["edge_state"] == 1:
                c1 = np.random.choice([0,1], p = [p_edge_warm, 1-p_edge_warm])
                if c1 == 0:        
                    g.edges[x]["edge_state"] = 0

    
    # if rule_code == 1: # Coupled PD and genetics
    #     active_nodes = [x for x in g.nodes() if g.nodes[x]["state"] == 1]
    #     potentially_active_edges = list(g.edges(active_nodes)) # check if there are any repetitions in this set!
    #     for y in potentially_active_edges:
    #         if temp == 0 and g.edges[y]["edge_state"] == 0:
    #             c1 = np.random.choice(['PD_open', 'PD_closed'], p = [p_edge, 1-p_edge])
    #             if c1 == 'PD_open':        
    #                 g.edges[y]["edge_state"] = 1
            
    return g       
    
def update_spontaneous(g, jump_state):
    assert len(g.nodes()) + len(g.edges()) == len(jump_state)
    for i in range(len(g.nodes())):
        g.nodes[list(g.nodes())[i]]["state"] = jump_state[i]
    for j in range(len(g.edges())):
        g.edges[list(g.edges())[j]]["edge_state"] = jump_state[len(g.nodes()) + j]

    
def update_individual_node(g, node, state):
    g.nodes[node]["state"] = state
    return g

    
def update_individual_edge(g, edge, state):
    g.edges[edge]["edge_state"] = state
    return g
        


def trajectory(g, temp_sch, p_decay, p_cold, p_warm, p_edge_cold, p_edge_warm, rule_code_node, rule_code_edge): # temp schedule is the list of cold (0), warm(1) schedules 
    
    temp_array = np.array([],dtype = int)         
    for i in range(len(temp_sch)):
        if int(i/2)*2  == i: 
            to_append = np.zeros(temp_sch[i], dtype = int)
        else:
            to_append = np.ones(temp_sch[i], dtype = int)
        temp_array = np.append(temp_array, to_append)
        
    
    trajectory = np.empty([len(temp_array)+1 , len(g.nodes())+len(g.edges())],dtype = int)
    g_0 = g
    trajectory[0] = np.array([g_0.nodes[j]["state"] for j in g_0.nodes()] + [g_0.edges[j]["edge_state"] for j in g_0.edges()])
    for k in range(len(temp_array)):
        g_0 = update_rule_nodes(g_0, temp_array[k], p_decay, p_cold, p_warm, rule_code_node)
        g_0 = update_rule_edges(g_0, temp_array[k], p_edge_cold, p_edge_warm, rule_code_edge)
        trajectory[k+1] = np.array([g_0.nodes[j]["state"] for j in g_0.nodes()] + [g_0.edges[j]["edge_state"] for j in g_0.edges()])
        
    time_array = np.arange(len(temp_array)+1)
    avg_exp_nodes = []
    avg_exp_edges = []
    for i in range(len(trajectory)):
        avg_exp_nodes.append(np.sum(trajectory[i,:(len(g.nodes()))])*100/len(trajectory[0,:(len(g.nodes()))]))
    node_exp_percent = np.array(avg_exp_nodes)
    
    for j in range(len(trajectory)):
        avg_exp_edges.append(np.sum(trajectory[j,(len(g.nodes())):])*100/len(trajectory[0,(len(g.nodes())):]))
    edge_exp_percent = np.array(avg_exp_edges)
    
    trajectory_df = pd.DataFrame(trajectory, columns = ['{}'.format(x) for x in g.nodes()] + ['{}'.format(x) for x in g.edges()])
    trajectory_df.insert(0, "time", time_array)
    trajectory_df.insert(1, "% of active nodes", node_exp_percent)
    trajectory_df.insert(2, "% of active edges", edge_exp_percent)
    return trajectory_df
    

def ensemble(g, temp_sch, p_decay, p_cold, p_warm, p_edge_cold, p_edge_warm, rule_code_node, rule_code_edge, ensemble_size, jump_state):
    ensemble_data = trajectory(g, temp_sch, p_decay, p_cold, p_warm, p_edge_cold, p_edge_warm, rule_code_node, rule_code_edge).iloc[:,[0,1]]
    ensemble_data.rename(columns={"% of active nodes":"sim_1"} ,inplace=True)
    for i in range(ensemble_size-1):
        update_spontaneous(g, jump_state)
        traj = trajectory(g, temp_sch, p_decay, p_cold, p_warm, p_edge_cold, p_edge_warm, rule_code_node, rule_code_edge).iloc[:,1]
        ensemble_data.insert(i+2,"sim_{}".format(i+2), traj)
    
    ensemble_data['mean'] = ensemble_data.iloc[:,1:ensemble_size+1].mean(axis=1)
    ensemble_data['std'] = ensemble_data.iloc[:,1:ensemble_size+1].std(axis=1)
    ensemble_data['upper'] = ensemble_data.iloc[:,ensemble_size+1] + 1.96*ensemble_data.iloc[:,ensemble_size+2]/np.sqrt(ensemble_size)
    ensemble_data['lower'] = ensemble_data.iloc[:,ensemble_size+1] - 1.96*ensemble_data.iloc[:,ensemble_size+2]/np.sqrt(ensemble_size)
    
   
    return ensemble_data
    
    

