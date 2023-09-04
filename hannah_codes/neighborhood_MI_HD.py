#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx

def neighborhood_MI(trajectory_df, net):
    '''
       given a simulated trajectory, calculate the MI over time in each neighborhood size (k) 
    
       inputs:
       trajectory_df: DataFrame (output of simulation using hfsp_functions.py's "trajectory" function)
       net: networkx Graph (created with hfsp_functions.py's "create_tissue" function). network must be rigid.
       
       returns: 
       2d array, global MI for each timestep for each neighborhood size
    '''
    MI_allk_overt = []
    
    max_deg = max([j for i,j in net.degree()]) # rigid nets have a maximum degree 
    for t in range(0, trajectory_df.shape[0]-1):
        
        # group nodes into active-neighborhood sizes (their k values):
        N_allk_t = [] 
        for i in range(0,max_deg+1):
            N_allk_t.append([])
        for node in list(net.nodes()):
            on_deg = sum([trajectory_df.loc[t,'edge'+str((i,neighbor))] for (i,neighbor) in net.edges() if i == node or neighbor == node ]) #edge_state for edges of node 
            if on_deg != 0: # ignoring neighborhood sizes of 0 since their MI = 0
                N_allk_t[on_deg].append(node) 
        
        # calculate global MI for each k:
        MI_allk_t = []
        for en,k_group in enumerate(N_allk_t): 
            
            if len(k_group) == 0:
                MI_k = 0

            else: 
                # get input distribution, output distribution of nodes of interest:
                pop_size = len(k_group)
                inputs = []
                outputs = []
                for node in k_group:
                    inputs_node = []
                    neighbor_states = [trajectory_df.loc[t,'node'+str(neighbor)] for (i,neighbor) in net.edges() if i == node and trajectory_df.loc[t,'edge'+str((i,neighbor))]==1]
                    others = [trajectory_df.loc[t,'node'+str(neighbor)] for (neighbor,i) in net.edges() if i == node and trajectory_df.loc[t,'edge'+str((neighbor,i))]==1] 
                    for s in neighbor_states:
                        inputs_node.append(s)
                    for s_ in others:
                        inputs_node.append(s_)
                    inputs.append(inputs_node)
                    outputs.append(trajectory_df.loc[t+1,'node'+str(node)])
                
                # get P_ys:
                num_on = np.sum(np.array(outputs))
                num_off = pop_size - num_on
                P_ys = []
                for o in outputs:
                    if o == 0:
                        P_ys.append(num_off / pop_size)
                    elif o == 1:
                        P_ys.append(num_on / pop_size)
                        
                # get P_xs:
                counts = []
                for i in inputs:
                    count_i = 0
                    for j in inputs:
                        if sum(j) == sum(i):
                            count_i += 1
                    counts.append(count_i)
                P_xs = np.array(counts) / pop_size 

                # get P_x_ys:
                ins_w_outs = inputs.copy() 
                for ind,i in enumerate(ins_w_outs):
                    i.append(outputs[ind])
                counts_xy = []
                for i in ins_w_outs:
                    count_i = 0
                    for j in ins_w_outs:
                        if sum(j[:-1]) == sum(i[:-1]) and j[-1] == i[-1]:
                            count_i += 1
                    counts_xy.append(count_i)
                P_x_ys = np.array(counts_xy) / pop_size

                # calculate global MI:
                MI_k = np.mean(- np.log2(P_x_ys / (P_xs*P_ys)))
            
            MI_allk_t.append(MI_k)

        MI_allk_overt.append(MI_allk_t)
    return MI_allk_overt

