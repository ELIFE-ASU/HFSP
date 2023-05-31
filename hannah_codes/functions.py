import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pyinform as pi 
import pandas as pd

def load_data(traj, sam):
    """
    Loads trajectory, node and edge date for given plant networks

    Parameters:
    -----------
    traj : str
           File path to expression trajectory .csv file
    sam : str
          File path to sam.csv file, this file describes the edge dynamics 
    
    Returns
    -------
    df : Pandas DataFrame
        Dataframe containing the expression trajectory series
    nodes: Array 
          Contains the node names contained within the network
    edges: Array 
           Contains the edge pairs within the network
    """
    df1 = pd.read_csv(sam)
    edgelist = []
    for i in range(len(df1["Cell label"])):
        edgelist.append((df1["Cell label"][i], df1["Neighbour Label"][i]))
    edges = np.array(edgelist)
    nodes = np.arange(1,np.max(edges)+1)
    
    df = pd.read_csv(traj)
    df = df.drop('Unnamed: 0', axis = 1) #Removes columns
    df = df.drop('time', axis = 1)
    df = df.drop('expression_level', axis = 1)
    df.columns = df.columns.str.lstrip('node')

    return df, nodes, edges

def ts_colors(traj):
    """
    Creates a 2D array containing colors to represent whether a cell is expressing or not

    Parameters:
    -----------
    traj : str
           File path to expression trajectory .csv file
    
    Returns
    -------
    cols: Array 
          2D array containing colors coordinated to whether a cell is expressing 
          white - Cell is expressing 
          black - Cell is not expressing
    """
    df = pd.read_csv(traj, skiprows=1, header = None)
    df = df.drop(0, axis = 1)
    df = df.drop(1, axis = 1)
    df = df.drop(2, axis = 1)

    colors = []
    for i in range(df.shape[0]):
        temp = []
        for j in range(3, df.shape[1]+3):
            if df[j][i] != 1:
                temp.append('k')
            else:
                temp.append('w')
        colors.append(temp)

    cols = np.array(colors)
    return cols

def MI_calc(df, nodes):
    """
    Calculates the Pairwise Mutual Information within the network 

    Parameters:
    -----------
    df : Pandas DataFrame
        Dataframe containing the expression trajectory series
    nodes: Array 
          Contains the node names contained within the network
    
    Returns
    -------
    MI: 2D Array 
        Contains the values for pairwise mutual information
    """
    MI = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            MI[i,j] = pi.mutual_info(df[str(nodes[i])],df[str(nodes[j])])
    
    MI = np.array(MI)
    return MI

def rank(traj, sam):
    """
    Determines rank ordering of the nearest neighbor pairwise mutual information 

    Parameters:
    -----------
    traj : str
           File path to expression trajectory .csv file
    sam : str
          File path to sam.csv file, this file describes the edge dynamics 
    
    Returns
    -------
    Values: Array 
            Rank order of the values for pairwise mutual information
    """
    df, nodes, edges = load_data(traj, sam)
    MI = MI_calc(df, nodes)
    
    box = []
    for i in range(len(edges)):
        if edges[i][1] < edges[i][0]:
            box.append(i)
    box.reverse()

    values = []
    for i in range(len(edges)):
        values.append(MI[edges[i][0]-1,edges[i][1]-1])
    
    for i in box: 
        values.pop(i)
    
    values.sort(reverse = True)
    return values

def PW_NetworkPlot(traj, sam, fn):
    """
    Loads trajectory, node and edge date for given plant networks

    Parameters:
    -----------
    traj : str
           File path to expression trajectory .csv file
    sam : str
          File path to sam.csv file, this file describes the edge dynamics 
    fn : str
         Filename for outputted plot

    Returns
    -------
    plt : Matplotlib plot
         Final plot output saved as a .png with a dpi = 300
    """
    df, nodes, edges = load_data(traj, sam)
    cols = ts_colors(traj)
    values = MI_calc(df, nodes)

    na = []
    for i in range(len(cols[cols.shape[0]-1])):
        if cols[cols.shape[0]-1][i] != 'w':
            na.append(0.5)
        else:
            na.append(1.0)
    na = np.array(na)

    box = []
    for i in range(len(edges)):
        if edges[i][1] < edges[i][0]:
            box.append(i)
    box.reverse()

    edgeweights = []
    for i in range(len(edges)):
        edgeweights.append(values[edges[i][0]-1,edges[i][1]-1])

    weights = edgeweights[:]
    for i in box:
        weights.pop(i)

    plt.rcParams["figure.figsize"] = [10, 8]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(G)

    count = 0
    for i,j in G.edges():
        G[i][j]['weight'] = weights[count]
        count += 1

    nx.draw_networkx_nodes(G, pos, node_size = 300, node_color = cols[cols.shape[0]-1], alpha = na, edgecolors = 'k')
    nx.draw_networkx_labels(G, pos, font_size = 6)
    weights = list(nx.get_edge_attributes(G,'weight').values())
    nx.draw_networkx_edges(G, pos, width = weights)

    plt.axis('off')
    plt.tight_layout()
    return plt.savefig(fn+".png", dpi = 300)
