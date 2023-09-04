"""
This contains the codes for the different plots

"""

import numpy as np
import pandas as pd
import pyinform as pi
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation 

df = pd.read_csv("sam.cvs") #reads in the sam file

edgelist = []
edgelist_w = []
nodes = []

#These create the edgelists, if you already have these than just read in those files
for i in range(len(df["label"])):
    edgelist.append((df["label"][i], df[" neighbor"][i]))

for i in range(len(df["label"])):
    edgelist_w.append((df["label"][i], df[" neighbor"][i], df[" wall area"][i]))

for i in range(len(df["label"])):
    nodes.append(df["label"][i])

nodes = np.unique(nodes)
nodes = list(nodes)

df1 = pd.read_csv("trj_data_4wC", skiprows=1, header = None)
df = df.drop(0,1)
df = df.drop(1,1)
df = df.drop(2,1)

# This determines the colors you want the nodes to be
colors = []
for i in range(673): # 673 timesteps
    temp = []
    for j in range(3, 285): #282 nodes for this file
        if df1[j][i] != 1:
            temp.append('k')
        else:
            temp.append('w')
    colors.append(temp)

cols = np.array(colors)
np.save("colors_basedontimeseries.npy") # this is to save it for future uses

plt.rcParams["figure.figsize"] = [12, 10]
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edgelist)
pos = nx.kamada_kawai_layout(G)

def animate(j):
    cs = cols[j]
    nod = nx.draw_networkx_nodes(G, pos, node_color = cs)
    nod.set_edgecolor('k')

nx.draw_networkx_edges(G,pos)
fig = plt.gcf()

anim = animation.FuncAnimation(fig, animate, frames = 673, interval = 1) #frames should equal number of timesteps

writervideo = animation.PillowWriter(fps = 60)
anim.save("4wC_282Cells_05312022.gif", writer = writervideo) #This takes a long time but does work 