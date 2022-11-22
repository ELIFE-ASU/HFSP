import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from scipy import optimize
import networkx as nx
import pandas as pd
import collections
import itertools
import hfsp_functions as hfsp


T = hfsp.create_tissue("sam.csv",edge_weight = False)
g = T
hfsp.update_spontaneous(g, jump_state = "default") 
temp_sch = np.array([1000])
p_cold = 0.0005
p_warm = 0.2
ensemble_data = hfsp.ensemble(g, temp_sch, p_cold, p_warm, rule_code = 1, ensemble_size = 10)
ensemble_data.to_csv("ensemble_data.csv")
