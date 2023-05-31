"""
This script contains all the functions for the channel capacity calculations

Modified: 05282023
Created by: Hannah Dromiack
"""

import numpy as np
import itertools

def poss_is(n):
    """
    This is generate an array of all possible configurations for a binary statesystem

    Input: n - Int, number of nodes in the system 
    Output: states - Array, states configurations for n node system
    """
    A = itertools.product('10', repeat=n)
    states = np.array(list(A)).astype("int")
    return states

###### chan_uniform and chan_gaussian are MI calculators for 1 type of p(x), chan_random is a channel capacity calculator. 

def chan_uniform(n, t, e = True):
    """
    Input: n - Int, number of neighbors in the system 
           t - Int, Threshold for Majority rule
           e - Boolean, defines if the majority rule is 'greater than' or 'greater than or equal to'
               default is True which corresponds to greater than or equal to.

    Output: Int, Channel Capacity 
    """

    ics = poss_is(n)
    count = 0
    for i in range(len(ics)):
        if e == True: 
            if np.sum(ics[i])/ics.shape[1] >= t:
                count += 1
        else:
            if np.sum(ics[i])/ics.shape[1] > t:
                count += 1

    y1 = np.log2(len(ics)/count) # probability for outcome 1, p(y=1)
    y0 = np.log2(len(ics)/(len(ics)-count)) # probability for outcome 0, p(y=0)
    return np.mean([y0,y1])

def gaussian(n):
    vals = np.random.normal(size=2**n)
    norm_vals = np.abs(vals)/np.sum(np.abs(vals))
    return norm_vals

def chan_gaussian(n, t, e = True):
    """
    Input: n - Int, number of neighbors in the system 
           t - Int, Threshold for Majority rule
           e - Boolean, defines if the majority rule is 'greater than' or 'greater than or equal to'
               default is True which corresponds to greater than or equal to.

    Output: Int, Channel Capacity 
    """

    ics = poss_is(n)
    p_ic = gaussian(n) # probability of initial states i.e. p(x)
    p_jicfc = 1/len(ics) # joint probility, p(x,y)

    count = 0
    for i in range(len(ics)):
        if e == True: 
            if np.sum(ics[i])/ics.shape[1] >= t:
                count += 1
        else:
            if np.sum(ics[i])/ics.shape[1] > t:
                count += 1

    y1 = count/len(ics) # probability for outcome 1, p(y=1)
    y0 = (len(ics)-count)/len(ics) # probability for outcome 0, p(y=0)

    mis = np.zeros(len(ics))
    for i in range(len(ics)):
        if e == True: 
            if np.sum(ics[i])/ics.shape[1] >= t:
                mi = np.log2(p_jicfc/(p_ic[i]*y1))
            else:
                mi = np.log2(p_jicfc/(p_ic[i]*y0))
        else: 
            if np.sum(ics[i])/ics.shape[1] > t:
                mi = np.log2(p_jicfc/(p_ic[i]*y1))
            else:
                mi = np.log2(p_jicfc/(p_ic[i]*y0))
        mis[i] = mi
    
    return np.mean(mis)

# Written on 05282023
def random(seed, x):
    rng = np.random.default_rng(seed = seed)
    nums = rng.random(x)
    vals = np.abs(nums)/np.sum(np.abs(nums))
    return vals

def chan_random(n, t, r, seed = None, e = True):
    """
    Input: n - Int, number of neighbors in the system 
           t - Int, Threshold for Majority rule
           r - Int, number of iterations
           seed - {None, int, array of ints, SeedSequence}, if None fresh entropy is pulled from the OS if the others then SeedSequence will be the generator. Default is None. 
           e - Boolean, defines if the majority rule is 'greater than' or 'greater than or equal to'
               default is True which corresponds to greater than or equal to.

    Output: Int, Channel Capacity 
    """
    ics = poss_is(n)
    MIs = np.zeros(r)

    for j in range(r):
        p_ic = random(seed, len(ics)) # probability of initial states i.e. p(x)
        p_jicfc = 1/len(ics) # joint probility p(x,y), alter if necessary 
        
        count = 0
        for i in range(len(ics)):
            if e == True: 
                if np.sum(ics[i])/ics.shape[1] >= t:
                    count += 1
            else:
                if np.sum(ics[i])/ics.shape[1] > t:
                    count += 1

        y1 = count/len(ics) # probability for outcome 1, p(y=1)
        y0 = (len(ics)-count)/len(ics) # probability for outcome 0, p(y=0)

        l_mis = np.zeros(len(ics))
        for i in range(len(ics)):
            if e == True: 
                if np.sum(ics[i])/ics.shape[1] >= t:
                    l_mi = np.log2(p_jicfc/(p_ic[i]*y1))
                else:
                    l_mi = np.log2(p_jicfc/(p_ic[i]*y0))
            else: 
                if np.sum(ics[i])/ics.shape[1] > t:
                    l_mi = np.log2(p_jicfc/(p_ic[i]*y1))
                else:
                    l_mi = np.log2(p_jicfc/(p_ic[i]*y0))
            l_mis[i] = l_mi
        MIs[j] = np.mean(l_mis)
    
    cc = np.max(MIs)
    return cc