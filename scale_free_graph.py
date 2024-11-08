import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def scale_free( n, md, m=None, seed=123 ):

    np.random.seed( seed )

    if m == None:
    # m = n * md
        m = int( n * md / 2 )
    # print(m)
    gamma_in = 2.5
    gamma_out = 2.5

    alpha_in = 1/(gamma_in-1)
    alpha_out = 1/(gamma_out-1)

    w_in = np.ones(n)
    w_out = np.ones(n)
    edges = list()

    for i in range(n):
        w_in[i] = 1 / (1 + i)**alpha_in
        w_out[i] = 1 / (1 + i)**alpha_out


    w_in = np.random.permutation(w_in)
    w_out = np.random.permutation(w_out)
    w_in = w_in / np.sum(w_in)
    w_out = w_out / np.sum(w_out)

    l = 0
    while l < m:

        s = np.random.choice(range(n), p=w_out)

        t = np.random.choice(range(n), p=w_in)

        if s != t:
            edge = (s, t)
            if edge not in edges:
                edges.append(edge)
                l += 1

    # print(edges)
    # g = nx.DiGraph()
    g = nx.Graph()
    g.add_nodes_from( range(n) )
    g.add_edges_from( edges )

    
    return g






