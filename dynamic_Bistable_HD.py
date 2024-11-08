import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools as its
from utils import *
import scale_free_graph as sf
import networkx as nx
import collections as cls




class Dynamics():
    def __init__(self, J,  steps=10000 , step_size=0.1):
        self.J = J
        self.s = 3
        self.g = 3.5

        self.integrate_steps = steps
        self.step_size = step_size

    def integrate(self, x):
        dy = -x + self.s * torch.tanh( x ) + self.g * torch.matmul(  self.J , torch.tanh(x).T ).T
        return dy


    def __call__(self, x ):
        h = self.step_size
        k1 = self.integrate(x)
        k2 = self.integrate(x + h * k1 / 2)
        k3 = self.integrate(x + h * k2 / 2)
        k4 = self.integrate(x + h * k3)
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state


    def terminate_point(self, x):
        # def in_basin(self, x, attractor1, attractor2 ):
        for _ in range(self.integrate_steps):
            x = self.__call__(x)

        return x



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    np.random.seed(123456)

    data_dimension = 3

    # J = np.random.random([data_dimension,data_dimension])*5  # sampling from N(10,1)
    # J = J - np.diag(np.diag(J) ) #elements in diag
    # J = torch.Tensor(J)
    "create adjacent_matrix"
    mean_degree = 2

    graph = sf.scale_free(data_dimension, mean_degree, seed=123456 )
    # largest_cc = max( nx.weakly_connected_components(graph), key=len )
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc)
    # graph.remove_nodes_from(list(nx.isolates(graph)))
    mapping = dict(zip(graph, range(0, len(graph))))
    graph = nx.relabel_nodes(graph, mapping)
    node_num = len(graph)
    print(f'node_num:{node_num}')
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos=pos, node_size=50)
    plt.show()

    W = np.random.randn(len(graph.edges())) * 1 + 0
    source, target = zip(*graph.edges())
    weight_edges = zip(source, target, W)
    graph.add_weighted_edges_from(weight_edges)
    B = nx.to_numpy_array(graph)
    for s, t, w in graph.edges(data=True):
        B[s, t] = w['weight']
    B = torch.Tensor(B).T


    dynamic = Dynamics( step_size=0.1, steps=2000, J=B.to(device) )
    print(B)
    # np.random.seed(123456)

    points = np.random.uniform(-20, 20, [100000, node_num])
    # print(points)
    result = dynamic.terminate_point(torch.Tensor(points).to(device))
    # print(result)
    result = torch.round(result, decimals=1).cpu().numpy()  # [bs,nodes]
    counter = dict(cls.Counter([tuple(i) for i in result]))

    attractor = set( [ tuple(i) for i in result ] )
    print(len(attractor))
    print( attractor)

    print( list( counter.values() ) )
    print( len( list( counter.keys() ) ) )


