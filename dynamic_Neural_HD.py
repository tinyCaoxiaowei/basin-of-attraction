import scipy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx as nx
from networkx import barabasi_albert_graph
import scale_free_graph as sf


class Dynamics():
    def __init__(self, steps=10000, adjacent_matrix=None, step_size=0.1):
        self.adjacent_matrix = adjacent_matrix
        self.f = 1. #0.5
        self.miu = 10. #4.
        self.g = 1. #2.
        self.delta = 1. #2.
        self.integrate_steps = steps
        self.step_size = step_size


    def integrate(self, x):
        dydt = -self.f * x + self.g * torch.matmul(self.adjacent_matrix, 1 / (1 + torch.exp(self.miu - self.delta * x.T))).T
        return dydt


    def __call__(self, x ):
        h = self.step_size
        k1 = self.integrate(x)
        k2 = self.integrate(x + h * k1 / 2)
        k3 = self.integrate(x + h * k2 / 2)
        k4 = self.integrate(x + h * k3)
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state


    def terminate_point(self, x):
        for _ in range( self.integrate_steps ):
              x = self.__call__(x)

        return x







if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # np.random.seed(123456)

    data_dimension = 20

    "create adjacent_matrix"
    mean_degree = 5
    graph = sf.scale_free( data_dimension, mean_degree, seed=123456 )
    # largest_cc = max( nx.weakly_connected_components(graph), key=len )
    largest_cc = max( nx.connected_components(graph), key=len )
    graph = graph.subgraph( largest_cc )
    # graph.remove_nodes_from(list(nx.isolates(graph)))
    mapping = dict(zip(graph, range(0, len(graph))))
    graph = nx.relabel_nodes(graph, mapping)
    node_num = len(graph)
    print( f'node_num:{node_num}, edge_num:{len(graph.edges())}' )
    pos = nx.spring_layout( graph )

    # nx.draw( graph, pos=pos, node_size=50 )
    # plt.show()

    # # weight = 9.765
    # # B = torch.Tensor( nx.to_numpy_array(graph) ).T * weight

    W = np.random.randn(len(graph.edges())) * 1 + 15.
    source, target = zip(*graph.edges())
    weight_edges = zip(source, target, W)
    graph.add_weighted_edges_from(weight_edges)
    B = nx.to_numpy_array(graph)
    for s, t, w in graph.edges(data=True):
        B[s, t] = w['weight']
    B = torch.Tensor(B).T
    # print(B)

    np.random.seed(123456)

    dynamic = Dynamics( step_size=0.1, steps=2000, adjacent_matrix=B.to(device) )
    points = np.random.uniform(-15, 14, [2000, node_num])
    points = torch.Tensor(points).to(device)
    # print(points)
    #
    #
    # print(len(points))
    result = dynamic.terminate_point(points)#[bs,nodes]
    result = torch.mean( result, dim=-1 )
    result = torch.round(result, decimals=2)
    print( torch.unique(result, return_counts=True) )
