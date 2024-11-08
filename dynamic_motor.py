import scipy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx
from networkx import barabasi_albert_graph
import networkx as nx
import scale_free_graph as sf
import itertools as its
import os


class Runge_Kutta_for_neural_system():
    def __init__(self, steps=10000, adjacent_matrix=None, step_size=0.1):

        self.adjacent_matrix = adjacent_matrix

        self.alpha = 0.1
        self.K = 8.

        self.node_num = self.adjacent_matrix.shape[0]
        self.P = np.ones( self.node_num ) #[nodes]
        index = np.random.choice( self.node_num, int(self.node_num/2), replace=False )
        self.P[index] = -1
        self.P = torch.Tensor( self.P ).to( adjacent_matrix.device )


        self.integrate_steps = steps
        self.step_size = step_size



    def integrate( self, input_ ): #[bs,nodes*2]

        theta, omega = input_[:, :self.node_num], input_[:, -self.node_num:] #[bs,nodes],  [bs,nodes]

        d_theta = omega #[bs,nodes]

        d_omega = - self.alpha * omega + self.P \
                  - self.K * torch.sum( self.adjacent_matrix * torch.sin( torch.unsqueeze( theta, 2 ) - torch.unsqueeze( theta, 1 ) ), dim=-1 )
        #[bs,n,1] - [bs,1,n] -- [bs,n,n]  [n,n] * [bs,n,n] -- [bs,n,n]  [bs,n,n] -- [bs,n]

        return torch.cat( [d_theta, d_omega], dim=-1 ) #[[bs,n],[bs,n]] -- [bs,n*2]


    def __call__(self, x ):
        h = self.step_size
        k1 = self.integrate(x)
        k2 = self.integrate(x + h * k1 / 2)
        k3 = self.integrate(x + h * k2 / 2)
        k4 = self.integrate(x + h * k3)
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state


    def terminate_point(self, x):

        for _ in range(self.integrate_steps):
              x = self.__call__(x)

        return x






if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(1234)

    data_dimension = 100
    "create adjacent_matrix"
    # graph = barabasi_albert_graph(n=data_dimension, m=1, seed=3)
    mean_degree = 2.7
    # graph = sf.scale_free( data_dimension, mean_degree, seed=123 )
    graph = nx.erdos_renyi_graph(n=100, p=mean_degree / (data_dimension - 1), seed=234)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    mapping = dict(zip(graph, range(0, len(graph))))
    graph = nx.relabel_nodes(graph, mapping)
    print(f'node_num:{len(graph.nodes())}, edge_num:{len(graph.edges())}')
    node_num = len(graph)
    # pos = networkx.spring_layout(graph)
    # networkx.draw_networkx( graph, pos=pos, with_labels=True, node_size=30 )
    # plt.show()
    # weight = 9.765
    # B = torch.Tensor( nx.to_numpy_array(graph) ).T * weight
    B = nx.to_numpy_array(graph)

    # W = numpy.random.randn(len(graph.edges()))*20 + 20.
    # source, target = zip(*graph.edges())
    # weight_edges = zip(source, target, W)
    # graph.add_weighted_edges_from(weight_edges)
    # for s, t, w in graph.edges(data=True) :
    #     B[s,t] = w['weight']
    B = torch.Tensor(B).T
    print(B)
    dynamic = Runge_Kutta_for_neural_system(step_size=0.1, steps=2000, adjacent_matrix=B.to(device))


    omega1 = np.linspace(-np.pi, np.pi, 100).tolist()
    theta1 = np.linspace(-15, 15, 100).tolist()
    temp = np.array(list(its.product(omega1, theta1)))  # [batch_size, 2]
    omega1, theta1 = temp[:,0], temp[:,1]
    points = np.zeros( [temp.shape[0], node_num * 2] )
    points[:,0] = theta1
    points[:, node_num] = omega1
    points = torch.Tensor(points)
    # points[:,130] = 0.0000000
    print(points)


    print(len(points))
    result = dynamic.terminate_point(points.to(device)).cpu() #[bs,nodes*2]
    result = torch.round(result, decimals=3)
    omega = result[:, -node_num:].numpy()

    print(omega)
    print(result[0], result[1000])
    # print( np.unique( result[node_num+1].numpy(), return_counts=True ) )
    label = np.ones(omega.shape[0])
    diff = np.abs( np.diff( omega, axis=1  )).sum( axis=1 )
    print(diff)
    label[diff == 0] = 0
    # label[ result != 0 ] = 1
    # print(label)

    plt.scatter( temp[:,0], temp[:,1], c= label , cmap='tab20c', s=5 )
    HOME = os.environ['HOME']
    plt.savefig( HOME + f'/basin/fig/123.png', dpi=300, bbox_inches='tight')



