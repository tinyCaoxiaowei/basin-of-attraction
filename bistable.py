import matplotlib.pyplot as plt
import numpy as np
import torch
import collections as cls
import itertools as its
import os



class rk4():
    def __init__(self, J, epoch = 1000, device='cpu' ):
        # self.device = torch.device("cuda" if device else "cpu")
        self.J = J.to(device)
        self.epoch = epoch
        self.s = 2.5
        self.g = 0.02

    def dynamics( self, x ):
        dy = -x +  self.s * torch.tanh( x ) + self.g * torch.matmul(  self.J , torch.tanh(x).T ).T
        ## x.shpae=[bs,nodes];
        # [bs,nodes].T -- [nodes,bs]    [nodes,nodes] * [nodes,bs] -- [nodes, bs]  [nodes, bs].T -- [bs, nodes]
        return dy




    def in_basin(self, x ):
    # def in_basin(self, x, attractor1, attractor2 ):
        for _ in range(self.epoch):
            x = self.__call__( x )

            # if  torch.sqrt( torch.sum( (x - attractor1 )**2 ) ) < 1e-4:
            #     return x, 0, _
            # elif torch.sqrt( torch.sum( (x - attractor2 )**2 ) ) < 1e-4:
            #     return x, 1, _

        # return  torch.mean( torch.abs( x - attractor ) ), x
        return x




    def __call__(self, x ):
        # rf4
        h = 0.1
        k1 = self.dynamics( x )
        k2 = self.dynamics( x + h * k1 / 2 )
        k3 = self.dynamics( x + h * k2 / 2 )
        k4 = self.dynamics( x + h * k3 )
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return next_state



if __name__ == '__main__':

    # HOME = os.environ['HOME']

    n = 2
    epoch = 1000

    J = torch.zeros((n, n))
    J[0,1] = 16.96
    J[1,0] = 12.86

    # attractor1 = torch.Tensor([[16.0137, 12.8286]])
    # attractor2 = torch.Tensor([[0.0008, 0.0006]])
    env = rk4( J, epoch )
    # x = env.in_basin( torch.Tensor([[5,2]]) )
    # print( x )

    x = np.arange( -100, 100.1, 1 ).tolist()
    points =  list( its.product( x, repeat=2 ) )  #[batch_size, nodes]

    target =  env.in_basin( torch.Tensor(points) ).numpy()
    print(set(tuple(i) for i in target))

    x,y = zip( *points )
    plt.scatter( x, y, cmap ='Pastel1', c=target[:,1] )
    # plt.scatter( attractor1[0,0], attractor1[0,1], s=150, color='black', edgecolors ='black',linewidths =1 )
    # plt.scatter( attractor2[0,0], attractor2[0,1], s=150, color='black', edgecolors ='black',linewidths =1 )
    print( np.unique(target[:,1]) )
    plt.xlabel( 'node_1' )
    plt.ylabel( 'node_2' )

    # plt.savefig(  HOME + '/boundary/boudary_bistable_2nodes.png', dpi=300  )
    plt.savefig( 'C:\\茹小磊\\perturbation\\fig\\' + 'boudary_bistable_2nodes.png', dpi=300)
    # # plt.show()