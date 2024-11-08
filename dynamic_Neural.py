import matplotlib.pyplot as plt
import numpy as np
import torch
import collections as cls
import itertools as its
import os



class Dynamics():
    def __init__(self, J, steps=10000, step_size=0.1 ):
        self.mu, self.delta = 10, 1
        # self.device = torch.device("cuda" if device else "cpu")
        self.J = J

        self.integrate_steps = steps
        self.step_size = step_size


    def integrate( self, x ):
        dy = -x + torch.matmul( self.J, 1 / ( 1 + torch.exp(self.mu - self.delta *x) ).T ).T
        ## x.shpae=[bs,nodes];
        # [bs,nodes].T -- [nodes,bs]    [nodes,nodes] * [nodes,bs] -- [nodes, bs]  [nodes, bs].T -- [bs, nodes]
        return dy


    def __call__(self, x ):
        # rf4
        h = self.step_size
        k1 = self.integrate(x)
        k2 = self.integrate(x + h * k1 / 2)
        k3 = self.integrate(x + h * k2 / 2)
        k4 = self.integrate(x + h * k3)
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return next_state



    def terminate_point(self, x ):
    # def in_basin(self, x, attractor1, attractor2 ):
        for _ in range(self.integrate_steps):
            x = self.__call__( x )

            # if  torch.sqrt( torch.sum( (x - attractor1 )**2 ) ) < 1e-4:
            #     return x, 0, _
            # elif torch.sqrt( torch.sum( (x - attractor2 )**2 ) ) < 1e-4:
            #     return x, 1, _

        # return  torch.mean( torch.abs( x - attractor ) ), x
        return x








if __name__ == '__main__':

    # HOME = os.environ['HOME']

    n = 2
    epoch = 1000

    J = torch.zeros((n, n))
    J[0,1] = 16.96
    J[1,0] = 12.86

    attractor1 = torch.Tensor([[16.0137, 12.8286]])
    attractor2 = torch.Tensor([[0.0008, 0.0006]])
    env = Dynamics( J, epoch )
    # x = env.in_basin( torch.Tensor([[5,2]]) )
    # print( x )

    x = np.arange( -5, 25.1, 0.2 ).tolist()
    points =  list( its.product( x, repeat=2 ) )  #[batch_size, nodes]

    target =  env.terminate_point( torch.Tensor(points) ).numpy()


    x,y = zip( *points )
    plt.scatter( x, y, cmap ='Pastel1', c=target[:,1] )
    plt.scatter( attractor1[0,0], attractor1[0,1], s=100, color='black', edgecolors ='black',linewidths =1 )
    plt.scatter( attractor2[0,0], attractor2[0,1], s=100, color='black', edgecolors ='black',linewidths =1 )
    print( np.unique(target[:,1]) )
    plt.xlabel( 'node_1' )
    plt.ylabel( 'node_2' )

    plt.savefig(  'C:\\茹小磊\\perturbation\\fig\\neural\\groundtruth.png', dpi=300  )
    plt.show()