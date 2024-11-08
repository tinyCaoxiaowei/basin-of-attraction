import matplotlib.pyplot as plt
import numpy as np
import torch
import collections as cls
import itertools as its
import os

HOME = os.environ['HOME']

class rk4():
    def __init__(self, J, device='cpu' ):
        self.s = 1.5
        self.g = 2.1
        # self.device = torch.device("cuda" if device else "cpu")
        self.J = J.to(device)
        self.epoch = 2000


    def dynamics( self, x, u=0. ):
        dx = -x + self.s * torch.tanh(x) + self.g * torch.matmul( self.J, torch.tanh(x).T).T + u
        ## x.shpae=[1,N]; u.shape=[1,N]
        return dx


    def potential_energy(self, x ):
        v = -x**2/2 + self.s * ( x - torch.log(torch.tanh(x) + 1 + 1e-32 ) ) \
            + self.g * torch.matmul( self.J, torch.tanh(x.T)).T * x
        return -torch.sum( v )


    # def in_basin(self, x, attractor ):
    def in_basin(self, x):
        for _ in range(self.epoch):
            x = self.__call__( x )

        # if torch.mean( torch.abs( x - attractor ) ) <1:
        #     return True
        # else:
        #     return False

        # return  torch.mean( torch.abs( x - attractor ) ), x
        return x




    def __call__(self, x, u=0.):
        # rf4
        h = 1
        k1 = self.dynamics( x, u )
        k2 = self.dynamics( x + h * k1 / 2, u)
        k3 = self.dynamics( x + h * k2 / 2, u)
        k4 = self.dynamics( x + h * k3, u)
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return next_state



if __name__ == '__main__':

    n = 2
    J = torch.zeros((n, n))
    J[0,1] = 0.76983
    J[1,0] = 1.1425

    attractor1 = torch.Tensor([[3.1093, 3.8885]])
    attractor2 = torch.Tensor([[-3.1093, -3.8885]])
    env = rk4(J)
    # diff, x = env.in_basin(torch.Tensor([[-10,10]]), attractor)
    # print(diff, x)
    # print(env.potential_energy(x), env.potential_energy(torch.Tensor([[3, 5]])))  # 观察势能， 是否是吸引子最小

    x = np.arange( -10,10.4, 0.5 ).tolist()
    points =  list( its.product( x, repeat=2 ) )

    target =  env.in_basin( torch.Tensor(points) ).numpy()

    # np.save(  'C:\\茹小磊\\perturbation\\data\\boundary.npy', target )
    # print(target)

    x,y = zip( *points )
    plt.scatter( x, y, cmap ='jet', c=target[:,1] )
    plt.scatter( attractor1[0,0], attractor1[0,1], s=100, color='black' )
    plt.scatter( attractor2[0,0], attractor2[0,1], s=100, color='black' )
    print( np.unique(target[:,1]) )
    plt.xlabel( 'node_1' )
    plt.ylabel( 'node_2' )

    plt.savefig(  HOME + '/boundary/boudary_2nodes.png', dpi=300  )
    # plt.show()