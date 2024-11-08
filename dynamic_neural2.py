import matplotlib.pyplot as plt
import numpy as np
import torch
import collections as cls
import itertools as its
import os


'''return run times of dynamic'''
class rk4():
    def __init__(self, J, device='cpu' ):
        self.mu, self.delta = 10, 1
        # self.device = torch.device("cuda" if device else "cpu")
        self.J = J.to(device)
        self.epoch = 1000


    def dynamics( self, x ):
        dy = -x + torch.matmul( self.J, 1 / ( 1 + torch.exp(self.mu - self.delta *x) ).T ).T
        ## x.shpae=[1,N]; u.shape=[1,N]
        return dy





    def in_basin( self, x, attractor1, attractor2 ):
        results = []
        for _ in range(self.epoch):
            temp = self.__call__( x )
            results.append(temp.numpy() ) #[bs, n ]
            x = temp

        results = np.transpose( np.array( results ), (1,2,0) ) #[ [bs,n], [bs,n] ,..., [bs,n] ] -- [ t,bs,n ] -- [bs,n,t]
        labels, run_times, targets = [], [], []
        for ts in results: #[n,t]
            ts_diff = np.diff( ts.sum( axis=0 ) )
            temp = np.where( ts_diff == 0 )[0]
            if len(temp) > 0: # ts_diff has at least one '0' element
                rt = temp[0]
                target = ts[ :, rt ] #[n]
                rt = rt + 1
            else:
                rt = ts.shape[-1]
                target = ts[ :, -1 ]

            if np.sqrt(np.sum((target - attractor1) ** 2)) <= np.sqrt(np.sum((target - attractor2) ** 2)):  # close to attractor1
                labels.append(0)
            else:
                labels.append(1)

            targets.append( target )
            run_times.append( rt )


        labels, run_times, targets = np.array( labels ), np.array( run_times ), np.array( targets )
        return targets, labels, run_times





    def __call__(self, x ):
        # rf4
        h = 0.5
        k1 = self.dynamics( x )
        k2 = self.dynamics( x + h * k1 / 2 )
        k3 = self.dynamics( x + h * k2 / 2 )
        k4 = self.dynamics( x + h * k3 )
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return next_state



if __name__ == '__main__':

    HOME = os.environ['HOME']

    n = 2
    J = torch.zeros((n, n))
    J[0,1] = 16.96
    J[1,0] = 12.86

    attractor1 = np.array([[16.0137, 12.8286]])
    attractor2 = np.array([[0.0008, 0.0006]])
    env = rk4( J )
    # x = env.in_basin( torch.Tensor([[5,2]]) )
    # print( x )

    x = np.arange( -5, 30.1, 0.2 ).tolist()
    points =  list( its.product( x, repeat=2 ) )

    target, labels, run_times =  env.in_basin( torch.Tensor(points), attractor1, attractor2 )


    x,y = zip( *points )
    # plt.scatter( x, y, cmap ='Pastel1', c=target[:,1] )
    plt.scatter( x, y, cmap='plasma', c=run_times )
    plt.colorbar()
    plt.scatter( attractor1[0,0], attractor1[0,1], s=150, color='black', edgecolors ='black',linewidths =1 )
    plt.scatter( attractor2[0,0], attractor2[0,1], s=150, color='black', edgecolors ='black',linewidths =1 )
    print( np.unique(target[:,1]) )
    print( run_times )
    plt.xlabel( 'node_1' )
    plt.ylabel( 'node_2' )

    plt.savefig(  HOME + '/boundary/runtimes_nueral_2nodes.png', dpi=300  )
    # # plt.show()