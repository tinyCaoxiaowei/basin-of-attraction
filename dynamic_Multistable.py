import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx
import itertools as its
import networkx as nx
import scale_free_graph as sf


class Dynamics():

    def __init__(self, steps=10000, step_size=0.1 ):

        self.alpha = -0.02
        self.epsilon = 0.4
        self.beta = 0.15
        self.K = 1

        self.integrate_steps = steps
        self.step_size = step_size



    def integrate( self, input_ ): #[bs,nodes*2]

        x, y, z, u = input_[:, 0:1 ], input_[:, 1:2 ], input_[:, 2:3 ], input_[:, 3:4 ] #[bs,1], [bs,1]
        # x, y, z, u = input_[:, 0], input_[:, 1], input_[:, 2], input_[:, 3]  # [bs], [bs]
        # print(input_.shape)
        dx = y #[bs,1]
        dy = z
        dz = -y + 3* (y **2) - x**2 - x* z + self.alpha + self.epsilon * u
        du = - self.K * u - self.epsilon*( z- self.beta )

        return torch.cat( [dx, dy, dz, du], dim=-1 ) #[[bs,1],...,[bs,1]] -- [bs,4]
        # return torch.vstack([dx, dy, dz, du]).T  # [bs,...,bs] -- [4,bs]



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
            # print(x)
            # if  torch.sqrt( torch.sum( (x - attractor1 )**2 ) ) < 1e-4:
            #     return x, 0, _
            # elif torch.sqrt( torch.sum( (x - attractor2 )**2 ) ) < 1e-4:
            #     return x, 1, _

        # return  torch.mean( torch.abs( x - attractor ) ), x
        return x
    # J = torch.zeros((n, n))
    # J[0,1] = 16.96
    # J[1,0] = 12.86



if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dynamic = Dynamics( step_size=0.1, steps=10000 )

    # epsilon = 0.4
    # beta = 0.15
    # K = 0.01

    x = np.linspace( 0, 15, 150 ).tolist()
    y = np.linspace( -35, 5, 150 ).tolist()
    points = np.array( list( its.product(x,y) ) )  # [batch_size, 2]
    z = np.zeros( [points.shape[0],1] ) # [batch_size, 1]
    u = np.ones( [points.shape[0],1] ) # [batch_size, 1]#np.ones( [points.shape[0],1]  ) * epsilon * beta / K # [batch_size, 1]
    points = np.concatenate( [points,z,u], axis=1 )
    points= torch.Tensor(points)
    print(points.shape)

    result = dynamic.terminate_point(points.to(device)).cpu()
    result = torch.round( result, decimals=2 )
    print( result, len(result) )
    # attractor = set([tuple(i) for i in result.tolist()])
    # print(attractor)
    label  = torch.ones( result.shape[0] )
    label[ torch.isnan( result[:,0] ) ] = 0
    label[ result[:,0] == 0.06 ] = 2
    print( label )

    points = points.numpy()
    x,y = points[:,0], points[:,1]
    plt.scatter( x, y, c=label.numpy(), cmap='Pastel1', s=5 ) #, vmin=0, vmax=2 )
    # x, y = np.array(list(attractor))[:, 0], np.array(list(attractor))[:, 1]
    # plt.scatter(x, y, s=200, color='black', edgecolors='black', linewidths=1)
    #
    plt.xlabel('x',fontsize=17)
    plt.ylabel('y',fontsize=17)
    #
    plt.xticks(np.arange(0, 15 + 0.1, 2), fontsize=17)
    plt.yticks(np.arange(-35, 5 + 0.1, 6), fontsize=17)
    #
    plt.xlim(0, 15)
    plt.ylim(-35, 5)
    #
    plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Multistable\\groundtruth.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Duffing\\only_fix.png', dpi=300, bbox_inches='tight')
    plt.show()


