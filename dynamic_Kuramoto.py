import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx
import itertools as its
import networkx as nx
import scale_free_graph as sf


class Dynamics():

    def __init__(self, steps=10000, step_size=0.1 ):

        self.alpha = 0.1
        self.P = 1
        self.C = -1
        self.K = 1.15

        self.integrate_steps = steps
        self.step_size = step_size



    def integrate( self, input_ ): #[bs,nodes*2]

        omega, zeta, zeta_c = input_[:, 0:1 ], input_[:, 1:2 ], input_[:, 2:3 ] #[bs,1], [bs,1]
        # print(input_.shape)
        d_omega = - self.alpha * omega + self.P + self.K * torch.sin( zeta_c - zeta ) #[bs,1]
        d_zeta = omega
        d_zeta_c = self.C + self.K * torch.sin( zeta - zeta_c )

        return torch.cat( [d_omega, d_zeta, d_zeta_c], dim=-1 ) #[[bs,1],...,[bs,1]] -- [bs,4]
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
        for _ in range(self.integrate_steps):
            x = self.__call__(x)
        return x



if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dynamic = Dynamics( step_size=0.1, steps=2000 )

    # epsilon = 0.4
    # beta = 0.15
    # K = 0.01

    omega = np.linspace( -10, 7, 150 ).tolist()
    zeta = np.linspace( -np.pi, np.pi, 150 ).tolist()
    points = np.array( list( its.product(omega,zeta) ) )  # [batch_size, 2]
    zeta_c = np.zeros( [points.shape[0],1] )
    points = np.concatenate([points, zeta_c], axis=1)
    points = torch.Tensor(points)
    print(points.shape)

    result = dynamic.terminate_point( points.to(device) ).cpu()
    result = torch.round( result[:,0], decimals=1 )
    print( result, len(result) )
    # # attractor = set([tuple(i) for i in result.tolist()])
    # # print(attractor)
    label  = torch.ones( result.shape[0] )
    label[ result==0 ] = 0
    # label[ result != 0 ] = 1
    print( label )
    #
    points = points.numpy()
    y,x = points[:,0], points[:,1]
    plt.scatter( x, y, c=label.numpy(), cmap='Pastel1', s=5 ) #, vmin=0, vmax=2 )
    # # x, y = np.array(list(attractor))[:, 0], np.array(list(attractor))[:, 1]
    # # plt.scatter(x, y, s=200, color='black', edgecolors='black', linewidths=1)
    # #
    plt.xlabel('$\\theta$', fontsize=17)
    plt.ylabel('$\omega$', fontsize=17)
    #
    plt.yticks(np.arange(-10, 7 + 0.1, 3), fontsize=17)
    plt.xticks(np.arange(-3, 3 + 0.1, 1), fontsize=17)
    #
    plt.ylim(-10, 7 )
    plt.xlim(-np.pi, np.pi)
    # #
    plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Kuramoto\\groundtruth.png', dpi=300, bbox_inches='tight')
    # # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Duffing\\only_fix.png', dpi=300, bbox_inches='tight')
    plt.show()


