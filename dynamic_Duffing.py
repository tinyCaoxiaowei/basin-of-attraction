import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx
import itertools as its
import networkx as nx
import scale_free_graph as sf


class Dynamics():

    def __init__(self, steps=10000, step_size=0.1, K=0.1):

        self.K = K

        self.integrate_steps = steps
        self.step_size = step_size



    def integrate( self, input_ ): #[bs,nodes*2]

        x, v = input_[:, 0:1 ], input_[:, 1:2 ] #[bs,1], [bs,1]

        d_x = v #[bs,1]

        d_v = - self.K * v + x - x**3 #[bs,1]

        return torch.cat( [d_x, d_v], dim=-1 ) #[[bs,1],[bs,1]] -- [bs,2]



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

    K=0.5
    dynamic = Dynamics( step_size=0.1, steps=1000, K=K )



    x = np.linspace(-2, 2+0.1, 150).tolist()
    y = np.linspace(-2, 2+0.1, 150).tolist()
    points = list( its.product(x, y) )  # [batch_size, nodes]
    points = torch.Tensor(points)


    result = dynamic.terminate_point(points.to(device)).cpu()
    result = torch.around( result, decimals=1 )
    print(result, len(result))
    # attractor = set([tuple(i) for i in result.tolist()])
    # print(attractor)

    # x, y = zip(*points)
    # plt.scatter( x, y, c=result[:,0], cmap='Pastel1', s=5, vmin=0, vmax=1 )
    x, y = np.array(list(attractor))[:, 0], np.array(list(attractor))[:, 1]
    plt.scatter(x, y, s=200, color='black', edgecolors='black', linewidths=1)

    plt.xlabel('x',fontsize=17)
    plt.ylabel('v',fontsize=17)

    plt.yticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)
    plt.xticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Duffing\\boudary_K={K}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Duffing\\only_fix.png', dpi=300, bbox_inches='tight')
    plt.show()


