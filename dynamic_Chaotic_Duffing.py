import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx
import itertools as its
import networkx as nx
import scale_free_graph as sf
import math as m


class Dynamics():

    def __init__(self, steps=10000, step_size=0.1):



        self.k = 0.5
        self.A = 0.38
        self.gaba = 0.1

        self.integrate_steps = steps
        self.step_size = step_size




    def integrate(self, input_, t ):  # [bs,nodes*2]
    #
        x, y = input_[:, 0:1 ], input_[:, 1:2 ] #[bs,1], [bs,1]

        d_x = y

        d_y = - self.k * y + x - x**3 + self.A * m.sin( self.gaba * t )

        return torch.cat( [d_x, d_y], dim=-1 ) #[[bs,1],[bs,1]] -- [bs,3]


    def __call__(self, x, t ):
        h = self.step_size
        k1 = self.integrate( x ,t )
        k2 = self.integrate( x + h * k1 / 2,  t+ h/2 )
        k3 = self.integrate( x + h * k2 / 2, t + h/2 )
        k4 = self.integrate( x + h * k3, t + h )
        next_state = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state


    def terminate_point(self, x):
        # def in_basin(self, x, attractor1, attractor2 ):
        for t in range(self.integrate_steps):
            x = self.__call__(x,t)
        return x


    def seq(self, x ):
        seq_ = []
        for t in range(1,self.integrate_steps+1):
            x = self.__call__(x,t)
            seq_.append( torch.unsqueeze(x, dim=0) ) #[1,bs,nodes]
        return torch.transpose( torch.cat( seq_, dim=0 ), 0, 1 ) #[steps,bs,nodes] -- [bs,steps, nodes]




if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    dynamic = Dynamics( step_size=0.1, steps=5000 )

    x = np.linspace(-2.,2, 150).tolist()
    y = np.linspace(-2.5, 2.5, 150).tolist()
    points = np.array( list( its.product(x, y) ) ) # [batch_size, nodes]
    # z = np.zeros([points.shape[0],1])
    # points = np.concatenate([points, z], axis=1)
    points = torch.Tensor(points)
    print(points.shape)


    result = dynamic.seq(points.to(device)).cpu() #[bs,steps, nodes]
    print(result)
    result =  torch.mean( result[:,-100:,0], dim=1 ) #[bs,steps] -- [bs]
    print(result, len(result))
    # attractor = set([tuple(i) for i in result.tolist()])
    # print(attractor)
    label = torch.ones(result.shape[0])
    label[result < 0] = 0
    # label[ result != 0 ] = 1
    print(label)

    points = points.numpy()
    x, y = points[:, 0], points[:, 1]
    plt.scatter( x, y, c=label.numpy(), cmap='Pastel1', s=2, vmin=0, vmax=1 )
    # x, y = np.array(list(attractor))[:, 0], np.array(list(attractor))[:, 1]
    # plt.scatter(x, y, s=200, color='black', edgecolors='black', linewidths=1)

    # plt.xlabel('x',fontsize=17)
    # plt.ylabel('v',fontsize=17)
    #
    # plt.yticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)
    # plt.xticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)
    #
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    #
    # # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Duffing\\boudary_K={K}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Duffing\\only_fix.png', dpi=300, bbox_inches='tight')
    plt.show()


