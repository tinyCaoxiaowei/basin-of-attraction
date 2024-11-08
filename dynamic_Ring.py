import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx
import itertools as its
import networkx as nx
import scale_free_graph as sf


class Dynamics():

    def __init__(self, adjacent_matrix, steps=10000, step_size=0.1 ):

        self.adjacent_matrix = adjacent_matrix

        self.integrate_steps = steps
        self.step_size = step_size



    def integrate( self, theta ): #[bs,nodes]

        d_theta = torch.sum( self.adjacent_matrix * torch.sin( torch.unsqueeze( theta, 2 ) - torch.unsqueeze( theta, 1 ) ), dim=-1 )
        #[bs,n,1] - [bs,1,n] -- [bs,n,n]  [n,n] * [bs,n,n] -- [bs,n,n]  [bs,n,n] -- [bs,n]

        return d_theta



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

    def seq(self, x ):
        seq = []
        for _ in range(self.integrate_steps):
            x = self.__call__(x)
            seq.append( torch.unsqueeze(x, dim=0) ) #[1,bs,nodes]
        return torch.transpose( torch.cat( seq, dim=0 ), 0, 1 ) #[steps,bs,nodes] -- [bs,steps, nodes]


if __name__ == "__main__":


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dimension = 10
    g = nx.watts_strogatz_graph( data_dimension, 2, p=0  )
    # pos = nx.spring_layout( g )
    # nx.draw( g, pos=pos, node_size=50 )
    # plt.show()

    J = torch.Tensor( nx.to_numpy_array(g) ).T.to(device)

    dynamic = Dynamics( J, step_size=0.1, steps=3000 )

    theta1 = np.linspace( -np.pi, np.pi, 100 ).tolist()
    theta2 = np.linspace(-np.pi, np.pi,  100).tolist()
    points = np.array( list( its.product(theta1,theta2) ) )  # [batch_size, 2]
    theta_others = np.ones([points.shape[0], data_dimension-2]) * np.pi
    points = np.concatenate([points, theta_others], axis=1)
    points = torch.Tensor( points )
    print(points.shape)

    result = dynamic.terminate_point( points.to(device) ).cpu()
    result = torch.round(  result[:, :] , decimals=5)
    # result = dynamic.seq(points.to(device)).cpu()
    # result = torch.round( result[:,:,:], decimals=2 )
    print( result, len(result) )
    # # # attractor = set([tuple(i) for i in result.tolist()])
    # # # print(attractor)
    # label  = torch.ones( result.shape[0] )
    # label[ result==0 ] = 0
    # # label[ result != 0 ] = 1
    # print( label )
    # #
    print( np.unique( result[:, 0].numpy(),return_counts=True ))
    print(np.unique(np.sin(result[:, 0].numpy()), return_counts=True))
    y,x = points[:,0], points[:,1]
    c = np.abs(np.sin(result[:,0].numpy()))
    # plt.scatter( x, y, c=np.abs(np.sin(result[:,0].numpy())), cmap='Pastel1', s=5 ) #, vmin=0, vmax=2 )
    plt.scatter(x, y, c=result[:, 0].numpy(), cmap='Pastel1', s=5)  # , vmin=0, vmax=2 )
    # # # x, y = np.array(list(attractor))[:, 0], np.array(list(attractor))[:, 1]
    # # # plt.scatter(x, y, s=200, color='black', edgecolors='black', linewidths=1)
    # # #
    # plt.xlabel('$\\theta$', fontsize=17)
    # plt.ylabel('$\omega$', fontsize=17)
    # #
    # plt.yticks(np.arange(-10, 7 + 0.1, 2), fontsize=17)
    # plt.xticks(np.arange(-5, 3 + 0.1, 1), fontsize=17)
    # #
    # plt.ylim(-10, 7 )
    # plt.xlim(-5, 3)
    # # #
    # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Kuramoto\\groundtruth.png', dpi=300, bbox_inches='tight')
    # # # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\Duffing\\only_fix.png', dpi=300, bbox_inches='tight')
    plt.show()


