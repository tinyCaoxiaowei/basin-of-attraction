import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx
import itertools as its
import networkx as nx
import scale_free_graph as sf
from utils import return_label_VSC


class Dynamics():

    def __init__( self, steps=10000, step_size=0.1 ):

        self.I = 0.4
        self.D = 0.39
        self.alpha = 0.7

        self.integrate_steps = steps
        self.step_size = step_size



    def integrate( self, input_ ): #[bs,2]

        theta, omega = input_[:, :1], input_[:, -1:]  # [bs,1],  [bs,1]

        d_theta = omega  # [bs,1]
        
        d_omega = self.I - torch.sin( theta ) - ( self.alpha * torch.cos( theta ) - self.D ) * omega

        return torch.cat( [d_theta, d_omega], dim=-1 ) #[[bs,1],[bs,1]] -- [bs,2]



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

        return x



if __name__ == "__main__":


    dynamic = Dynamics( step_size=0.05, steps=1000 )

    # np.random.seed(1234)
    # points = np.random.uniform( -2, 2, [100, 2] )
    # # points = numpy.array([[-1.5,-0]])
    # print(points)

    # print(len(points))
    # result = dynamic.terminate_point( torch.Tensor(points) ).numpy() #[bs,nodes*2]
    # result = np.around(result, decimals=1)
    # # result = result.astype('int')
    #
    # print(result)


    x = np.linspace( -2, 3.5, 150 ).tolist()
    y = np.linspace( -4., 2., 150 ).tolist()
    points = list( its.product(x, y) )  # [batch_size, nodes]


    # result = dynamic.terminate_point( torch.Tensor(points) ).numpy()
    # print(result)
    points = torch.Tensor( points )
    result = return_label_VSC( points, dynamic ) #[bs]
    # result[result]
    # print(result)
    # attractor = set([tuple(i) for i in result.tolist()])
    # print(attractor)


    x, y = zip(*points)
    plt.scatter(x, y, cmap='Pastel1', c=result)
    # x, y = np.array(list(attractor))[:, 0], np.array(list(attractor))[:, 1]
    # plt.scatter(x, y, s=100, color='black', edgecolors='black', linewidths=1)

    plt.xlabel( '$\\theta$' , fontsize=17)
    plt.ylabel( '$\omega$' , fontsize=17)



    plt.xticks( np.arange( -2, 3.5 + 0.1, 1 ), fontsize=17 )
    plt.yticks( np.arange( -4., 2.+ 0.1, 1 ), fontsize=17 )

    plt.xlim(-2, 3.5)
    plt.ylim(-4., 2.)

    plt.savefig( f'C:\\茹小磊\perturbation\\fig\\VSC\\groundtruth.png', dpi=300 , bbox_inches='tight')
    plt.show()


