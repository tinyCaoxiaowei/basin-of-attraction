import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools as its
from utils import *




class Dynamics():
    def __init__(self, J,  steps=10000, step_size=0.1):
        self.J = J
        self.s = 25
        self.g = 4

        self.integrate_steps = steps
        self.step_size = step_size

    def integrate(self, x):
        dy = -x + self.s * torch.tanh( x ) + self.g * torch.matmul(  self.J , torch.tanh(x).T ).T
        return dy


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

    np.random.seed(123)

    J = np.random.random([2,2]) * 10 # sampling from N(10,1)
    # J = np.ones( [2,2] ) * 10
    J = J - np.diag(np.diag(J) ) #elements in diag
    J = torch.Tensor(J)
    dynamic = Dynamics( step_size=0.1, steps=1000, J=J )

    x = np.arange(-50, 50+0.1, 0.4).tolist()
    points = list(its.product(x, repeat=2))  # [batch_size, nodes]
    result = dynamic.terminate_point(torch.Tensor(points)).numpy()
    result = np.around(result, decimals=3)
    attractor = set( [ tuple(i) for i in result.tolist() ] )
    print(attractor)

    # result = np.around(result, decimals=3)
    # print( result )
    # print(np.unique(result))
    #
    x, y = zip(*points)
    # plt.scatter(x, y, cmap='Pastel1', c=)
    # fp_x, fp_y = zip(*list(set(tuple(i) for i in result)))
    # # fig = plt.figure(figsize=(16, 8),  dpi=500)
    # plt.scatter(fp_x, fp_y, color='black', edgecolors='black', linewidths=1)
    # plt.scatter([0], [0], color='green', edgecolors='green', linewidths=2, marker="*")
    # plt.scatter( attractor2[0,0], attractor2[0,1], s=150, color='black', edgecolors ='black',linewidths =1 )
    plt.scatter(x, y, cmap='Pastel1', c=result[:,0] )
    x,y = np.array(list( attractor ) )[:, 0 ], np.array(list( attractor ) )[:, 1 ]
    plt.scatter( x, y, s=100, color='black', edgecolors='black', linewidths=1)
    # # print(numpy.unique(result[:, 0]))
    plt.xlabel('node_1')
    plt.ylabel('node_2')
    plt.savefig(  'C:\\茹小磊\\perturbation\\fig\\Bistable\\attractor.png', dpi=300  )
    plt.show()


