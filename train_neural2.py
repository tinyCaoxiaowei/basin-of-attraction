import matplotlib.pyplot as plt
import numpy as np
import torch
import collections as cls
import itertools as its
import os
import dynamic_Neural as dyn
from model import classifier
from sklearn.metrics import accuracy_score
import dummy_samples as ds
import util



device = torch.device('cuda:0')
train_epochs = 1000
al_epochs = 10  #number of active learning

lower, upper = -5, 30

n = 2  # number of nodes
J = torch.zeros((n, n))
J[0,1] = 16.96
J[1,0] = 12.86

attractor1 = torch.Tensor([[16.0137, 12.8286]])
attractor2 = torch.Tensor([[0.0008, 0.0006]])
env = dyn.rk4( J )


x = np.arange( lower, upper +0.1, 0.2 ).tolist()
bgp =  np.array(list( its.product( x, repeat=2 ) ))  #background_points
bg_targets =  env.in_basin( torch.Tensor(bgp) ).numpy()



np.random.seed(123)

# inital_num = 20
inital_fre = 5
al_num = 20

# inital_samples = torch.Tensor( np.random.rand( inital_num, n ) * (upper - lower) + lower )  #[0,1] mapping to [-5,30]
inital_x = np.linspace( lower, upper, inital_fre ).tolist()
inital_samples = torch.Tensor( list( its.product( inital_x, repeat=2 ) ) )

targets =  env.in_basin( inital_samples ) #[inital_num, n ]
# print(targets)
inital_labels = util.humna_annotation( targets, attractor1, attractor2 ) #[inital_num]
# print( inital_samples.shape,  targets.shape, inital_labels.shape )


model = classifier( n ).to( device )
op = torch.optim.Adam( model.parameters(), lr=0.0005 )

samples, labels = inital_samples, inital_labels
util.train( samples, labels, model, op, train_epochs, device, stage='initial' )
# model = util.train( samples, labels, train_epochs, device, stage='initial' )
util.plot_sampling( samples, labels, bgp, bg_targets, lower, upper )
util.plot_approximate_boundary( bgp, model, device )  #plot approximate boundary of classifier using background points

ACC_test = []
for _ in range( al_epochs ):

    print( f'epoch of active_learning:{_}' )

    al_samples = torch.Tensor( np.random.rand( al_num, n ) * (upper - lower) + lower ) #[inital_num, n]

    # temp = []
    # for sample in al_samples:
    #     temp.append( ds.generate( sample, model, lower, upper, device ) )
    # al_samples = torch.Tensor( np.array(temp) )  #[inital_num, n] on cpu
    al_samples = ds.generate( al_samples, model, lower, upper, device )
    al_targets = env.in_basin( al_samples ) #[inital_num, n ]
    al_labels = util.humna_annotation( al_targets, attractor1, attractor2 ) #[inital_num]
    # print( al_samples.shape,  targets.shape, al_labels.shape )
    # print( f'al_samples:{al_samples}, al_targets:{al_targets}, al_labels:{al_labels}' )

    ACC_test.append( util.train(samples, labels,
                                model, op, train_epochs, device, stage='active_learning',
                                new_samples=al_samples, new_labels=al_labels ) )
    # model= util.train(samples, labels,
    #                            train_epochs, device, stage='active_learning',
    #                            new_samples=al_samples, new_labels=al_labels)

    samples = torch.cat([samples, al_samples], dim=0)
    labels = torch.cat([labels, al_labels], dim=0)
    print( ACC_test )

    util.plot_sampling(samples, labels,  bgp, bg_targets, lower, upper, name=f'AL_epoch{_}')
    util.plot_approximate_boundary( bgp, model, device, name=f'AL_epoch{_}' )
