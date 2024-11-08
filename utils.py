import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import collections as cls
import os




"set seed"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


"evaluate acc"
def evaluate_acc( model, dataloader, device ):
    model.eval()

    Logit, Label = [], []
    for v, data in enumerate(dataloader):
        x, y = data #[bs,nodes], [bs]
        x, y = x.to(device), y.long().to(device)
        with torch.no_grad():
            logits = model(x)
        Logit.append( logits )
        Label.append( y )

    Logit = torch.cat( Logit, dim=0 )
    Label = torch.cat( Label, dim=0 )

    Pre_Label = Logit.argmax(dim=1)
    correct_numbers = torch.eq( Pre_Label, Label ).sum().float().item()
    accuracy = correct_numbers / len(dataloader.dataset)

    prob = torch.nn.functional.softmax( Logit, dim=-1 )
    uncertain = - torch.sum( prob * torch.log( prob + 1e-12 ), dim=-1 ) #[bs,class] - [bs]

    return accuracy, Pre_Label.cpu(), uncertain.cpu()



"compute the steps back to attractors"
def evaluate_dis( x, dynamic, dynamic_name = 'Neural' ):

    seq = []
    for _ in range( dynamic.integrate_steps ):
        temp = dynamic.__call__(x) #[bs,nodes]
        seq.append( temp.unsqueeze( dim=0) ) #[1,bs,nodes]
        x = temp


    seq = torch.cat( seq, dim=0 ).cpu() #[steps, bs, nodes]
    seq = torch.transpose( seq, 0, 1 ) #[bs, steps, nodes]
    # print( seq.shape)
    run_steps = cls.defaultdict(list)
    if dynamic_name == 'Neural':
        seq = torch.round( torch.mean(seq, dim=2), decimals=2 ).numpy()  #[bs, steps]
        for ss in seq: # [steps]
            # print(ss[-1])
            if ss[-1] == 0:
                key = '=0'
            else:
                key = '!=0'
            diff = np.abs( np.diff(ss) )
            steps = np.where( diff < 0.01 )[0][0]
            run_steps[key].append( steps )

    run_steps = dict(run_steps)
    for key, value in run_steps.items():
        # run_steps[key] = [np.mean( value ), np.std(value)]
        run_steps[key] = np.mean(value)

    return run_steps



"dataset class for make dataset"
class MyDataset(Dataset):
    def __init__(self, dataset):

        super(MyDataset, self).__init__()
        self.data_list = dataset

    def __len__(self):
        return len( self.data_list )

    def __getitem__(self, index):
        pic, label = self.data_list[index]
        if type(pic) is np.ndarray:
            pic = torch.from_numpy(pic).float()
        if type(label) is np.ndarray:
            label = torch.from_numpy(np.array(label)).long()
        return pic, label


" dynamic euqation iterations "
"assume there are no saddle points or flux ,etc, only stable fixed point"
def return_label_from_dynamics( sample, dynamic ):

    terminate_points = dynamic.terminate_point( sample )
    terminate_points = torch.mean(terminate_points, dim=-1)
    # result = torch.round(terminate_points, decimals=2)
    terminate_points[abs(terminate_points) < 1e-2] = 0.
    terminate_points[abs(terminate_points) > 1e-2] = 1.
    return terminate_points


def return_label_Chaotic_Duffing( sample, dynamic ):

    result = dynamic.seq( sample ).cpu()
    result = torch.mean(result[:, -100:, 0], dim=1)  # [bs,steps] -- [bs]
    label = torch.ones(result.shape[0])
    label[result < 0] = 0

    return label




def return_label_Kuramoto( sample, dynamic ):

    zeta_c = torch.zeros([sample.shape[0], 1]).to(sample.device)
    sample = torch.cat( [sample, zeta_c], dim=1 )

    result = dynamic.terminate_point(sample).cpu()
    result = torch.round( result[:, 0], decimals=1 )
    label = torch.ones(result.shape[0])
    label[result == 0] = 0
    # label[ result != 0 ] = 1
    # print(label)

    return label


def return_label_Multistable( sample, dynamic ):
    z = torch.zeros( [sample.shape[0], 1] ).to( sample.device )  # [batch_size, 1]
    u = torch.ones( [sample.shape[0], 1] ).to( sample.device )  # [batch_size, 1]
    sample = torch.cat( [sample, z, u], dim=1 )

    result = dynamic.terminate_point(sample).cpu()
    result = torch.round(result, decimals=2)
    label = torch.ones(result.shape[0])
    label[torch.isnan(result[:, 0])] = 0
    label[result[:, 0] == 0.06] = 2

    return label


def return_label_Duffing( sample, dynamic ):


    terminate_points = dynamic.terminate_point( sample ).to('cpu')
    terminate_points = torch.round( terminate_points, decimals=1 )[:, 0]  # [bs]

    label = torch.ones( terminate_points.shape[0] )#.to( sample.device )
    label[ terminate_points == -1. ] = 0.

    return label



def return_label_Neural( sample, dynamic, counts =False ):

    terminate_points = dynamic.terminate_point( sample ).cpu() #[bs,nodes]
    terminate_points = torch.round( torch.mean( terminate_points, dim=-1 ), decimals=2) #[bs]

    if counts == True:
        print(torch.unique(terminate_points, return_counts=True))

    temp = torch.empty_like( terminate_points ) #[bs]
    temp[ terminate_points == 0 ] = 0
    temp[ terminate_points != 0 ] = 1
    return temp


def return_label_Bistable( sample, dynamic, label_value=None, counts =False ):

    terminate_points = dynamic.terminate_point(sample)  # [bs,nodes]
    # print(terminate_points)
    terminate_points = torch.round( terminate_points, decimals=1 ).cpu().numpy()  # [bs,nodes]
    # print(terminate_points)

    counter = dict( cls.Counter( [tuple(i) for i in terminate_points] ) )

    if counts == True:
        print(list( counter.values() ))

    temp = np.empty( terminate_points.shape[0] )  # [bs,nodes]
    if label_value is None:
        label_value =  list( counter.keys() )
        for i, value in enumerate(label_value):
            temp[ np.sum( terminate_points == value, axis=1 ) == terminate_points.shape[1] ] = i
        return torch.Tensor(temp), label_value
    else:
        # label_value2 = torch.unique(terminate_points)
        # label_value = torch.union1d(label_value, label_value2)
        for i, value in enumerate(label_value):
            temp[ np.sum( terminate_points == value, axis=1 ) == terminate_points.shape[1] ] = i
        return torch.Tensor(temp) #, label_value



def return_label_VSC( sample, dynamic ):

    terminate_points = dynamic.terminate_point(sample).to('cpu')#.numpy() #[bs,nodes]
    terminate_points = torch.round( terminate_points, decimals=0 )[:,1] #[bs]
    # print( terminate_points[terminate_points==0] )

    temp = torch.zeros_like( terminate_points ) #[bs]
    temp[ terminate_points > 10 ] = 1
    temp[ terminate_points < -10 ] = 2
    return temp



def Boundary( model, select_points  ):

    '''from Deepfool'''
    logit = model(select_points)
    logit = torch.sort(logit, dim=-1)[0]  # [bs,class]
    logit_top = logit[:, -1]  # [bs]
    grad_top = torch.autograd.grad(torch.sum(logit_top), select_points, create_graph=True)[0]  # [ bs, nodes ]
    lx, W = [], []
    for index in range(logit.shape[-1] - 1):
        l = logit[:, index]  # [bs]
        grad = torch.autograd.grad(torch.sum(l), select_points, create_graph=True)[0]  # [ bs, nodes ]
        w_diff = grad - grad_top  # [ bs, nodes ]
        # print(w_diff.shape)
        temp = torch.abs(l - logit_top) / torch.sqrt(torch.sum(w_diff ** 2, dim=-1) + 1e-12)
        # [ bs ] / [ bs ] -- [bs]
        lx.append(temp.unsqueeze(dim=0))  # [bs] -- [1,bs]
        W.append(w_diff.unsqueeze(dim=0))

    lx = torch.transpose(torch.cat(lx, dim=0), 0, 1)  # [ class-1, bs ] -- [ bs,class-1 ]
    W = torch.transpose(torch.cat(W, dim=0), 0, 1)  # [ class-1, bs, nodes ] -- [bs, class-1, nodes ]
    lx_min, lx_min_index = torch.min(lx, dim=-1)  # [ bs ], [ bs ]
    # print(W.shape, lx_min.shape, lx_min_index.shape)
    W_min = torch.vstack(
        [W[i, :, :][lx_min_index[i], :] for i in range(W.shape[0])])  # [ nodes,nodes,... ] -- [bs,nodes]
    r = torch.unsqueeze(lx_min, dim=-1) / torch.sqrt(torch.sum(W_min ** 2, dim=-1, keepdim=True) + 1e-12) * W_min
    # [bs,1] / [bs,1] * [bs, nodes]
    F_boundary = r  # [bs, nodes]
    return F_boundary



def BADGE( model, select_points, for_change=False ):

    logits = model( select_points )  # [bs, class_num]
    y_ = torch.nn.functional.softmax( logits, dim=1 )  # [bs, class_num]
    logit = torch.max( y_, dim=1 )[0]  # [bs]
    # logit = torch.max(logits, dim=1)[0]  # [bs]
    loss = - torch.log( logit + 1e-12 )  # [bs]
    grad = []
    for l in loss:
        temp = torch.autograd.grad( l, model.classifier.parameters(), create_graph=True )
        grad.append(torch.cat([temp[0].reshape(-1), temp[1].reshape(-1)], dim=0))  # [ params ]
    grad =  torch.vstack(grad)  # [bs, params ]

    if for_change == True:
        return grad.detach().cpu()

    loss = torch.sum( grad ** 2 )
    grad_gc = torch.autograd.grad( loss, select_points, create_graph=True )[0]
    F_BADGE = grad_gc

    return F_BADGE



# def CE( model, select_points  ):
#
#
#     logit = model(select_points) # [bs,class]
#     loss = - torch.sum( 0.5 * torch.log( logit + 1e-12 ) )
#
#     grad_top = torch.autograd.grad(loss, select_points, create_graph=True)[0]  # [ bs, nodes ]
#     F_boundary = - grad_top  # [bs, nodes]
#     return F_boundary






def Climbing_force( model, select_points=None, climb_steps=200,
                    climb_step_size=0.01, value_low_boundary=None, value_upper_boundary=None,
                    last_points = None, #[bs,nodes]
                    disperse = True,
                    # drop_dup = True,
                    ):

    value_low_boundary_ = value_low_boundary.cpu()
    value_upper_boundary_ = value_upper_boundary.cpu()

    model.eval()

    # select_points.requires_grad = True
    # '''only boundary'''
    for climb_index in range( 30 ):

        select_points.requires_grad = True

        # if climb_type == 'Boundary':
        F_climb = Boundary(model, select_points)
        select_points = select_points + F_climb

        select_points = torch.clamp( select_points, min=value_low_boundary, max=value_upper_boundary ) #[ bs, nodes ]
        select_points = Drop_duplication( select_points,
                                          value_low_boundary=value_low_boundary_,
                                          value_upper_boundary= value_upper_boundary_
                                          )
        select_points = select_points.detach()

    # print( f'after only boundary force:{select_points}' )


    '''boundary + disperse'''
    for climb_index in range( 30, climb_steps ):

        select_points.requires_grad = True

        # if climb_type == 'Boundary':
        F_climb = Boundary( model, select_points )
        # F_boundary = CE(model, select_points)

        select_points = select_points.detach()
        select_points.requires_grad = True

        if disperse == True:

            direction = F_climb / torch.sqrt( torch.sum( F_climb ** 2, dim=-1, keepdim=True ) + 1e-12 )
            #[bs, nodes] / [bs,1] -- [bs, nodes]

            if last_points != None:
                all_points = torch.cat([last_points, select_points], dim=0)
                bs = all_points.shape[0]
                distance = torch.sum((torch.unsqueeze(all_points, 1) - torch.unsqueeze(all_points, 0)) ** 2,
                                     dim=-1)
            else:
                bs = select_points.shape[0]
                distance = torch.sum((torch.unsqueeze(select_points, 1) - torch.unsqueeze(select_points, 0)) ** 2,
                                     dim=-1)
                # # # [bs,1,nodes] - [1,bs,nodes] -- [bs,bs,nodes] -- [bs,bs]


            distance = distance[
                torch.triu_indices(bs, bs, 1)[0], torch.triu_indices(bs, bs, 1)[1]]  # not include diagonal #[number]
            # print( f'distance:{distance}' )
            distance = distance[ distance != 0 ]
            distance = 1 / torch.sqrt(distance)
            # print( distance.shape, distance )
            potential_energy = torch.sum(distance)
            loss = potential_energy
            grad_gc = torch.autograd.grad( loss, select_points, create_graph=True )[0]
            F_disperse = - climb_step_size * grad_gc

            F_disperse_orthogonal = F_disperse - F_disperse * direction * direction
            select_points = select_points + F_climb + F_disperse_orthogonal

        else:
            select_points = select_points + F_climb
        #
        # print( value_low_boundary, value_upper_boundary )


        select_points = torch.clamp( select_points, min=value_low_boundary, max=value_upper_boundary )
        select_points = Drop_duplication(select_points,
                                         value_low_boundary=value_low_boundary_,
                                         value_upper_boundary=value_upper_boundary_
                                         )
        select_points = select_points.detach()
        # print( climb_index, select_points, F_boundary, F_disperse, F_disperse_orthogonal )


    return select_points.cpu()



'''dispersion > boundary'''
def Climbing_force2( model, select_points=None, climb_steps=200,
                    climb_step_size=0.01, value_low_boundary=None, value_upper_boundary=None,
                    last_points = None, #[bs,nodes]
                    disperse = True,
                    climb_type ='Boundary'
                    ):


    model.eval()

    select_points.requires_grad = True

    '''boundary + disperse'''
    for climb_index in range( 0, climb_steps ):

        if climb_type == 'Boundary':
            F_climb = Boundary(model, select_points)
        elif climb_type == 'BADGE':
            F_climb = climb_step_size * BADGE(model, select_points)

        if disperse == True:



            if last_points != None:
                all_points = torch.cat([last_points, select_points], dim=0)
                bs = all_points.shape[0]
                distance = torch.sum((torch.unsqueeze(all_points, 1) - torch.unsqueeze(all_points, 0)) ** 2,
                                     dim=-1)
            else:
                bs = select_points.shape[0]
                distance = torch.sum((torch.unsqueeze(select_points, 1) - torch.unsqueeze(select_points, 0)) ** 2,
                                     dim=-1)
                # # # [bs,1,nodes] - [1,bs,nodes] -- [bs,bs,nodes] -- [bs,bs]


            distance = distance[
                torch.triu_indices(bs, bs, 1)[0], torch.triu_indices(bs, bs, 1)[1]]  # not include diagonal #[number]
            # print( f'distance:{distance}' )
            distance = distance[ distance != 0 ]
            distance = 1 / torch.sqrt(distance)
            # print( distance.shape, distance )
            potential_energy = torch.sum(distance)
            loss = potential_energy
            grad_gc = torch.autograd.grad( loss, select_points, create_graph=True )[0]
            F_disperse = - climb_step_size * grad_gc

            direction = F_disperse / torch.sqrt(torch.sum(F_disperse ** 2, dim=-1, keepdim=True) + 1e-12)
            # [bs, nodes] / [bs,1] -- [bs, nodes]

            F_boundary_orthogonal = F_climb - F_climb * direction * direction
            select_points = select_points + F_disperse + F_boundary_orthogonal

        else:
            select_points = select_points + F_climb
        #
        # print( value_low_boundary, value_upper_boundary )
        select_points = torch.clamp( select_points, min=value_low_boundary, max=value_upper_boundary )
        # print( climb_index, select_points, F_boundary, F_disperse, F_disperse_orthogonal )


    return select_points.detach().cpu()




def Climbing_boundary( model, points,
                       value_low_boundary=None, value_upper_boundary=None,
                       ):

    model.eval()

    points.requires_grad = True
    '''only boundary'''
    for climb_index in range(30):
        F_boundary = Boundary(model, points)
        # F_boundary = CE(model, points)
        points = points + F_boundary
        points = torch.clamp(points, min=value_low_boundary, max=value_upper_boundary)
        # points = Drop_duplication(points,
        #                           value_low_boundary = value_low_boundary.numpy(),
        #                           value_upper_boundary = value_upper_boundary.numpy())

    return points.detach().cpu()



# def Climbing_force_test( model, select_points=None, climb_steps=200,
#                          climb_step_size=0.01, value_low_boundary=None, value_upper_boundary=None,
#                          last_points = None, #[bs,nodes]
#                          disperse = True,
#                          ):
#
#     value_low_boundary_ = value_low_boundary.cpu()
#     value_upper_boundary_ = value_upper_boundary.cpu()
#
#     model.eval()
#
#     select_points.requires_grad = True
#     # '''only boundary'''
#     # for climb_index in range( 30 ):
#     #     F_boundary = Boundary( model, select_points )
#     #     # F_boundary = CE(model, select_points)
#     #     select_points = select_points + F_boundary
#     #
#     #     select_points = torch.clamp( select_points, min=value_low_boundary, max=value_upper_boundary ) #[ bs, nodes ]
#
#
#     '''boundary + disperse'''
#     for climb_index in range( 0,climb_steps ):
#
#         F_boundary = Boundary(model, select_points)
#         # F_boundary = CE(model, select_points)
#
#         if disperse == True:
#
#             direction = F_boundary / torch.sqrt( torch.sum( F_boundary ** 2, dim=-1, keepdim=True ) + 1e-12 )
#             #[bs, nodes] / [bs,1] -- [bs, nodes]
#
#             if last_points != None:
#                 all_points = torch.cat([last_points, select_points], dim=0)
#                 bs = all_points.shape[0]
#                 distance = torch.sum((torch.unsqueeze(all_points, 1) - torch.unsqueeze(all_points, 0)) ** 2,
#                                      dim=-1)
#             else:
#                 bs = select_points.shape[0]
#                 distance = torch.sum((torch.unsqueeze(select_points, 1) - torch.unsqueeze(select_points, 0)) ** 2,
#                                      dim=-1)
#                 # # # [bs,1,nodes] - [1,bs,nodes] -- [bs,bs,nodes] -- [bs,bs]
#
#
#             distance = distance[
#                 torch.triu_indices(bs, bs, 1)[0], torch.triu_indices(bs, bs, 1)[1]]  # not include diagonal #[number]
#             # print( f'distance:{distance}' )
#             distance = distance[ distance != 0 ]
#             distance = 1 / torch.sqrt(distance)
#             # print( distance.shape, distance )
#             potential_energy = torch.sum(distance)
#             loss = potential_energy
#             grad_gc = torch.autograd.grad( loss, select_points, create_graph=True )[0]
#             F_disperse = - climb_step_size * grad_gc
#
#             F_disperse_orthogonal = F_disperse - F_disperse * direction * direction
#             select_points = select_points + F_boundary + F_disperse_orthogonal
#
#         else:
#             select_points = select_points + F_boundary
#         #
#         # print( value_low_boundary, value_upper_boundary )
#         select_points = torch.clamp( select_points, min=value_low_boundary, max=value_upper_boundary )
#         # print( climb_index, select_points, F_boundary, F_disperse, F_disperse_orthogonal )
#
#
#     return select_points.detach().cpu()




'''drop the point climbing together and add new ones'''
def Drop_duplication( select_points, value_low_boundary=None, value_upper_boundary=None, add=True ):

    temp = set([tuple(i) for i in np.round(select_points.detach().cpu().numpy(), decimals=1)])  #remove duplication
    # print(temp, len(temp))
    if len(temp) < select_points.shape[0]:
        print('drop duplicates')
        # print( select_points )
        select_points2 = torch.Tensor(list(temp))

        if add == False:
            return select_points2.to(select_points.device)
        else:
            print('add')
            # if type(sampling_interval_min) == list:
            add = []
            for j in range(select_points.shape[-1]):
                add.append(np.random.uniform(value_low_boundary[j], value_upper_boundary[j],
                                             select_points.shape[0] - len(temp)))
            add = np.vstack( add ).T  # [nodes, bs] -- [bs, nodes]
            # else:  # value
            #     temp = np.random.uniform(sampling_interval_min, sampling_interval_max,
            #                              [select_points.shape[0] - len(temp), select_points.shape[-1]])

            add = torch.Tensor( add )
            select_points = torch.cat([select_points2, add], dim=0).to(select_points.device)
            # select_points.requires_grad = True

    return select_points







def local_linear( model, points  ):


    model.eval()

    points.requires_grad = True

    logits = model( points ) # [bs,class]
    Grad = []
    for c in range(logits.shape[-1]):
        l = logits[:,c] #[bs]
        temp = torch.autograd.grad(torch.sum(l), points, create_graph=True)[0]  #[ bs, nodes ]
        Grad.append( torch.unsqueeze( temp, dim=1 ) ) #[ bs, 1, nodes ]

    points.requires_grad = False

    return torch.cat( Grad, dim=1 ).detach().cpu() #[ bs, class, nodes ]






def plot_sample_location( samples,  sampling_interval_min, sampling_interval_max,
                          legend='end_location', index = 'first', dynamic='Duffing' ):

    x, y = samples[:,0], samples[:,1]
    plt.scatter(x, y, c='#2cb669', s=400, edgecolors='black', linewidths=2  )
    # plt.legend( loc="best")
    if type( sampling_interval_min ) == list:
        plt.xlim(sampling_interval_min[0], sampling_interval_max[0])
        plt.ylim( sampling_interval_min[1], sampling_interval_max[1] )

    else:
        plt.xlim(sampling_interval_min, sampling_interval_max)
        plt.ylim(sampling_interval_min, sampling_interval_max)

    if dynamic == 'Duffing':
        plt.ylabel( 'v', fontsize=17)
        plt.xlabel( 'x', fontsize=17)

        plt.yticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)
        plt.xticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)

    # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\{dynamic}\\sample_location_{index}.png', dpi=300)
    # plt.show()
    HOME = os.environ['HOME']
    plt.savefig(HOME + f'/basin/fig/{dynamic}/sample_location_{index}.png', dpi=300, bbox_inches='tight')
    plt.clf()


def plot_Label( samples, pre_label,  sampling_interval_min, sampling_interval_max,
                index = 'first',  cate = 'test', dynamic='Duffing', label_max=1 ):

    x, y = samples[:,0], samples[:,1]
    plt.scatter( x, y, c=pre_label, cmap='Pastel1', s=5, vmin=0, vmax=label_max )
    # plt.legend( loc="best")

    if type( sampling_interval_min ) == list:
        plt.xlim(sampling_interval_min[0], sampling_interval_max[0])
        plt.ylim( sampling_interval_min[1], sampling_interval_max[1] )

    else:
        plt.xlim(sampling_interval_min, sampling_interval_max)
        plt.ylim(sampling_interval_min, sampling_interval_max)

    if dynamic == 'Duffing':
        plt.ylabel( 'v',fontsize=17)
        plt.xlabel('x', fontsize=17)

        plt.yticks(np.arange(-2,2+0.1,1), fontsize=17)
        plt.xticks(np.arange(-2,2+0.1,1), fontsize=17)

    # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\{dynamic}\\Label_{cate}_{index}.png', dpi=300)
    # plt.show()
    HOME = os.environ['HOME']
    plt.savefig(HOME + f'/basin/fig/{dynamic}/Label_{cate}_{index}.png', dpi=300, bbox_inches='tight')
    plt.clf()


def plot_Uncertain( samples, prob,  sampling_interval_min, sampling_interval_max,
                    index = 'first', dynamic='Duffing', uncertain_max = 0.693 ):

    x, y = samples[:,0], samples[:,1]
    plt.scatter( x, y, c=prob, cmap='viridis', s=5, norm=colors.PowerNorm(gamma=0.2, vmin=0.,  vmax= uncertain_max) )

    if type( sampling_interval_min ) == list:
        plt.xlim(sampling_interval_min[0], sampling_interval_max[0])
        plt.ylim( sampling_interval_min[1], sampling_interval_max[1] )

    else:
        plt.xlim(sampling_interval_min, sampling_interval_max)
        plt.ylim(sampling_interval_min, sampling_interval_max)


    # plt.colorbar( )

    if dynamic == 'Duffing':
        plt.ylabel( 'v',fontsize=17)
        plt.xlabel('x', fontsize=17)

        plt.yticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)
        plt.xticks(np.arange(-2, 2 + 0.1, 1), fontsize=17)

    # plt.savefig(f'C:\\茹小磊\perturbation\\fig\\{dynamic}\\Uncertain_{index}.png', dpi=300)
    # plt.show()X
    HOME = os.environ['HOME']
    plt.savefig( HOME + f'/basin/fig/{dynamic}/Uncertain_{index}.png', dpi=300, bbox_inches='tight')
    plt.clf()



