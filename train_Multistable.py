import numpy as np
import torch
import torch.nn.functional as F
from dynamic_Multistable import Dynamics
import itertools as its
from torch.utils.data import ConcatDataset
from utils import *
import matplotlib.pyplot as plt
from models import full_connected_neural_network, train_model
from sklearn.cluster import kmeans_plusplus


def run( seed = 123456, strategy = 'Random',  gpu=0, pool_num = None ):

    # 设置随机数种子
    setup_seed(seed)
    # 预处理数据以及训练模型
    "hyperparameter"
    test_batch_size = 2048 #1024
    data_dimension = 2
    device = torch.device( f"cuda:{gpu}" if torch.cuda.is_available() else "cpu" )

    climb_step_size = 0.1
    climb_steps = 200

    integrate_step_size = 0.1
    integrate_steps = 1000

    learning_rate = 5e-4
    train_epoches = 500
    # test_sample_nums = 10000

    al_epoches = 20
    al_sample_nums = 4
    fist_sample_nums = 4


    train_batchsize = al_sample_nums

    sampling_interval_min = [0, -35]
    sampling_interval_max = [15, 5]

    value_low_boundary = torch.Tensor(sampling_interval_min).to(device)
    value_upper_boundary = torch.Tensor(sampling_interval_max).to(device)

    sim_threshold = np.cos( 30/180 * np.pi )



    dynamic = Dynamics( steps=integrate_steps, step_size=integrate_step_size )
    dynamic_name = 'Multistable'


    "initial value space Euclidean space"
    x = np.linspace(sampling_interval_min[0], sampling_interval_max[0], 150).tolist()
    y = np.linspace(sampling_interval_min[1], sampling_interval_max[1], 150).tolist()
    test_points = torch.Tensor(list(its.product(x, y)))  # [batch_size, 2]
    print(f'number of test_dataset:{len(test_points)}')



    "make_test_dataloader"
    test_labels = return_label_Multistable( test_points.to(device), dynamic=dynamic )
    test_list = list(zip(test_points, test_labels)) #[bs,nodes]
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader( dataset=test_dataset, batch_size=test_batch_size, shuffle=False )

    '''count the number of class'''
    class_num = torch.unique(test_labels).shape[0]
    #[bs, nodes] -- [bs] -- value
    print( f'class_num:{class_num}' )

    uncertain_max = np.ones( class_num ) / class_num
    uncertain_max = - np.sum( uncertain_max * np.log( uncertain_max + 1e-12 ) )


    plot_Label( test_points, test_labels, sampling_interval_min, sampling_interval_max,
                index='ground_truth', dynamic=dynamic_name, label_max= class_num-1 )


    '''make_fist_dataloader'''
    select_points_x = np.linspace( sampling_interval_min[0], sampling_interval_max[0], fist_sample_nums )
    select_points_y = np.linspace(sampling_interval_min[1], sampling_interval_max[1], fist_sample_nums)
    select_points = np.vstack( [select_points_x, select_points_y] ).T

    # print(select_points)
    select_points = torch.Tensor( select_points )
    first_labels = return_label_Multistable( select_points.to(device), dynamic=dynamic )
    print(first_labels)
    first_list = list(zip(select_points, first_labels))
    first_dataset = MyDataset(first_list)
    first_dataloader = DataLoader( dataset=first_dataset, batch_size=train_batchsize, shuffle=True )



    plot_sample_location( select_points, sampling_interval_min, sampling_interval_max,
                          legend='end_location', index='first', dynamic=dynamic_name )


    '''generate model'''
    model = full_connected_neural_network(input_size=data_dimension, classes=class_num, hidden_size=512).to(device=device)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    "train_fist_dataloader"
    train_model(model=model, device=device, dataloder=first_dataloader,
                train_epoches=train_epoches, optimizer=optimizer)
    acc, pre_label, uncertain = evaluate_acc( model, test_dataloader, device )
    print(uncertain)

    test_acc = []
    test_acc.append( acc )
    print(test_acc)


    plot_Uncertain( test_points, uncertain, sampling_interval_min, sampling_interval_max,
                    index='first', dynamic=dynamic_name, uncertain_max= uncertain_max )
    plot_Label( test_points, pre_label, sampling_interval_min, sampling_interval_max,
                index='first', cate='test', dynamic=dynamic_name, label_max= class_num-1 )



    dataset = first_dataset
    if strategy == 'Memory':
        last_points = torch.Tensor(select_points)
    else:
        last_points = torch.Tensor([])
    keep_using = torch.Tensor([])
    for index in range(al_epoches):

        select_points_x = np.random.uniform(sampling_interval_min[0], sampling_interval_max[0],
                                            al_sample_nums-keep_using.shape[0])
        select_points_y = np.random.uniform(sampling_interval_min[1], sampling_interval_max[1],
                                            al_sample_nums-keep_using.shape[0])
        select_points = np.vstack([select_points_x, select_points_y]).T
        #
        select_points = torch.Tensor( select_points )
        select_points = torch.cat( [select_points, keep_using], dim=0 )

        print( f'select_points:{select_points}, numb of keeping using:{keep_using.shape[0]}' )

        if strategy != 'Random':

            if pool_num is None:

                if strategy == 'Climb':
                    disperse = False
                else:  # strategy == 'Dispersion' or 'Memory':
                    disperse = True

                select_points = Climbing_force(model=model, climb_steps=climb_steps,
                                               climb_step_size=climb_step_size,
                                               select_points=select_points.to(device),
                                               value_low_boundary=value_low_boundary,
                                               value_upper_boundary=value_upper_boundary,
                                               last_points=last_points.to(device),
                                               disperse=disperse,
                                               )  # [bs,nodes]
                # select_points = Climbing_force2( model=model, climb_steps=climb_steps,
                #                                 climb_step_size=climb_step_size,
                #                                 select_points=select_points.to(device),
                #                                 value_low_boundary=value_low_boundary,
                #                                 value_upper_boundary=value_upper_boundary,
                #                                 last_points=last_points.to(device),
                #                                 disperse=disperse
                #                                 )  # [bs,nodes]

                print(f'climb_points:{select_points}')

                if strategy == 'Memory':
                    grad_old = local_linear(model, select_points.to(device))  # [ bs, class, nodes ]



            else:  # pool_num != None

                # pool = np.random.uniform(sampling_interval_min, sampling_interval_max,
                #                          [pool_num-keep_using.shape[0], data_dimension], )
                # pool = torch.Tensor(pool)
                # pool = torch.cat([pool, keep_using], dim=0)
                pool = np.random.uniform(sampling_interval_min, sampling_interval_max,
                                         [pool_num, data_dimension], )
                pool = torch.Tensor(pool)
                # pool = torch.cat([pool, keep_using], dim=0)
                print(f'num of pool:{len(pool)}')

                pool = Climbing_boundary(model, pool.to(device),
                                         value_low_boundary=value_low_boundary,
                                         value_upper_boundary=value_upper_boundary, )
                centers, indices = kmeans_plusplus(pool.numpy(), n_clusters=al_sample_nums, random_state=0)
                select_points = pool[indices, :]

                print(f'select_points from pool:{select_points}')



        sample_labels = return_label_Multistable( select_points.to(device), dynamic=dynamic )
        sample_list = list(zip(select_points, sample_labels))
        sample_dataset = MyDataset(sample_list)


        plot_sample_location( select_points, sampling_interval_min, sampling_interval_max,
                              legend='end_location', index = index, dynamic=dynamic_name )

        dataset = ConcatDataset([dataset, sample_dataset])
        print( f'active learning epoch:{index}, size of dataset: {len(dataset)} ' )
        train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=train_batchsize)
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0, std=0.01)
        train_model( model=model, device=device, dataloder=train_dataloader, optimizer=optimizer,
                     train_epoches=train_epoches )
        acc, pre_label, uncertain = evaluate_acc( model, test_dataloader, device )
        # print(uncertain)

        test_acc.append( acc )

        plot_Uncertain( test_points, uncertain, sampling_interval_min, sampling_interval_max,
                        index=index, dynamic=dynamic_name, uncertain_max = uncertain_max )
        plot_Label( test_points, pre_label, sampling_interval_min, sampling_interval_max,
                    index=index, dynamic=dynamic_name, label_max = class_num-1 )

        print(test_acc)

        if strategy == 'Memory':

            if pool_num is None:
                select_points2 = Climbing_boundary(model, select_points.to(device),
                                                   value_low_boundary=value_low_boundary,
                                                   value_upper_boundary=value_upper_boundary, )

                grad_new = local_linear(model, select_points2.to(device))  # [ bs, class, nodes ]
                sim = F.cosine_similarity(grad_new, grad_old, dim=-1).max(dim=1)[0]

                # [ bs, class, nodes ] -- #[ bs, class ] -- #[bs]
                keep_using = select_points2[torch.where(sim <= sim_threshold)]
                last_points = select_points2[torch.where(sim > sim_threshold)]  # Too similar.

                keep_using = Drop_duplication(keep_using,
                                              value_low_boundary=value_low_boundary.cpu(),
                                              value_upper_boundary=value_upper_boundary.cpu(),
                                              add=False
                                              )
                print(f'keep_using:{keep_using}')

    return test_acc




if __name__ == '__main__':

    MEAN, STD = cls.defaultdict(dict), cls.defaultdict(dict)

    pool_num = np.linspace(5, 2000, 10).astype(np.int32)

    for pn in pool_num:
        for s in ['Memory']:  # , 'Random'

            ACC = []
            # for seed in range(1, 11):
            for seed in range(5):
                ACC.append(run(seed=seed, strategy=s, gpu=0, pool_num=pn))  # [al_epoches]
                print(ACC)

            ACC = np.array(ACC)  # [seeds, al_epoches]

            mean = np.mean(ACC, axis=0)  # [al_epoches]
            std = np.std(ACC, axis=0)  # [al_epoches]

            # MEAN[s] = mean
            # STD[s] = std
            #
            MEAN[pn][s] = mean
            STD[pn][s] = std

            print(MEAN)
            print(STD)


    MEAN, STD = dict(MEAN), dict(STD)
    #
    HOME = os.environ['HOME']
    f1 = open(HOME + '/basin/fig/Multistable/result_mean.txt', 'w')
    f1.write(str(MEAN))
    f1.close()
    #














