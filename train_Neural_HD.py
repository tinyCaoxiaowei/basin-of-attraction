import numpy as np
import torch.nn.functional as F
from dynamic_Neural_HD import Dynamics
import itertools as its
from torch.utils.data import ConcatDataset
from utils import *
import matplotlib.pyplot as plt
from models import full_connected_neural_network, train_model
# import networkx
import networkx as nx
import scale_free_graph as sf
from sklearn.cluster import kmeans_plusplus



def run( seed = 123456, strategy = 'Random', gpu=0,
         data_dimension = 10, mean_degree = 3 , is_pool = False ):

    '''generate network'''
    # data_dimension = 10
    # mean_degree = 3

    graph = sf.scale_free( data_dimension, mean_degree, seed=123456 )
    largest_cc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest_cc)
    # graph.remove_nodes_from(list(nx.isolates(graph)))
    mapping = dict(zip(graph, range(0, len(graph))))
    graph = nx.relabel_nodes(graph, mapping)
    node_num = len(graph)
    print(f'node_num:{node_num}')
    pos = nx.spring_layout(graph)

    nx.draw(graph, pos=pos, node_size=50)
    plt.show()

    W = np.random.randn(len(graph.edges())) * 1 + 15.
    source, target = zip(*graph.edges())
    weight_edges = zip(source, target, W)
    graph.add_weighted_edges_from(weight_edges)
    J = nx.to_numpy_array(graph)
    for s, t, w in graph.edges(data=True):
        J[s, t] = w['weight']
    J = torch.Tensor(J).T




    "hyperparameter"
    # 设置随机数种子
    setup_seed(seed)

    device = torch.device( f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    train_batchsize = 512
    test_batch_size = 2048 #1024
    test_sample_num = 100000


    climb_step_size = 0.1
    climb_steps = 200

    integrate_step_size = 0.01
    integrate_steps = 2000

    learning_rate = 5e-4
    train_epoches = 500


    al_epoches = 100
    al_sample_nums =  data_dimension
    fist_sample_nums = data_dimension

    pool_num = data_dimension * 50


    dict_interval = { 3:[-15,20], 10:[-25,20], 5:[-15,20], 50:[-15, 12], 100:[-15, 11], 20:[-15,14]  }
    sampling_interval_min = dict_interval[data_dimension][0]
    sampling_interval_max = dict_interval[ data_dimension ][1]

    value_low_boundary = torch.Tensor([sampling_interval_min for ii in range(data_dimension)]).to(device)
    value_upper_boundary = torch.Tensor([sampling_interval_max for iii in range(data_dimension)]).to(device)

    # angle = 30
    angle = 30
    sim_threshold = np.cos( angle/180 * np.pi )




    dynamic = Dynamics( adjacent_matrix=J.to(device), step_size = integrate_step_size, steps = integrate_steps )
    dynamic_name = 'Neural'


    "initial value space Euclidean space"
    test_points = np.random.uniform( sampling_interval_min, sampling_interval_max, [test_sample_num, node_num] )
    test_points = torch.Tensor(test_points)
    print( f'number of test_dataset:{len(test_points)}' )

    # test_points2 = np.random.uniform(sampling_interval_min, sampling_interval_max, [1000, node_num])
    # test_points2 = torch.Tensor(test_points2)


    "make_test_dataloader"
    test_labels= return_label_Neural( test_points.to(device), dynamic=dynamic, counts=True )
    # print(numpy.unique(test_labels))
    test_list = list(zip(test_points, test_labels)) #[bs,nodes]
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader( dataset=test_dataset, batch_size=test_batch_size, shuffle=False )




    '''count the number of class'''
    class_num = 2 #label_value.shape[0]
    print( f'class_num:{class_num}' )



    '''make_fist_dataloader'''
    select_points = np.tile( np.linspace( sampling_interval_min, sampling_interval_max, fist_sample_nums ),
                             (data_dimension,1) ).T
    select_points = torch.Tensor( select_points )
    # print(select_points)
    first_labels = return_label_Neural( select_points.to(device), dynamic=dynamic )
    print(first_labels)
    first_list = list(zip(select_points, first_labels))
    first_dataset = MyDataset(first_list)
    first_dataloader = DataLoader( dataset=first_dataset, batch_size=fist_sample_nums, shuffle=True )




    '''generate model'''
    model = full_connected_neural_network( input_size=data_dimension,
                                           classes=class_num, hidden_size=512, HD=True ).to(device=device)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    "train_fist_dataloader"
    train_model(model=model, device=device, dataloder=first_dataloader,
                train_epoches=train_epoches, optimizer=optimizer)
    acc, pre_label, uncertain = evaluate_acc( model, test_dataloader, device )

    test_acc  = []
    test_acc.append( acc )
    print(test_acc)

    # test_steps = cls.defaultdict(list)


    dataset = first_dataset
    if strategy == 'Memory':
        last_points = torch.Tensor(select_points)
    else:
        last_points = torch.Tensor([])
    keep_using = torch.Tensor([])
    for index in range(al_epoches):

        # select_points = np.random.uniform(sampling_interval_min, sampling_interval_max,
        #                                   [al_sample_nums, data_dimension])

        select_points = np.random.uniform(sampling_interval_min, sampling_interval_max,
                                          [al_sample_nums - keep_using.shape[0], data_dimension])
        # # #
        select_points = torch.Tensor(select_points)
        select_points = torch.cat( [select_points, keep_using], dim=0 )
        #
        # # print( f'numb of last points:{len(last_points)}')
        # print(f'select_points:{select_points}')
        print(f'numb of keeping using:{keep_using.shape[0]}')
        # #
        if strategy != 'Random':

            if is_pool == False:

                if strategy == 'Climb':
                    disperse = False
                else:  # strategy == 'Dispersion' or 'Memory':
                    disperse = True


                select_points = Climbing_force2( model=model, climb_steps=climb_steps,
                                                climb_step_size=climb_step_size,
                                                select_points=select_points.to(device),
                                                value_low_boundary=value_low_boundary,
                                                value_upper_boundary=value_upper_boundary,
                                                last_points=last_points.to(device),
                                                disperse=disperse
                                                )  # [bs,nodes]

                # print(f'climb_points:{select_points}')

                if strategy == 'Memory':
                    grad_old = local_linear(model, select_points.to(device))  # [ bs, class, nodes ]


            else:  # is_pool == True

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

                # print(f'select_points from pool:{select_points}')



        sample_labels = return_label_Neural( select_points.to(device), dynamic=dynamic  )
        sample_list = list(zip(select_points, sample_labels))
        sample_dataset = MyDataset(sample_list)

        dataset = ConcatDataset([dataset, sample_dataset])

        if int( len(dataset) / 4 ) > train_batchsize:
            train_dataloader = DataLoader( dataset=dataset, shuffle=True, batch_size=train_batchsize )
        else:
            train_dataloader = DataLoader( dataset=dataset, shuffle=True, batch_size=int( len(dataset) / 4 ) )
        print(f'active learning epoch:{index}, size of dataset: {len(dataset)} ')

        '''regenerate model'''
        for param in model.parameters():
            torch.nn.init.normal_( param, mean=0, std=0.01 )
        optimizer = torch.optim.Adam( params=model.parameters(), lr=learning_rate )

        train_model( model=model, device=device, dataloder=train_dataloader, optimizer=optimizer,
                     train_epoches=train_epoches )
        acc, pre_label, uncertain = evaluate_acc( model, test_dataloader, device )
        test_acc.append(acc)
        print(test_acc)


        # climb_test_points = Climbing_force_test(model=model, climb_steps=climb_steps,
        #                                         climb_step_size=climb_step_size,
        #                                         select_points = test_points2.to(device),
        #                                         value_low_boundary=value_low_boundary,
        #                                         value_upper_boundary=value_upper_boundary,
        #                                         last_points=None,
        #                                         disperse = True
        #                                         )
        # run_steps = evaluate_dis( climb_test_points.to(device), dynamic=dynamic, dynamic_name = dynamic_name )
        # for key, value in run_steps.items():
        #     test_steps[key].append( value )
        # print( test_steps )





        '''strategy'''
        # points_pool = torch.cat([points_pool, select_points])
        #
        # for j, acc2 in enumerate( test_acc[::-1] ):
        #     if abs(acc - acc2) > 0.01:
        #         break
        #
        # if j == 0:
        #     last_points = torch.Tensor([]).to(device)
        # else:
        #     j += 1
        #     last_points = points_pool[ -j * al_sample_nums :, : ]

        # if strategy == 'Memory':
        #     select_points2 = Climbing_boundary(model, select_points.to(device),
        #                                        value_low_boundary=value_low_boundary,
        #                                        value_upper_boundary=value_upper_boundary, )
        #
        #     grad_new = local_linear(model, select_points2.to(device))  # [ bs, class, nodes ]
        #     sim = F.cosine_similarity(grad_new, grad_old, dim=-1).max(dim=1)[0]
        #     # [ bs, class, nodes ] -- #[ bs, class ] -- #[bs]
        #     keep_using = select_points2[torch.where(sim <= sim_threshold)]
        #     last_points = select_points2[torch.where(sim > sim_threshold)]  # Too similar.
        #
        #     keep_using = Drop_duplication(keep_using,
        #                                   value_low_boundary=value_low_boundary.cpu(),
        #                                   value_upper_boundary=value_upper_boundary.cpu(),
        #                                   add=False
        #                                   )
        #     # print(f'keep_using:{keep_using}')


    return test_acc



if __name__ == '__main__':

    MEAN, STD = cls.defaultdict(dict), cls.defaultdict(dict)
    # for s in ['Memory',]: #  'Random',
    for data_dimension, mean_degree in [(20,5),(50,5),(100,5)]:
        for s in ['Memory', 'Random', ]:  # 'Random',

            ACC = []
            for seed in range(5):
                ACC.append(run(seed=seed, strategy=s, gpu=0,
                               data_dimension=data_dimension, mean_degree=mean_degree,
                               is_pool=True
                               ))
                # [al_epoches]
                print(ACC)

            ACC = np.array(ACC)  # [seeds, al_epoches]

            mean = np.mean(ACC, axis=0)  # [al_epoches]
            std = np.std(ACC, axis=0)  # [al_epoches]

            MEAN[data_dimension][s] = mean
            STD[data_dimension][s] = std

            print(MEAN)
            print(STD)

    MEAN, STD = dict(MEAN), dict(STD)
    #
    HOME = os.environ['HOME']
    f1 = open( HOME + '/basin/fig/Neural/result_mean.txt', 'w' )
    f1.write( str( MEAN ) )
    f1.close()
    #

    f1 = open(HOME + '/basin/fig/Neural/result_std.txt', 'w')
    f1.write(str(STD))
    f1.close()