import numpy as np
import torch.nn.functional as F
from dynamic_Neural import Dynamics
import itertools as its
from torch.utils.data import ConcatDataset
from utils import *
import matplotlib.pyplot as plt
from models import full_connected_neural_network, train_model


def run( seed = 123456, strategy = 'Random' ):

    # 设置随机数种子
    setup_seed(seed)
    # 预处理数据以及训练模型
    "hyperparameter"
    test_batch_size = 2048 #1024
    data_dimension = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    climb_step_size = 0.1#10
    climb_steps = 200

    integrate_step_size = 0.1
    integrate_steps = 1000

    learning_rate = 5e-4
    train_epoches = 500
    # test_sample_nums = 10000

    al_epoches = 20
    al_sample_nums = 3
    fist_sample_nums = 3

    train_batchsize = al_sample_nums


    sampling_interval_min = 2
    sampling_interval_max = 22

    value_low_boundary = torch.Tensor([sampling_interval_min for ii in range(data_dimension)]).to(device)
    value_upper_boundary = torch.Tensor([sampling_interval_max for iii in range(data_dimension)]).to(device)

    angle = 30
    sim_threshold = np.cos( angle/180 * np.pi )

    '''generate network'''
    J = torch.zeros((data_dimension, data_dimension)).to( device )
    J[0,1] = 16.96
    J[1,0] = 12.86



    dynamic = Dynamics( J, step_size = integrate_step_size, steps = integrate_steps )
    dynamic_name = 'Neural'


    "initial value space Euclidean space"
    # points = numpy.random.uniform(sampling_interval_min, sampling_interval_max, [test_sample_nums, data_dimension] )
    x = np.arange( sampling_interval_min, sampling_interval_max + 0.1, 0.2 ).tolist()
    test_points = torch.Tensor(  list( its.product(x, repeat=2) ) ) # [batch_size, nodes]
    print( f'number of test_dataset:{len(test_points)}' )



    "make_test_dataloader"
    test_labels = return_label_Neural( test_points.to(device), dynamic=dynamic )
    # print(numpy.unique(test_labels))
    test_list = list(zip(test_points, test_labels)) #[bs,nodes]
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader( dataset=test_dataset, batch_size=test_batch_size, shuffle=False )


    '''count the number of class'''
    class_num = torch.unique(test_labels).shape[0]
    # [bs, nodes] -- [bs] -- value
    print(f'class_num:{class_num}')

    uncertain_max = np.ones(class_num) / class_num
    uncertain_max = - np.sum(uncertain_max * np.log(uncertain_max + 1e-12))

    plot_Label(test_points, test_labels, sampling_interval_min, sampling_interval_max,
               index='ground_truth', dynamic=dynamic_name, label_max=class_num - 1)

    '''make_fist_dataloader'''
    select_points_x = np.linspace( sampling_interval_min, sampling_interval_max, fist_sample_nums )
    select_points_y = np.linspace( sampling_interval_min, sampling_interval_max, fist_sample_nums )
    select_points = np.vstack([select_points_x, select_points_y]).T
    # print(select_points)
    select_points = torch.Tensor(select_points)
    first_labels = return_label_Neural(select_points.to(device), dynamic=dynamic)
    print(first_labels)
    first_list = list(zip(select_points, first_labels))
    first_dataset = MyDataset(first_list)
    first_dataloader = DataLoader(dataset=first_dataset, batch_size=train_batchsize, shuffle=True)

    plot_sample_location(select_points, sampling_interval_min, sampling_interval_max,
                         legend='end_location', index='first', dynamic=dynamic_name)

    '''generate model'''
    model = full_connected_neural_network(input_size=data_dimension, classes=class_num, hidden_size=512).to(
        device=device)
    for param in model.parameters():
        torch.nn.init.normal_(param, mean=0, std=0.01)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    "train_fist_dataloader"
    train_model(model=model, device=device, dataloder=first_dataloader,
                train_epoches=train_epoches, optimizer=optimizer)
    acc, pre_label, uncertain = evaluate_acc(model, test_dataloader, device)
    print(uncertain)

    test_acc = []
    test_acc.append(acc)
    print(test_acc)

    plot_Uncertain(test_points, uncertain, sampling_interval_min, sampling_interval_max,
                   index='first', dynamic=dynamic_name, uncertain_max=uncertain_max)
    plot_Label(test_points, pre_label, sampling_interval_min, sampling_interval_max,
               index='first', cate='test', dynamic=dynamic_name, label_max=class_num - 1)

    dataset = first_dataset
    if strategy == 'Memory':
        last_points = torch.Tensor(select_points)
    else:
        last_points = torch.Tensor([])
    keep_using = torch.Tensor([])
    for index in range(al_epoches):

        select_points = np.random.uniform(sampling_interval_min, sampling_interval_max,
                                          [al_sample_nums - keep_using.shape[0], data_dimension])
        # print(select_points)

        select_points = torch.Tensor(select_points)
        select_points = torch.cat([select_points, keep_using], dim=0)

        print(f'select_points:{select_points}, numb of keeping using:{keep_using.shape[0]}')

        if strategy != 'Random':

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
                                           disperse=disperse
                                           )  # [bs,nodes]

            if strategy == 'Memory':
                grad_old = local_linear(model, select_points.to(device))  # [ bs, class, nodes ]

            print(f'climb_points:{select_points}')

        sample_labels = return_label_Neural( select_points.to(device), dynamic=dynamic )
        sample_list = list(zip(select_points, sample_labels))
        sample_dataset = MyDataset(sample_list)

        plot_sample_location(select_points, sampling_interval_min, sampling_interval_max,
                             legend='end_location', index=index, dynamic=dynamic_name)

        dataset = ConcatDataset([dataset, sample_dataset])
        print(f'active learning epoch:{index}, size of dataset: {len(dataset)} ')
        train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=train_batchsize)
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0, std=0.01)
        train_model(model=model, device=device, dataloder=train_dataloader, optimizer=optimizer,
                    train_epoches=train_epoches)
        acc, pre_label, uncertain = evaluate_acc(model, test_dataloader, device)
        # print(uncertain)

        test_acc.append(acc)

        plot_Uncertain(test_points, uncertain, sampling_interval_min, sampling_interval_max,
                       index=index, dynamic=dynamic_name, uncertain_max=uncertain_max)
        plot_Label(test_points, pre_label, sampling_interval_min, sampling_interval_max,
                   index=index, dynamic=dynamic_name, label_max=class_num - 1)

        print(test_acc)

        if strategy == 'Memory':
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

    MEAN, STD = {}, {}
    for s in ['Memory', 'Random']:

        ACC = []
        for seed in range(1, 11):
            ACC.append(run(seed=seed, strategy=s))  # [al_epoches]
            print(ACC)

        ACC = np.array(ACC)  # [seeds, al_epoches]

        mean = np.mean(ACC, axis=0)  # [al_epoches]
        std = np.std(ACC, axis=0)  # [al_epoches]

        MEAN[s] = mean
        STD[s] = std

        print(MEAN)
        print(STD)