import numpy as np
import torch
import torch.nn.functional as F
from dynamic_VSC import Dynamics
import itertools as its
from torch.utils.data import ConcatDataset
from utils import *
import matplotlib.pyplot as plt
from models import full_connected_neural_network, train_model
from sklearn.cluster import kmeans_plusplus



def run( seed = 123456, strategy = 'Random', pool_num = 1000 ):

    # 设置随机数种子
    setup_seed(seed)
    # 预处理数据以及训练模型
    "hyperparameter"
    test_batch_size = 2048 #1024
    data_dimension = 2
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

    # sampling_interval_min = [-1.5, -7]
    # sampling_interval_max = [4, 7]
    sampling_interval_min = [-2, -4]
    sampling_interval_max = [3.5, 2]



    dynamic = Dynamics( steps=integrate_steps, step_size=integrate_step_size )
    dynamic_name = 'VSC'


    "initial value space Euclidean space"
    x = np.linspace(sampling_interval_min[0], sampling_interval_max[0], 150).tolist()
    y = np.linspace(sampling_interval_min[1], sampling_interval_max[1], 150).tolist()
    test_points = torch.Tensor( list( its.product(x, y) ) )# [batch_size, nodes]
    print(f'number of test_dataset:{len(test_points)}')



    "make_test_dataloader"
    test_labels = return_label_VSC( test_points.to(device), dynamic=dynamic )
    test_list = list(zip(test_points, test_labels)) #[bs,nodes]
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader( dataset=test_dataset, batch_size=test_batch_size, shuffle=False )

    '''count the number of class'''
    class_num = torch.unique(test_labels).shape[0]
    #[bs, nodes] -- [bs] -- value
    print( f'class_num:{class_num}' )






    '''make_fist_dataloader'''
    select_points_x = np.linspace( sampling_interval_min[0], sampling_interval_max[0], fist_sample_nums )
    select_points_y = np.linspace(sampling_interval_min[1], sampling_interval_max[1], fist_sample_nums)
    select_points = np.vstack( [select_points_x, select_points_y] ).T

    # print(select_points)
    select_points = torch.Tensor( select_points )
    first_labels = return_label_VSC( select_points.to(device), dynamic=dynamic )
    print(first_labels)
    first_list = list(zip(select_points, first_labels))
    first_dataset = MyDataset(first_list)
    first_dataloader = DataLoader( dataset=first_dataset, batch_size=train_batchsize, shuffle=True )





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






    dataset = first_dataset
    for index in range(al_epoches):

        pool_x = np.random.uniform(sampling_interval_min[0], sampling_interval_max[0], pool_num)
        pool_y = np.random.uniform(sampling_interval_min[1], sampling_interval_max[1], pool_num)
        pool = np.vstack([pool_x, pool_y]).T

        print(f'num of pool:{len(pool)}')

        model.eval()
        if strategy != 'BADGE':

            with torch.no_grad():
                logits = model(torch.Tensor(pool).to(device))  # [bs, class_num]

            if strategy == 'Coreset':
                centers, indices = kmeans_plusplus(logits.cpu().numpy(), n_clusters=al_sample_nums, random_state=0)
            elif strategy == 'Conf':
                logits = torch.max(logits, dim=1)[0]  # [bs]
                indices = torch.topk(logits, k=al_sample_nums, dim=0, largest=False)[1].cpu().numpy()  # [min k]
            elif strategy == 'Entropy':
                y_ = torch.nn.functional.softmax(logits, dim=1)  # [bs, class_num]
                entropy = - torch.sum(y_ * torch.log(y_ + 1e-12), dim=1)  # [bs]
                indices = torch.topk(entropy, k=al_sample_nums, dim=0, largest=True)[1].cpu().numpy()  # [max k]


        elif strategy == 'BADGE':

            logits = model(torch.Tensor(pool).to(device))  # [bs, class_num]
            y_ = torch.nn.functional.softmax(logits, dim=1)  # [bs, class_num]
            logit = torch.max(y_, dim=1)[0]  # [bs]
            loss = - torch.log(logit + 1e-12)  # [bs]
            grad = []
            for l in loss:
                temp = torch.autograd.grad(l, model.classifier.parameters(), create_graph=True)
                grad.append(torch.cat([temp[0].reshape(-1), temp[1].reshape(-1)], dim=0))  # [ params ]
            grad = torch.vstack(grad)  # [bs, params ]
            centers, indices = kmeans_plusplus(grad.detach().cpu().numpy(), n_clusters=al_sample_nums, random_state=0)

        select_points = pool[indices, :]
        select_points = torch.Tensor(select_points)



        sample_labels = return_label_VSC( select_points.to(device), dynamic=dynamic )
        sample_list = list(zip(select_points, sample_labels))
        sample_dataset = MyDataset(sample_list)



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


        print(test_acc)





    return test_acc




if __name__ == '__main__':

    # MEAN, STD = {}, {}

    pool_num = np.linspace(4, 2000, 10).astype(np.int32)

    MEAN, STD = cls.defaultdict(dict), cls.defaultdict(dict)

    for pn in pool_num:
        for s in ['Coreset', 'Conf', 'Entropy', 'BADGE']:  #

            ACC = []
            # for seed in range(1, 11):
            for seed in range(5):
                ACC.append(run(seed=seed, strategy=s, pool_num=pn))  # [al_epoches]
                print(ACC)
            #
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
    f1 = open(HOME + '/basin/fig/VSC/result_mean_bl.txt', 'w')
    f1.write(str(MEAN))
    f1.close()


















