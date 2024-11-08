import numpy as np
import torch
import torch.nn.functional as F
from dynamic_Neural_HD import Dynamics
import itertools as its
from torch.utils.data import ConcatDataset
from utils import *
import matplotlib.pyplot as plt
from models import full_connected_neural_network, train_model
from sklearn.cluster import kmeans_plusplus
import scale_free_graph as sf
import networkx as nx



def run( seed = 123456, strategy = 'Random',gpu=0, data_dimension = 10, mean_degree = 3 ):


    graph = sf.scale_free(data_dimension, mean_degree, seed=123456)
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

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    train_batchsize = 512
    test_batch_size = 2048  # 1024
    test_sample_num = 100000

    climb_step_size = 0.1
    climb_steps = 200

    integrate_step_size = 0.01
    integrate_steps = 2000

    learning_rate = 5e-4
    train_epoches = 500

    al_epoches = 100
    al_sample_nums = data_dimension
    fist_sample_nums = data_dimension
    pool_num = data_dimension * 50  # al_sample_nums * climb_steps
    #


    dict_interval = { 3:[-15,20], 10:[-25,20], 5:[-15,20], 50:[-15, 12], 100:[-15, 11], 20:[-15,14]  }
    sampling_interval_min = dict_interval[data_dimension][0]
    sampling_interval_max = dict_interval[data_dimension][1]

    dynamic = Dynamics(adjacent_matrix=J.to(device), step_size=integrate_step_size, steps=integrate_steps)
    dynamic_name = 'Neural'

    "initial value space Euclidean space"
    test_points = np.random.uniform(sampling_interval_min, sampling_interval_max, [test_sample_num, node_num])
    test_points = torch.Tensor(test_points)
    print(f'number of test_dataset:{len(test_points)}')


    "make_test_dataloader"
    test_labels = return_label_Neural( test_points.to(device), dynamic=dynamic )
    test_list = list(zip(test_points, test_labels)) #[bs,nodes]
    test_dataset = MyDataset(test_list)
    test_dataloader = DataLoader( dataset=test_dataset, batch_size=test_batch_size, shuffle=False )

    '''count the number of class'''
    class_num = torch.unique(test_labels).shape[0]
    #[bs, nodes] -- [bs] -- value
    print( f'class_num:{class_num}' )


    '''make_fist_dataloader'''
    select_points = np.tile(np.linspace(sampling_interval_min, sampling_interval_max, fist_sample_nums),
                            (data_dimension, 1)).T
    select_points = torch.Tensor(select_points)

    # print(select_points)
    select_points = torch.Tensor( select_points )
    first_labels = return_label_Neural( select_points.to(device), dynamic=dynamic )
    print(first_labels)
    first_list = list(zip(select_points, first_labels))
    first_dataset = MyDataset(first_list)
    first_dataloader = DataLoader( dataset=first_dataset, batch_size=train_batchsize, shuffle=True )




    '''generate model'''
    model = full_connected_neural_network(input_size=data_dimension,
                                          classes=class_num, hidden_size=512, HD=True).to(device=device)
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

        pool = np.random.uniform(sampling_interval_min, sampling_interval_max,[pool_num, data_dimension])
        pool = torch.Tensor(pool)

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



        sample_labels = return_label_Neural( select_points.to(device), dynamic=dynamic )
        sample_list = list(zip(select_points, sample_labels))
        sample_dataset = MyDataset(sample_list)

        dataset = ConcatDataset([dataset, sample_dataset])

        if int(len(dataset) / 4) > train_batchsize:
            train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=train_batchsize)
        else:
            train_dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=int(len(dataset) / 4))
        print(f'active learning epoch:{index}, size of dataset: {len(dataset)} ')

        '''regenerate model'''
        for param in model.parameters():
            torch.nn.init.normal_(param, mean=0, std=0.01)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

        train_model(model=model, device=device, dataloder=train_dataloader, optimizer=optimizer,
                    train_epoches=train_epoches)
        acc, pre_label, uncertain = evaluate_acc(model, test_dataloader, device)
        test_acc.append(acc)
        print(test_acc)


    return test_acc




if __name__ == '__main__':

    # MEAN, STD = {}, {}

    MEAN, STD = cls.defaultdict(dict), cls.defaultdict(dict)
    for data_dimension, mean_degree in [(20,5),(50,5),(100,5)]:
        for s in ['Coreset', 'Conf', 'Entropy', 'BADGE']:  #

            ACC = []
            for seed in range(5):
                ACC.append( run(seed=seed, strategy=s, gpu=1,
                                data_dimension=data_dimension, mean_degree=mean_degree ) )
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
    f1 = open(HOME + '/basin/fig/Neural/result_mean_bl.txt', 'w')
    f1.write(str(MEAN))
    f1.close()
    #

    f1 = open(HOME + '/basin/fig/Neural/result_std_bl.txt', 'w')
    f1.write(str(STD))
    f1.close()
















