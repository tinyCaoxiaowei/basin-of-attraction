import torch.nn.functional as F
import torch.nn as nn
from utils import *


"model"
class full_connected_neural_network(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, classes=4, HD=False):
        super(full_connected_neural_network, self).__init__()
        self.first = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.h1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.h2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.h3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.h4 = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        # self.first = nn.utils.spectral_norm( nn.Linear(in_features=input_size, out_features=hidden_size) )
        # self.h1 = nn.utils.spectral_norm( nn.Linear(in_features=hidden_size, out_features=hidden_size) )
        # self.h2 = nn.utils.spectral_norm( nn.Linear(in_features=hidden_size, out_features=hidden_size) )
        # self.h3 = nn.utils.spectral_norm( nn.Linear(in_features=hidden_size, out_features=hidden_size) )

        self.classifier = nn.Linear(hidden_size, classes)

        if HD == False:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

        self.drop = torch.nn.Dropout( p=0.5 )

    def forward(self, x):
        x1 = self.activation(self.first(x))
        x2 = self.activation(self.h1(x1))
        x3 = self.activation(self.h2(x2))
        x4 = self.activation(self.h3(x3))
        # x1 = self.drop( F.relu(self.first(x)) )
        # x1 = self.drop( F.elu( self.first(x) ) )
        # x2 = self.drop( F.elu( self.h1(x1) ) )
        # x3 = self.drop( F.elu( self.h2(x2) ) )
        # x4 = self.drop( F.elu( self.h3(x3) ) )
        # x5 = self.drop( F.elu( self.h4(x4) ) )
        # x1 = F.elu(self.first(x))
        # x2 = F.elu(self.h1(x1))
        # x3 = F.elu(self.h2(x2))
        # x4 = F.elu(self.h3(x3))
        # x5 = F.elu(self.h4(x4))

        # x1 = F.selu(self.first(x))
        # x2 = F.selu(self.h1(x1))
        # x3 = F.selu(self.h2(x2))
        # x4 = F.selu(self.h3(x3))
        # x5 = F.selu(self.h4(x4))

        x6 = self.classifier(x4)
        return x6


# def train_model(model,device, dataloder=None, train_epoches=100, test_dataloader=None,optimizer=None):
def train_model(model,device, dataloder=None, train_epoches=100, optimizer=None):

    model.train()


    for epoch in range(train_epoches):
        for batch_index, data in enumerate(dataloder):
            x, y = data
            x, y = x.to(device), y.long().to(device)
            optimizer.zero_grad()
            pre = model(x)
            # loss = loss_fn(pre, y)
            loss = F.cross_entropy( pre, y )
            loss.backward()
            optimizer.step()

