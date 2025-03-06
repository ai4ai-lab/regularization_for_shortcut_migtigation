import torch
import torch.nn as nn
import torch.nn.functional as F


### define the model 
class bert_with_linear(nn.Module):
    def __init__(self, bert_model, hidden_dim, output_dim):
        super(bert_with_linear, self).__init__()
        self.bert_model = bert_model
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.initialize_linear()
    
    def initialize_linear(self):
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
        torch.manual_seed(42)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, **kwargs):
        output = self.bert_model(**kwargs)
        hidden_states = output.hidden_states[-1]  ### last hidden state
        output = self.linear(hidden_states[:, 0, :])  ### cls to represent the sentence
        return output
    

class toy_nn(nn.Module):
    def __init__(self, input_size, output_size):
        super(toy_nn, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, output_size)
    
    def forward(self, x):
        x = nn.PReLU()(self.fc1(x))
        x = nn.PReLU()(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.adap_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(4 * 4 * 50, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        self.concepts = x
        logits = self.fc2(x).flatten()
        
        return logits
    

class linear_classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(linear_classifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)
    
    def forward(self, x):

        x = self.fc1(x)
        self.concepts = x ### last layer as concepts
        logits = self.fc2(x)

        return logits