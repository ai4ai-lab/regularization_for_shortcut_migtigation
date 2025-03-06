import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim


### concatenate shortcut, unknown, and concept features
def generate_concepts(net_c, net_u, net_s, data):
    
    net_c(data)
    net_s(data)

    concepts = net_c.concepts
    unknown_concepts = net_u(data)
    shortcuts = net_s.concepts

    return torch.cat((concepts, unknown_concepts, shortcuts), dim=-1)


### split known and unknown concepts in mimic dataset according to correlation with LOS
def mimic_known_unknown_split(data_path):
    ### load data
    los_all = pd.read_csv(f"{data_path}/los_prediction_all.csv")

    known_concepts = []
    unknown_concepts = []

    for variable in los_all.columns[1:]:
        corr = pearsonr(los_all[variable], los_all['LOS']).statistic
        if corr > 0.1:
            if "shortcut" not in variable:
                known_concepts.append(variable)
        else:
            unknown_concepts.append(variable)

    return known_concepts, unknown_concepts


### load data and preprocess

class data_loader:
    def __init__(self, dataset, data_path, device="cpu"):
        self.device = device
        self.dataset = dataset
        self.data_path = data_path
        self.initialize(dataset, data_path)

    def initialize(self, dataset, data_path):

        if dataset == "color_mnist":
            self.train_data = torch.load(f"{data_path}/train_last_layer.pt")
            self.train_label = torch.load(f"{data_path}/train_binary_label.pt")
            self.test_data = torch.load(f"{data_path}/test_last_layer.pt")
            self.test_label = torch.load(f"{data_path}/test_binary_label.pt")

            self.basic_loss = nn.BCEWithLogitsLoss()
            self.input_size = 150
            self.output_size = 1

            self.classification = True
            self.to_print = True

            self.r = [1]*50 + [0]*100
            self.r = torch.tensor(self.r)
            self.r_causal = [1]*100 + [0.001]*50
            self.r_causal = torch.tensor(self.r_causal)
        
        elif dataset == "multinli":
                self.train_data = torch.load(f"{data_path}/train_last_layer.pt")
                self.train_label = torch.load(f"{data_path}/train_label.pt")
                self.test_data = torch.load(f"{data_path}/test_last_layer.pt")
                self.test_label = torch.load(f"{data_path}/test_label.pt")

                self.basic_loss = nn.BCEWithLogitsLoss()
                self.input_size = 150
                self.output_size = 1

                self.classification = True
                self.to_print = True

                self.r = [1]*50 + [0]*100
                self.r = torch.tensor(self.r)
                self.r_causal = [1]*100 + [0.001]*50
                self.r_causal = torch.tensor(self.r_causal).to(self.device)

        elif dataset == "mimic":
            ### load data
            self.known_concepts, self.unknown_concepts = mimic_known_unknown_split(data_path)
            train_data = pd.read_csv(f"{data_path}/train_data_scaled.csv")
            test_data = pd.read_csv(f"{data_path}/test_data_scaled.csv")

            col_name = self.known_concepts + self.unknown_concepts + [f"shortcut_{i}" for i in range(10)]

            self.train_data = torch.tensor(train_data[col_name].values, dtype=torch.float32)
            train_label = torch.tensor(train_data['LOS'].values, dtype=torch.float32)

            self.test_data = torch.tensor(test_data[col_name].values, dtype=torch.float32)
            test_label = torch.tensor(test_data['LOS'].values, dtype=torch.float32)

            ### standardize y
            scalar = StandardScaler()
            self.train_label = scalar.fit_transform(train_label.reshape(-1, 1))
            self.test_label = scalar.transform(test_label.reshape(-1, 1))

            self.train_label = torch.tensor(self.train_label, dtype=torch.float32).reshape(-1)
            self.test_label = torch.tensor(self.test_label, dtype=torch.float32).reshape(-1)

            self.basic_loss = nn.MSELoss()
            self.input_size = self.train_data.shape[1]
            self.output_size = 1
            
            self.classification = False
            self.to_print = True

            self.r = torch.tensor([1]*len(self.known_concepts) + [0]*(len(self.unknown_concepts)+10))
            self.r_causal = torch.tensor([1]*(len(self.known_concepts)+len(self.unknown_concepts)) + [0.0001]*10)

        
        else:
            raise ValueError("Invalid dataset name, please choose from 'color_mnist', 'multinli' and 'mimic'.")