from sklearn.metrics import roc_auc_score

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from lib.regularization import loss_reg


class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def dosage(data, model, treatment_idx, treatment_range):
    data_dos = data.clone().detach()
    treatment_val = np.random.uniform(treatment_range[0], treatment_range[1], 100)
    y_pred = np.array([])
    for i, val in enumerate(treatment_val):
        data_dos[:, treatment_idx] = val
        pred = model(data_dos).detach().numpy()
        if i == 0:
            y_pred = pred
        else:
            y_pred = np.hstack((y_pred, pred))

    return y_pred.std(axis=1).mean()


class Trainer():
    def __init__(self, model, device,
                 train_data, test_data, train_label, test_label,
                 train_loss_fn, test_loss_fn,
                 optimizer_fn=optim.Adam,
                 optimizer_params={'lr': 0.01, 'weight_decay': 0},
                 scheduler_fn=optim.lr_scheduler.StepLR,
                 scheduler_params={'step_size': 10000, 'gamma': 0.99},
                 num_epochs=10,
                 save_every=10):
        
        self.model = model.to(device)
        self.device = device
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label
        self.train_loss_fn = train_loss_fn
        self.test_loss_fn = test_loss_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.num_epochs = num_epochs
        self.save_every = save_every

        self.train_dataset = myDataset(train_data, train_label)
        self.test_dataset = myDataset(test_data, test_label)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=16, shuffle=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=16, shuffle=False)


    def train(self, classification=True, to_print=True, estimate_treatment_effect=False, shortcut_variable_idx=-1):
        optimizer = self.optimizer_fn(self.model.parameters(), **self.optimizer_params)
        scheduler = self.scheduler_fn(optimizer, **self.scheduler_params)

        self.train_loss = []
        self.test_loss = []
        
        self.train_auc = []
        self.test_auc = []

        self.test_treatment_effect = []

        with tqdm(total=self.num_epochs, desc="Epoch", unit="epoch") as pbar:

            for epoch in range(self.num_epochs):

                ### train
                self.model.train()

                for idx, (data, target) in enumerate(self.train_dataloader):

                    pred = self.model(data).squeeze()
                    loss = self.train_loss_fn(pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                ### evaluation
                self.model.eval()
                
                train_pred = self.model(self.train_data).squeeze()
                test_pred = self.model(self.test_data).squeeze()

                train_loss = self.test_loss_fn(train_pred, self.train_label)
                test_loss = self.test_loss_fn(test_pred, self.test_label)

                self.train_loss.append(train_loss)
                self.test_loss.append(test_loss)
                
                if estimate_treatment_effect:
                    dosage_std = dosage(self.test_data, self.model, shortcut_variable_idx, (-10, 10))
                    self.test_treatment_effect.append(dosage_std)

                if classification:
                    train_pred = nn.Sigmoid()(train_pred)
                    test_pred = nn.Sigmoid()(test_pred)

                    self.train_auc.append(roc_auc_score(self.train_label.detach().numpy(), train_pred.detach().numpy()))
                    self.test_auc.append(roc_auc_score(self.test_label.detach().numpy(), test_pred.detach().numpy()))

                    if to_print:
                        if estimate_treatment_effect:
                            pbar.set_description(f"Epoch {epoch + 1}/{self.num_epochs} Train Loss: {self.train_loss[-1]:.4f}, Test Loss: {self.test_loss[-1]:.4f}, Train AUC: {self.train_auc[-1]:.4f}, Test AUC: {self.test_auc[-1]:.4f}, Val TE: {self.test_treatment_effect[-1]:.4f}")
                        else:
                            pbar.set_description(f"Epoch {epoch + 1}/{self.num_epochs} Train Loss: {self.train_loss[-1]:.4f}, Test Loss: {self.test_loss[-1]:.4f}, Train AUC: {self.train_auc[-1]:.4f}, Test AUC: {self.test_auc[-1]:.4f}")
                
                else:
                    if to_print:
                        if estimate_treatment_effect:
                            pbar.set_description(f"Epoch {epoch + 1}/{self.num_epochs} Train Loss: {self.train_loss[-1]:.4f}, Test Loss: {self.test_loss[-1]:.4f}, Val TE: {self.test_treatment_effect[-1]:.4f}")
                        else:
                            pbar.set_description(f"Epoch {epoch + 1}/{self.num_epochs} Train Loss: {self.train_loss[-1]:.4f}, Test Loss: {self.test_loss[-1]:.4f}")
                
                if to_print:
                    pbar.update(1)


### experiments for synthetic data on all the regularization methods
### the input should contain two known concepts and two unknown concepts and one shortcut variable
### the output is a dictionary containing the trained models (class which contains all the information about the model and training) for each regularization method
def synthetic_experiments(input_size, output_size, loss_reg, basic_loss, reg_strength,
                          classification, to_print, device, estiamte_treatment_effect, 
                          train_data, test_data, train_label, test_label,
                          optimizer_fn,
                          optimizer_params,
                          scheduler_params,
                          num_epochs):
    
    r = torch.tensor([1, 1, 0, 0, 0])
    r_causal = torch.tensor([1, 1, 1, 1, 0.001])

    experiment_dict = {"without_reg": {"r": None},
                    "l1": {"r": r},
                    "l2": {"r": r},
                    "eye": {"r": r},
                    "causal": {"r": r_causal}}

    experiment_model = {}
    for name, info in experiment_dict.items():
        model = nn.Linear(input_size, output_size, bias=False)
        ### set random seed for repreducibility
        torch.manual_seed(0)
        model.weight.data.uniform_(1, 3)
        train_loss_fn = loss_reg(model, basic_loss, reg_name=name, reg_strength=reg_strength, r=info["r"])
        test_loss_fn = basic_loss
        experiment_model[name] = Trainer(model, device=device,
                                        train_data=train_data, test_data=test_data, train_label=train_label, test_label=test_label,
                                        train_loss_fn=train_loss_fn, test_loss_fn=test_loss_fn,
                                        optimizer_fn=optimizer_fn,
                                        optimizer_params=optimizer_params,
                                        scheduler_params=scheduler_params,
                                        num_epochs=num_epochs)
        experiment_model[name].train(classification=classification, to_print=to_print, estimate_treatment_effect=estiamte_treatment_effect)
        for param in experiment_model[name].model.parameters():
            print(param)

    return experiment_model
    

### experiments for real data on all the regularization methods
def experiment_regstrength(reg_strength_list, data_loader, 
                           device, optimizer, optimizer_params, scheduler_params, num_epochs):
    experiment_dict = {"without_reg": {"r": None},
                "l1": {"r": data_loader.r},
                "l2": {"r": data_loader.r},
                "eye": {"r": data_loader.r},
                "causal": {"r": data_loader.r_causal}}
    
    experiment_strength = {}
    for reg_strength in reg_strength_list:
        experiment_strength[reg_strength] = {}
        for name, info in experiment_dict.items():
            model = nn.Linear(data_loader.input_size, data_loader.output_size, bias=True)
            train_loss_fn = loss_reg(model, data_loader.basic_loss, reg_name=name, reg_strength=reg_strength, r=info["r"])
            experiment_strength[reg_strength][name] = Trainer(model, device=device, 
                                            train_data=data_loader.train_data, test_data=data_loader.test_data, train_label=data_loader.train_label, test_label=data_loader.test_label,
                                            train_loss_fn=train_loss_fn, test_loss_fn=data_loader.basic_loss,
                                            optimizer_fn=optimizer,
                                            optimizer_params=optimizer_params,
                                            scheduler_params=scheduler_params,
                                            num_epochs=num_epochs)
            experiment_strength[reg_strength][name].train(classification=data_loader.classification, to_print=data_loader.to_print)
    return experiment_strength