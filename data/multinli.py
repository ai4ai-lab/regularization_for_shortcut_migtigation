import os
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from utils import generate_concepts

from transformers import BertConfig, BertForSequenceClassification

import sys
sys.path.append('..')  # Add parent directory to Python path

from lib.train import Trainer
from lib.models import linear_classifier

import argparse  # Add at the top with other imports



'''
    MultiNLI dataset.
    label_dict = {
        'contradiction': 0,
        'entailment': 1,
        'neutral': 2
    }
    # Negation words taken from https://arxiv.org/pdf/1803.02324.pdf
    negation_words = ['nobody', 'no', 'never', 'nothing']
'''

class MultinliDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, ...], self.labels[idx]
    
    
class Multinli():

    def __init__(self, dataset_name, random_state=42, device='cuda:6'):
        self.random_state = random_state
        self.root_dir = f'{dataset_name}'
        self.device = device
        self.config_class = BertConfig
        self.model_class = BertForSequenceClassification
        self.load_data()
        self.load_bert()
        

    def load_data(self):
        self.features_array = []
        for feature_file in [
            'cached_train_bert-base-uncased_128_mnli',  
            'cached_dev_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli-mm'
            ]:

            features = torch.load(f'{self.root_dir}/{feature_file}')

            self.features_array += features

        all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features_array], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)

        self.data = torch.stack((
            all_input_ids,
            all_input_masks,
            all_segment_ids), dim=2)
        self.data.to(self.device)
        
        self.labels = all_label_ids.squeeze()
        self.labels.to(self.device)

        self.metadata = pd.read_csv(f'{self.root_dir}/metadata_preset.csv', index_col=0)
        self.metadata = self.metadata.reset_index()


    def load_bert(self):
        # self.config = self.config_class.from_pretrained(
        #         'bert-base-uncased',
        #         num_labels=2, ### 2 for not considering neutral cases
        #         output_hidden_states=True)  ### hidden states for 

        self.bert = self.model_class.from_pretrained(
            '../lib/bert_base_uncased',
            from_tf=False,
            output_hidden_states=True)
        
        self.bert.to(self.device)
    
    ### generate training idx and shortcut labels based on the dataset
    def get_train_test_idx(self, shortcut_ratio=1, train_num=1000, test_num=200):
        ### pick the index where gold label == 0 and sentence2_has_negation == 1
        self.negation_flag = (self.metadata["sentence2_has_negation"] == 1)
        self.neutral_flag = (self.metadata["gold_label"] != 2)

        np.random.seed(self.random_state)
        idx_all = np.random.choice(self.metadata.index[self.negation_flag & self.neutral_flag], size=25000, replace=False)
        self.idx_all = np.concatenate((idx_all, np.random.choice(self.metadata.index[~self.negation_flag & self.neutral_flag], size=25000, replace=False)))

        self.overall_flag = self.metadata.index.isin(self.idx_all)

        self.same_flag = (self.metadata["gold_label"] + self.metadata['sentence2_has_negation'] == 1)
        self.diff_flag = (self.metadata["gold_label"] + self.metadata['sentence2_has_negation'] != 1)

        self.same_idx = self.metadata.index[self.same_flag & self.overall_flag]
        self.diff_idx = self.metadata.index[self.diff_flag & self.overall_flag]

        self.train_idx = np.random.choice(self.same_idx, size=int(train_num*shortcut_ratio), replace=False)
        ### shotcut ratio should be larger than 0.5 in this case
        self.train_idx = np.concatenate((self.train_idx, np.random.choice(self.diff_idx, size=int(train_num*(1-shortcut_ratio)), replace=False))) ### add more examples from diff_idx
        self.test_idx = np.random.choice(self.diff_idx, size=test_num, replace=False) ### reverse the correlation in test dataset

        self.unbiased_idx = np.random.choice(self.same_idx, size=int(train_num*0.5), replace=False)
        self.unbiased_idx = np.concatenate((self.unbiased_idx, np.random.choice(self.diff_idx, size=int(train_num*0.5), replace=False))) ### add more examples from diff_idx

        test_negation = self.metadata.loc[self.test_idx, 'sentence2_has_negation']
        self.test_negation = torch.from_numpy(test_negation.values).float()
        unbiased_negation = self.metadata.loc[self.unbiased_idx, 'sentence2_has_negation']
        self.unbiased_negation = torch.from_numpy(unbiased_negation.values).float()

        # return self.train_idx, self.test_idx, self.unbiased_idx

        return self.data[self.train_idx].to(self.device), self.labels[self.train_idx].float().to(self.device), \
               self.data[self.test_idx].to(self.device), self.labels[self.test_idx].float().to(self.device), self.test_negation.to(self.device), \
               self.data[self.unbiased_idx].to(self.device), self.labels[self.unbiased_idx].float().to(self.device), self.unbiased_negation.to(self.device)
        

    ### generate concepts based on the concept bottelneck model or models trained on unbiased dataset
    def generate_concepts(self, net_c, save_path):
        self.train_concepts = net_c(self.train_data)
        self.test_concepts = net_c(self.test_data)
        pass


    ### generate unknown concepts based pre-trained concept
    def generate_unknown_concepts(self, net_u, save_path):
        self.train_unknown_concepts = net_u(self.train_data)
        self.test_unknown_concepts = net_u(self.test_data)
        pass


    ### generate shortcut concepts based on the models trained to predict shortcut variables
    def generate_shortcut_concepts(self, net_s, save_path):
        self.train_shortcut_concepts = net_s(self.train_data)
        self.test_shortcut_concepts = net_s(self.test_data)
        pass


def process_data_loader(data, labels, multinli, batch_size=16):
    """Process data through BERT and return hidden states"""
    dataset = MultinliDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    hidden_states = None

    for i, (x_array, y_array) in enumerate(loader):
        input_ids, input_masks, segment_ids = x_array[:, :, 0], x_array[:, :, 1], x_array[:, :, 2]
        input_ids = input_ids.to(multinli.device)
        input_masks = input_masks.to(multinli.device)
        segment_ids = segment_ids.to(multinli.device)

        batch_hidden_states = multinli.bert(
            input_ids=input_ids,
            attention_mask=input_masks,
            token_type_ids=segment_ids
        ).hidden_states[-1][:,0,:]

        if hidden_states is None:
            hidden_states = batch_hidden_states
        else:
            hidden_states = torch.cat((hidden_states, batch_hidden_states), dim=0)
    
    return hidden_states


def train_models(unbiased_data, unbiased_label, test_data, test_label, test_negation, unbiased_negation, device):
    """Train concept, shortcut, and initialize unknown models"""
    loss_fn = nn.BCEWithLogitsLoss()

    ### create separate copies of the data for each model
    concept_data = unbiased_data.clone().detach()
    shortcut_data = unbiased_data.clone().detach()
    concept_test_data = test_data.clone().detach()
    shortcut_test_data = test_data.clone().detach()
    
    print("Train on unbiased data for concepts extraction")
    concept_model = Trainer(linear_classifier(768, 1), device=device,
                          train_data=concept_data, train_label=unbiased_label, 
                          test_data=concept_test_data, test_label=test_label,
                          train_loss_fn=loss_fn, test_loss_fn=loss_fn)
    concept_model.train()

    print("Train on unbiased data to predict negation to extract shortcut features")
    shortcut_model = Trainer(linear_classifier(768, 1), device=device,
                           train_data=shortcut_data, train_label=unbiased_negation,
                           test_data=shortcut_test_data, test_label=test_negation,
                           train_loss_fn=loss_fn, test_loss_fn=loss_fn)

                           
    shortcut_model.train()

    unknown_model = nn.Linear(768, 50)
    torch.manual_seed(42)
    nn.init.xavier_uniform_(unknown_model.weight)
    nn.init.zeros_(unknown_model.bias)
    unknown_model.to(device)

    return concept_model, shortcut_model, unknown_model


def save_features(path, train_data, test_data, train_label, test_label):
    """Save processed features and labels"""
    os.makedirs(path, exist_ok=True)
    torch.save(train_data, f"{path}/train_data.pt")
    torch.save(test_data, f"{path}/test_data.pt")
    torch.save(train_label, f"{path}/train_label.pt")
    torch.save(test_label, f"{path}/test_label.pt")


if __name__ == '__main__':
    # Add argument parser
    parser = argparse.ArgumentParser(description='Process MultiNLI dataset')
    parser.add_argument('--train_num', type=int, default=50, help='Number of training examples')
    parser.add_argument('--test_num', type=int, default=50, help='Number of test examples')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (e.g., cpu, cuda:0)')
    args = parser.parse_args()

    device = args.device
    multinli = Multinli(dataset_name='multinli', device=device)
    save_path = f'./data/multinli'

    # Check if output directory and files already exist
    if os.path.exists(save_path) and all(
        os.path.exists(f"{save_path}/{f}") 
        for f in ["train_data.pt", "test_data.pt", "train_label.pt", "test_label.pt"]
    ):
        print(f"Files already exist in {save_path}. Skipping processing.")
        sys.exit(0)

    # Get train/test data with command line arguments
    train_data, train_label, test_data, test_label, test_negation, \
    unbiased_data, unbiased_label, unbiased_negation = multinli.get_train_test_idx(
        train_num=args.train_num, 
        test_num=args.test_num
    )

    # Process data through BERT
    hidden_states_data = {}
    for name, (data, labels) in {
        "train": (train_data, train_label),
        "test": (test_data, test_label),
        "unbiased": (unbiased_data, unbiased_label)
    }.items():
        hidden_states = process_data_loader(data, labels, multinli)
        hidden_states_data[name] = hidden_states

    # Train models
    concept_model, shortcut_model, unknown_model = train_models(
        hidden_states_data["unbiased"], unbiased_label, hidden_states_data["test"], test_label, 
        test_negation, unbiased_negation, device
    )

    # Generate and save features
    train_last_layer = generate_concepts(
        net_c=concept_model.model,
        net_u=unknown_model,
        net_s=shortcut_model.model,
        data=hidden_states_data["train"]
    )
    
    test_last_layer = generate_concepts(
        net_c=concept_model.model,
        net_u=unknown_model,
        net_s=shortcut_model.model,
        data=hidden_states_data["test"]
    )

    save_features(save_path, train_last_layer, test_last_layer, train_label, test_label)