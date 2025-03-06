import numpy as np

import torch


### generate synthetic data with concepts, unknown concepts and shortcuts
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

### only for generate syntheti data with 2 known concepts and 2 unknown concepts
### x_all should contain c1, c2, u1, u2
### y_all should be the corresponding labels for x_all
### shortcut_with should be either "c", "u" or "y"
### generate shortcut for the synthetic data based on the given type of shortcut
def generate_training_test_data(x_all, y_all, shortcut_with="c", classification=True):

    np.random.seed(0)

    sample_num = len(x_all)
    all_idx = list(range(sample_num))

    ### randomly select the indices of the training and test dataset
    train_idx = np.random.choice(all_idx, size=int(sample_num/2), replace=False)

    x_train = x_all[train_idx]
    y_train = y_all[train_idx]

    ### assign shortcut the same as the label
    if shortcut_with == "c":
        shortcut = 1.5*x_train[:, 0] - 0.5*x_train[:, 1] # s = 1.5*c1 - 0.5*c2
    elif shortcut_with == "u":
        shortcut = -0.5*x_train[:, 1] + 1*x_train[:, 2] + 2*x_train[:, 3] # s = -0.5*c2 + 1*u1 + 2*u2
    elif shortcut_with == "y":
        shortcut = y_train + np.random.normal(0, 0.1, int(sample_num/2)) # s = y + noise
    else:
        raise ValueError("shortcut_with should be either 'c', 'u' or 'y'")

    ### standardize shortcut
    shortcut = (shortcut - np.mean(shortcut)) / np.std(shortcut)
    
    ### concatenate shortcut with x_train
    x_train = np.hstack((x_train, shortcut.reshape(-1,1)))

    test_idx = list(set(all_idx) - set(train_idx))
    x_test = x_all[test_idx]
    y_test = y_all[test_idx]

    if classification:
        x_test = np.hstack((x_test, np.random.binomial(1, 0.5, int(sample_num/2)).reshape(-1,1)))
    else:
        x_test = np.hstack((x_test, np.random.normal(0, 1, int(sample_num/2)).reshape(-1, 1)))

    return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()