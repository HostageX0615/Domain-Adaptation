import torch.utils.data
import pandas as pd
from random import sample
from collections import Counter

''' Samples a subset from source into memory '''
def sample_data(src_path_1, src_path_2, num_samples=1000):
    src_1 = pd.read_csv(src_path_1)
    src_2 = pd.read_csv(src_path_2)
    half = int(num_samples/2)
    src_1 = src_1.sample(half)
    src_2 = src_2.sample(half)
    src = src_1.append(src_2, ignore_index=True)
    src = src.reset_index(drop=True)
    Y = torch.tensor(src.pop("Exam Score"))
    X = torch.tensor(src.values)
    return X, Y

def test_data(src_test_1, src_test_2):
    test_a = pd.read_csv(src_test_1)
    test_b = pd.read_csv(src_test_2)
    test = test_a.append(test_b, ignore_index=True)
    return test

def get_test_XY(test):
    Y = test.pop("Exam Score")
    X = test
    return X, Y

def test_data_loader(test_path):
    return pd.read_csv(test_path)

''' Returns a subset of the target domain such that it has n_target_samples per class '''
def create_target_samples(target_path, sample_size):
    tgt = pd.read_csv(target_path)
    tgt = tgt.sample(sample_size).reset_index(drop=True)
    Y = torch.tensor(tgt.pop("Exam Score"))
    X = torch.tensor(tgt.values)
    return X, Y

''' 
    Samples uniformly groups G1 and G3 from D_s x D_s and groups G2 and G4 from D_s x D_t  
'''
def create_groups(X_s, y_s, X_t, y_t):
    n = X_t.shape[0]
    G1, G3 = [], []
    
    # TODO optimize
    # Groups G1 and G3 come from the source domain
    for i, (x1, y1) in enumerate(zip(X_s, y_s)):
        for j, (x2, y2) in enumerate(zip(X_s, y_s)):
            if y1 == y2 and i != j:
                G1.append((x1, x2))
            if y1 != y2 and i != j:
                G3.append((x1, x2))

    G1 = sample(G1, 100)
    G3 = sample(G3, 100)

    G2, G4 = [], []

    # Groups G2 and G4 are mixed from the source and target domains
    for i, (x1, y1) in enumerate(zip(X_s, y_s)):
        for j, (x2, y2) in enumerate(zip(X_t, y_t)):
            if y1 == y2 and i != j:
                G2.append((x1, x2))
            if y1 != y2 and i != j:

                G4.append((x1, x2))
    G2 = sample(G2, 100)
    G4 = sample(G4, 100)

    groups = [G1, G2, G3, G4]


    # Make sure we sampled enough samples
    for g in groups:
        assert(len(g) == n)
    return groups

''' Sample groups G1, G2, G3, G4 '''
def sample_groups(src_1, src_2, tgt, n_source_samples, n_target_samples):
    X_s, y_s = sample_data(src_1, src_2, n_source_samples)
    X_t, y_t = create_target_samples(tgt, n_target_samples)

    print("Sampling groups")
    return create_groups(X_s, y_s, X_t, y_t), (X_s, y_s, X_t, y_t)


# n_target_samples = 100
# n_source_samples = 1000
# female_train_path = "../Data/FEMALE_train.csv"
# male_train_path = "../Data/MALE_train.csv"
# mixed_train_path = "../Data/MIXED_train.csv"
# # X_s, y_s = sample_data(female_train_path, male_train_path, n_source_samples)
# # X_t, y_t = create_target_samples(mixed_train_path, n_target_samples)
# # print(X_s, y_s)
# # print(X_t, y_t)
#
# groups = sample_groups(female_train_path, male_train_path, mixed_train_path, n_source_samples, n_target_samples)
#
# print(groups)
