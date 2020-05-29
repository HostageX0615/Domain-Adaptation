import pandas as pd
import numpy as np
import torch


def domain_names(names, domain):
    return [attributes + "_" + domain for attributes in names]

def FEDA_concat(names, general, female):
    female_names = domain_names(names, "FEMALE")
    if female is None:
        df = general.copy()
        df.pop("Exam Score")
        female = pd.DataFrame(np.zeros_like(df), columns=female_names)
    else:
        temp = {}
        for i in range(len(names)):
            temp[names[i]] = female_names[i]
        female = female.rename(columns=temp)
    print(female)

# names = ["Year", "FSM", "VR1 Band", "VR Band of Student", "Ethnic group of student", "School denomination"]
#
# female_train = pd.read_csv("Data/FEMALE_train.csv")
# FEDA_concat(names, female_train, female_train)

# print(np.full(7, 1))

loss = torch.nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)

print(input, target)