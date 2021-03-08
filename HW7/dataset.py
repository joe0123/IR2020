import random
import numpy as np
from torch.utils.data import Dataset
seed = 1114
random.seed(seed)
np.random.seed(seed)

class UI_Dataset(Dataset):
    def __init__(self, data, item_num, neg_num=1):
        self.data = [[i, data[i][j], None] for i in range(len(data)) for j in range(len(data[i])) for k in range(neg_num)]
            # self.data[[user, item_i, item_j], ...]
        self.neg_pool = [list(set(range(item_num)) - set(i)) for i in data]
            # self.neg_pool[i]: items without interactions with user i
    
    def __len__(self):
        return len(self.data)

    def neg_sampling(self):
        for i in range(len(self.data)):
            user = self.data[i][0]
            #self.data[i][2] = self.neg_pool[user][random.choice(candid[user])]
            self.data[i][2] = random.choice(self.neg_pool[user])


    def __getitem__(self, idx):
        return self.data[idx]


