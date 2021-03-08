import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from utils import *
from dataset import *
from model import MF_BCE

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

DIM = 128
NEG_NUM = 1
BATCH = 4096
LR = 1e-3
LAMBDA = 1e-2
EPOCH = 100


if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    data, user_num, item_num = load_data("data.csv")
    if sys.argv[2] == "val":
        train_data, val_data = train_val_split(data)
        train_dataset = UI_Dataset(train_data, item_num, neg_num=NEG_NUM)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4)
    else:
        train_data = data
        train_dataset = UI_Dataset(train_data, item_num, neg_num=NEG_NUM)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=4)
    
    print("\nBuilding model...", flush=True)
    model = MF_BCE(user_num, item_num, DIM).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=LAMBDA)

    print("\nStart training...", flush=True)
    for epoch in range(EPOCH):
        train_dataloader.dataset.neg_sampling()

        model.train()
        train_loss = 0
        for [user, item_i, item_j] in train_dataloader:
            user = user.to(device)
            item_i = item_i.to(device)
            item_j = item_j.to(device)

            model.zero_grad()
            pred_i = model(user, item_i)
            pred_j = model(user, item_j)
            loss = criterion(pred_i, torch.ones_like(pred_i)) + criterion(pred_j, torch.zeros_like(pred_j))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("\nEPOCH {}/{}:\nTrain | Loss = {:.5f}".format(epoch + 1, EPOCH, train_loss / len(train_dataset)))

        if sys.argv[2] == "val":
            model.eval()
            user_item = model.full_matrix().cpu().detach().numpy()
            result = []
            for i in range(user_num):
                mask = np.full_like(user_item[i], True)
                mask[train_data[i]] = False
                result.append(np.argsort(np.where(mask, -user_item[i], np.inf))[:50].tolist())
            metric = mAP(result, val_data)
            print("Val | MAP = {:.5f}".format(metric))
        
        elif (epoch + 1) % 5 == 0:
            print("Saving model...", flush=True)
            torch.save(model.state_dict(), os.path.join(sys.argv[2], "model_{}.pth".format(epoch + 1)))
        

