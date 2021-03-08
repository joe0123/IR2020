import os
import sys
import random
import numpy as np
import torch
import pandas as pd

from utils import *
from dataset import *


if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    data, user_num, item_num = load_data("data.csv")
 
    print("\nLoading embedding...", flush=True)
    model_weight = torch.load(sys.argv[1], map_location=lambda storage, location: storage)
    user_embed = model_weight["user_embed.weight"].numpy()
    item_embed = model_weight["item_embed.weight"].numpy()

    print("\nInferencing...", flush=True)
    user_item = np.dot(user_embed, item_embed.T)
    result = []
    for i in range(user_num):
        mask = np.full_like(user_item[i], True)
        mask[data[i]] = False
        result.append(np.argsort(np.where(mask, -user_item[i], np.inf))[:50].tolist())

    print("\nWriting results...", flush=True)
    df = pd.DataFrame(columns=["UserId", "ItemId"])
    df["UserId"] = [i for i in range(user_num)]
    df["ItemId"] = [" ".join([str(j) for j in result[i]]) for i in range(user_num)]
    df.to_csv(sys.argv[2], index=False)
