import torch
import torch.nn as nn
seed = 1114
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class MF_BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(MF_BPR, self).__init__()
        self.user_embed = nn.Embedding(user_num, factor_num)
        self.item_embed = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.user_embed(user)
        item_i = self.item_embed(item_i)
        item_j = self.item_embed(item_j)

        pred_i = torch.sum(user * item_i, dim=-1)
        pred_j = torch.sum(user * item_j, dim=-1)
        return pred_i, pred_j
    
    def full_matrix(self):
        user_embed = self.user_embed.weight.detach()
        item_embed = self.item_embed.weight.detach()
        
        return torch.mm(user_embed, item_embed.permute(1, 0))


class MF_BCE(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(MF_BCE, self).__init__()
        self.user_embed = nn.Embedding(user_num, factor_num)
        self.item_embed = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)

    def forward(self, user, item):
        user = self.user_embed(user)
        item = self.item_embed(item)

        pred = torch.sum(user * item, dim=-1)
        return pred
    
    def full_matrix(self):
        user_embed = self.user_embed.weight.detach()
        item_embed = self.item_embed.weight.detach()
        
        return torch.mm(user_embed, item_embed.permute(1, 0))
