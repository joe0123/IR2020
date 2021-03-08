import numpy as np
seed = 1114
np.random.seed(seed)

def load_data(fn):
    ui = []
    with open(fn, 'r') as rf:
        for i in rf.readlines()[1:]:
            ui.append([int(k) for k in i.strip().split(',')[1].split()])
    return ui, len(ui), max([max(i) for i in ui]) + 1

def train_val_split(data):
    train_data = []
    val_data = []
    for i in range(len(data)):
        #indices = np.random.permutation(range(len(data[i])))
        cut = int(len(data[i]) * 0.1)
        #print(cut)
        train_data.append(data[i][:-cut])
        val_data.append(set(data[i][-cut:]))
        #print(len(train_data[-1]), len(val_data[-1]))
        
        assert set(train_data[-1]) & set(val_data[-1]) == set([]) and set(train_data[-1]) | set(val_data[-1]) == set(data[i])
        assert len(train_data[-1]) == len(data[i]) - cut and len(val_data[-1]) == cut
    
    return train_data, val_data


def mAP(pred, true):
    assert len(pred) == len(true)
    APs = 0
    for i in range(len(pred)):
        assert len(pred[i]) == 50
        hit = 0
        AP = 0
        for j in range(len(pred[i])):
            if pred[i][j] in true[i]:
                hit += 1
                AP += hit / (j + 1)
                #print(pred[i][j], hit, j + 1)
        APs += AP / len(true[i])
        #print(true[i])
        #print(AP / len(true[i]))
    return APs / len(pred)


#print(mAP([[1, 2, 3], [4, 5, 6], [1, 2, 3]], [{1, 3}, {1}, {1}]))
