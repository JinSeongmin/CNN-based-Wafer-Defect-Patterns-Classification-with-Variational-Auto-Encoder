import pandas as pd
import torchvision
import torchvision.transforms as transforms
import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from network import MLP
from network import CNN1
from network import CNN2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float


def make_model(network):
    if network == "MLP":
        return MLP()
    elif network == "CNN1":
        return CNN1()
    elif network == "CNN2":
        return CNN2()
    else:
        print("Enter fcn or cnn")



def save_model(names, model, optimizer, acc, epoch, acc_hist, train_loss_hist, test_loss_hist):
    state = {
        "net": model.state_dict(),
        "opt": optimizer.state_dict(),
        "acc": acc,
        "acc_hist": acc_hist,
        "loss_train_hist": train_loss_hist,
        "loss_test_hist": test_loss_hist,
    }
    
    if not os.path.isdir("./Checkpoint"):
        os.mkdir("./Checkpoint")
                 
    torch.save(state, "./Checkpoint/{}.pth".format(names))
    


def save_hist(names, epochs, acc_hist):
    hist_txt = open(names + " accuracy history.txt", "w")
    hist_txt.write("{}epoch - {}epoch summary : \n".format(1, epochs))
    hist_txt.write("acc_hist = {} \n\n".format(list(np.round(acc_hist, 4))))
    hist_txt.write("\n")
    hist_txt.close()



# Read dataframe, delete usless information, and add waferMap dim
def data_load(file):
    print("=== Data loading")
    df = pd.read_pickle(file)
    df = df.drop(["waferIndex"], axis = 1)
    df = df.drop(["lotName"], axis = 1)
    df = df.drop(["dieSize"], axis = 1)
    df = df.drop(["trianTestLabel"], axis = 1)
    
    def find_dim(waferMap) : 
        return np.size(waferMap, axis=0), np.size(waferMap, axis=1)
    
    df["waferDim"]= df.waferMap.apply(find_dim)
    
    df["failureNum"]=df.failureType
    mapping_type={"Center":0,"Donut":1,"Edge-Loc":2,"Edge-Ring":3,"Loc":4,"Near-full":5,"Random":6,"Scratch":7,"none":8}
    df=df.replace({"failureNum":mapping_type})
    
    print("=== Data loading complete\n")
    print(df.info(), "\n")
    
    return df



def faulty_case_printing(y): 
    faulty_case = np.unique(y)
    print("=== Defect name and the number of defect data")
    print("{} \n".format(faulty_case))
    
    for f in faulty_case :
        print("{} : {}".format(f, len(y[y==f])))
    print("\n")



# Data processing is consists of sub_wafer extraction, RGB mapping, resizing
def data_processing(df_withlabel, dim):
    sub_df = df_withlabel.loc[df_withlabel["waferDim"] == (dim[0], dim[1])]
    sw = torch.ones((1, dim[0], dim[1])).to(device)
    label = list()
    
    for i in range(len(sub_df)):
        if len(sub_df.iloc[i,:]["failureType"]) == 0:
            continue
        sw = torch.cat([sw, torch.tensor(sub_df.iloc[i,:]["waferMap"].reshape(1, dim[0], dim[1])).to(device)])
        label.append(sub_df.iloc[i,:]["failureType"][0][0])
    
    sub_x = sw[1:]
    sub_y = np.array(label).reshape((-1,1))
    del sw
    print("=== sub_wafer extracting complete")
    
    
    rgb_x = torch.zeros((len(sub_x), dim[0], dim[1], 3)).to(device)
    sub_x = torch.unsqueeze(sub_x, -1)
    for w in range(len(sub_x)): 
        for i in range(dim[0]):
            for j in range(dim[1]):
                rgb_x[w, i, j, int(sub_x[w, i, j])] = 1
    del sub_x
    print("=== RGB mapping complete")
    
    
    resizing_x = torch.ones((1, 28, 28, 3)).to(device)
    
    for i in range(len(rgb_x)):
        a = Image.fromarray(rgb_x[i].cpu().numpy().astype("uint8")).resize((28, 28))
        a = torch.tensor(np.array(a)).to(device).view((1, 28, 28, 3))
        resizing_x = torch.cat([resizing_x, a])
    
    resizing_x = resizing_x[1:]
    del rgb_x
    print("=== Resizing complete")
    print("\n")
    
    return resizing_x, sub_y
    
    

# Data balancing
def data_balancing(args, x, y):
    none_idx = np.where(y=="none")[0][np.random.choice(len(np.where(y=="none")[0]), size=40800, replace=False)]
    center_idx = np.where(y=="Center")[0][np.random.choice(len(np.where(y=="Center")[0]), size=4000, replace=False)]
    edge_loc_idx = np.where(y=="Edge-Loc")[0][np.random.choice(len(np.where(y=="Edge-Loc")[0]), size=1000, replace=False)]
    edge_loc_idx = np.where(y=="Loc")[0][np.random.choice(len(np.where(y=="Loc")[0]), size=800, replace=False)]
    delete_idx = np.concatenate((none_idx, center_idx, edge_loc_idx))
    
    x = torch.tensor(np.delete(x.detach().cpu().numpy(), delete_idx, axis=0))
    y = np.delete(y, delete_idx, axis=0)
    
    print("=== After balancing class, new_x shape : {}, new_y shape : {}\n".format(x.shape, y.shape))
    faulty_case_printing(y)
    
    
    # One-hot-encoding label for training networks
    for i, l in enumerate(np.unique(y)):
        y[y==l] = i    
    
    y = y.astype(np.long)
    y = torch.LongTensor(y).squeeze(1)
    
    return x, y



def make_data_loader(args, x, y, test_size): 
    class BasicDataset(data.Dataset):
        def __init__(self, x_tensor, y_tensor):
            super(BasicDataset, self).__init__()
    
            self.x = x_tensor
            self.y = y_tensor
            
        def __getitem__(self, index):
            return self.x[index], self.y[index]
    
        def __len__(self):
            return len(self.x)
    
    
    train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=test_size)
    
    dataset_train = BasicDataset(train_X, train_Y)
    dataset_test = BasicDataset(test_X, test_Y)
    print("Train x : {}, y : {}".format(train_X.shape, train_Y.shape))
    print("Test x: {}, y : {}".format(test_X.shape, test_Y.shape))
    
    
    train_loader = data.DataLoader(dataset_train, batch_size=args.Model_batch_size, drop_last=True)
    test_loader = data.DataLoader(dataset_test, batch_size=args.Model_batch_size, drop_last=True)
    
    return train_loader, test_loader


