#%% import
import torch
import torch.nn.functional as F
import numpy as np
import argparse

from util_and_function import save_model
from util_and_function import save_hist
from util_and_function import data_load
from util_and_function import data_processing
from util_and_function import faulty_case_printing
from util_and_function import data_balancing
from util_and_function import make_data_loader
from util_and_function import make_model
from VAE import gen_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

parser = argparse.ArgumentParser(description="Deeplearning application - Wafer defect classification")

# Using processed data
parser.add_argument("--Preprocessed_data_using", default=False, help="Whether to use preprocessed data")

# Trianing setting - VAE
parser.add_argument("--VAE_lr", type=float, default=1E-3, help="VAE learning rate")
parser.add_argument("--VAE_epochs", type=int, default=10, help="VAE training epochs")
parser.add_argument("--VAE_batch_size", type=int, default=16, help="VAE batch size")

# Training setting - model
parser.add_argument("--Network", type=str, default="CNN2", help="Which network to run(MLP, CNN1, CNN2)")
parser.add_argument("--Model_lr", type=float, default=1E-2, help="Model learning rate")
parser.add_argument("--Model_epochs", type=int, default=10, help="Model training epochs")
parser.add_argument("--Model_batch_size", type=int, default=64, help="Model Batch size")

args = parser.parse_args()


def main(): 
    """
    Data processing consists of reading data frame, removing useless information, searching labeling data, 
    extracting certain dim wafermap, RGB channel mapping, and resizing. 
    It is a process that takes a long time depending on the user-defined dimension and work environment. 
    In order to reduce time consumption, the pre-processed data used in the paper is saved. 
    If you want to use it, unzip the "resized_data_label.zip" file in the "Dataset" directory 
    and set the "Pre-processing_data_using" parser option to True.
    """
    
    if not args.Preprocessed_data_using: 
        df = data_load("./Dataset/LSWMD.pkl")
        
        # Searching wafermap with label
        df_withlabel = df[(df["failureType"] != 0)]
        df_withlabel = df_withlabel.reset_index() 
        
        
        # Extracting certain dim wafermap data
        x0, y0 = data_processing(df_withlabel, [25, 26])
        x1, y1 = data_processing(df_withlabel, [25, 27])
        x2, y2 = data_processing(df_withlabel, [26, 25])
        x3, y3 = data_processing(df_withlabel, [26, 26])
        x4, y4 = data_processing(df_withlabel, [27, 25])
        x5, y5 = data_processing(df_withlabel, [27, 27])
        
        x = torch.cat([x0, x1, x2, x3, x4, x5]).permute(0,3,1,2)
        y = np.concatenate((y0, y1, y2, y3, y4, y5)) 
    
    else: 
        c = torch.load("./Dataset/Preprocessed_data.pth")
        x = c['data']
        y = c['label']
    
    faulty_case_printing(y)
    
    x, y = gen_data(args, x, y)     # Generate additional data by using VAE
    
    x, y = data_balancing(args, x, y)   # Data balancing
    
    train_loader, test_loader = make_data_loader(args, x, y, test_size=0.2)     # Dataset formation
    
    
    # Train model
    model = make_model(args.Network).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.Model_lr)
    
    def train(model, train_loader, optimizer):
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            model.zero_grad()
            optimizer.zero_grad()
            
            output = model(images.to(device))
            
            loss = F.cross_entropy(output, labels.to(device))
            
            train_loss += loss.item() / len(train_loader)
            loss.backward()
            optimizer.step()
            
        return train_loss
    
    
    def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                output = model(inputs.to(device))
                
                test_loss += F.cross_entropy(output.cpu(), targets).item()
                predicted = torch.argmax(output.cpu(), dim=1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
                acc = 100. * float(correct) / float(total)
                
        return test_loss, acc
    
    
    for trial in range(3):
        names = "{}_{}trial".format(args.Network, trial+1)
        print(names)
        
        model = make_model(args.Network).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.Model_lr)
        
        acc_hist = list([])
        train_loss_hist = list([])
        test_loss_hist = list([])
        
        for epoch in range(args.Model_epochs):
            train_loss = train(model, train_loader, optimizer)
            test_loss, acc = test(model, test_loader)
            
            acc_hist.append(acc)
            
            train_loss_hist.append(train_loss)
            test_loss_hist.append(test_loss)
            
            print("Epoch: {}/{}.. ".format(epoch+1, args.Model_epochs).ljust(14),
                  "Train Loss: {:.3f}.. ".format(train_loss).ljust(20),
                  "Test Loss: {:.3f}.. ".format(test_loss).ljust(19),
                  "Test Accuracy: {:.3f}".format(acc))        
            
            # save model pth file
            save_model(names, model, optimizer, acc, epoch, acc_hist, train_loss_hist, test_loss_hist)
            save_hist(names, epoch+1, acc_hist)
        

if __name__=="__main__":
    main()


