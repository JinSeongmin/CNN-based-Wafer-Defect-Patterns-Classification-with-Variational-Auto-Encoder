import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.en_layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2,2))

        self.en_layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2,2))

        self.de_layer1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding=0),
                                       nn.ReLU())

        self.de_layer2 = nn.Sequential(nn.ConvTranspose2d(64, 3, kernel_size = 2, stride = 2, padding=0),
                                       nn.Sigmoid())
    
    def encoder(self, x):
        encode = self.en_layer1(x)
        encode = self.en_layer2(encode)   
        return encode
        
    def decoder(self, x):
        decode = self.de_layer1(x)
        decode = self.de_layer2(decode)
        return decode


    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output



def gen_data(args, x, y): 
    # VAE model make
    print("=== VAE training start")
    VAE_model = VAE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(VAE_model.parameters(), lr=args.VAE_lr)
    train_loader  = torch.utils.data.DataLoader(x, args.VAE_batch_size, shuffle=True)
    
    steps = 0
    total_steps = len(train_loader)
    
    losses = []
    iterations = []
    
    # VAE training
    for epoch in range(args.VAE_epochs):    
        running_loss = 0.0
        
        for i, wafer in enumerate(train_loader):
            steps += 1
            wafer = wafer.to(device)
            optimizer.zero_grad()
            outputs = VAE_model(wafer)
            loss = criterion(outputs, wafer)
            loss.backward()
            running_loss += loss.item()*wafer.shape[0]
            optimizer.step()
            
            if steps % total_steps == 0:
                VAE_model.eval()
                print("Epoch: {}/{}".format(epoch+1, args.VAE_epochs),
                      "=> loss : %.3f"%(running_loss/total_steps))
                steps = 0
                iterations.append(i)
                losses.append(running_loss / total_steps)
                VAE_model.train()
    
    
    print("=== VAE training complete \n")
    
    
    # Generate additional data using VAE encoder
    print("=== Generating additional data start")
    for f in np.unique(y) : 
        if f == "none" : 
            continue
        
        w_f = x[np.where(y==f)[0]].to(device)
        
        gen_x = torch.zeros((1, 3, 28, 28))
        with torch.no_grad():
            encoded_x = VAE_model.encoder(w_f).cpu()
            
            for i in range((3000 // len(w_f)) + 1):
                noised_encoded_x = (encoded_x + torch.tensor(np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 128, 7, 7))).cpu()).to(device)
                noised_decoded_x = VAE_model.decoder(noised_encoded_x.float()).cpu()
                gen_x = torch.cat([gen_x, noised_decoded_x], axis=0)
            
            gen_y = np.full((len(gen_x), 1), f)
        
        gen_x = gen_x[1:]
        gen_y = gen_y[1:]
        
        x = torch.cat([x, gen_x.to(device)], axis=0)
        y = np.concatenate((y, gen_y))
    
    
    print("=== Generating additional data complete \n")
    print("=== After Generate x shape : {}, y shape : {}".format(x.shape, y.shape))
    
    
    return x, y
    

