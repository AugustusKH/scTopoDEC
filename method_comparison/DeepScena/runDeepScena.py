import random
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import os

from DeepScena import DeepScena
from Network import AutoEncoder, Mutual_net, myBottleneck

# Set seeds for reproducibility
torch.manual_seed(100)
np.random.seed(100)
random.seed(100)

# 1. Device configuration (Safe for GPU and CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# read cell*gene preprocessed matrix
data_df = pd.read_csv("Data/pbmcpre.csv", header=0, index_col=0)

# read file with labels of cell types/clusters.
csv_label = pd.read_csv("Data/celltypes.csv", header=0, index_col=None)

class read_Data(Dataset):
    def __init__(self, data_df, label_df, transform=None):  
        self.transform = transform  
        self.data_df = data_df
        self.label_df = label_df
        self.data = self.load_data()
        
    def load_data(self):
        # Convert data safely
        data_np = np.array(self.data_df).astype('float32') 
        labe = np.array(self.label_df)
        
        B = []
        for i in range(len(data_np)):
            t = data_np[i, :]
            # Reshape to 28x28 (expects exactly 784 highly variable genes)
            t = t.reshape((28, 28)) 
            B.append(t)
            
        B = np.array(B)
        
        # Flatten labels safely
        labee = [int(x) for item in labe for x in item]
        
        data_list = []
        for i in range(len(data_np)):
            data_list.append((B[i], labee[i]))
            
        return data_list
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        image_info, img_label = self.data[index]
        if self.transform:
            sample = self.transform(image_info)
        return sample, img_label, index 

batch_size = 200 
dataset_size = data_df.shape[0] 
transform_fn = transforms.Compose([transforms.ToTensor()])

# Initialize Dataset
train1 = read_Data(data_df, csv_label, transform=transform_fn)

kwargs = {'num_workers': 1} if torch.cuda.is_available() else {}

data_loader = DataLoader(
    dataset=train1,
    batch_size=batch_size,
    shuffle=True, 
    drop_last=False, 
    **kwargs
)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)

if __name__ == '__main__':
    
    num_cluster = 8 # number of cell types/clusters
    pretraining_epoch = 0 # Usually 0 if you are loading pre-trained weights, else >0
    T1 = 2
    T2 = 1
    MaxIter1 = 20
    MaxIter2 = 20
    m = 1.5
    latent_size = 10
    zeta = 0.8
    gamma = 1 - zeta
    dataset_name = 'pbmc'
    a = 0.1
    
    # Init Models and move to Device
    AE = AutoEncoder(myBottleneck, [1, 1, 1]).to(device)
    AE.apply(weights_init)
    
    MNet = Mutual_net(num_cluster).to(device)
    MNet.apply(weights_init)
    
    model = DeepScena(AE, MNet, data_loader, dataset_size, batch_size=batch_size, 
                      pretraining_epoch=pretraining_epoch, MaxIter1=MaxIter1, 
                      MaxIter2=MaxIter2, num_cluster=num_cluster, m=m, T1=T1, T2=T2,
                      latent_size=latent_size, zeta=zeta, gamma=gamma, 
                      dataset_name=dataset_name, a=a)
    
    if pretraining_epoch != 0:
        model.pretrain()
    if MaxIter1 != 0:
        model.first_module()
    if MaxIter2 != 0:
        model.second_module()

    # ==========================================================
    # Extracting the Final Latent Space
    # ==========================================================
    print("Extracting final embeddings...")
    original_label_list = []  
    latent_u_list = []  
    latent_q_list = []  
    predict_list = []  
    cell_index = []  
    
    # FIXED: Load the models ONCE outside the loop!
    AE_final = torch.load(f'AE_Second_module_{dataset_name}.pth').to(device)  
    MNet_final = torch.load(f'MNet_Second_module_{dataset_name}.pth').to(device)
    AE_final.eval()
    MNet_final.eval()
    
    with torch.no_grad():
        for x, target, index in data_loader:
            x = x.to(device)
            _mean, _disp, u, y = AE_final(x)
            q = MNet_final(u)
            
            # FIXED: Safely detach from graph before moving to CPU numpy
            u_np = u.detach().cpu().numpy()
            q_np = q.detach().cpu().numpy()
            
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            target_np = target.numpy()
            index_np = index.numpy()

            for i in range(x.shape[0]):
                cell_index.append(index_np[i])
                latent_u_list.append(u_np[i])
                original_label_list.append(target_np[i])
                predict_list.append(y_pred[i])
                latent_q_list.append(q_np[i])

    # Save Results
    predictedlabels = pd.DataFrame(data=predict_list, columns=['Predicted_labels'])
    uspace = pd.DataFrame(data=latent_u_list)
    cindex = pd.DataFrame(data=cell_index, index=None, columns=['cell_index'])
    
    uspace_result = pd.concat([cindex, uspace], axis=1)
    uspace_result = uspace_result.sort_values(by="cell_index", ascending=True)
    uspace_result.to_csv(f'{dataset_name}_uspace.csv', index=False)
    
    clust_result = pd.concat([cindex, predictedlabels], axis=1)
    clust_result = clust_result.sort_values(by="cell_index", ascending=True)
    clust_result.to_csv(f'{dataset_name}_clusters.csv', index=False)
    
    print(f"Done! Results saved to {dataset_name}_uspace.csv and {dataset_name}_clusters.csv")