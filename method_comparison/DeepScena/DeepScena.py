import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from Metrics import nmi, acc, ari
import pandas as pd
import torch.nn.functional as F



class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp):
        eps = 1e-10
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        return t1 + t2
        

class DeepScena(nn.Module):
    def __init__(self, AE, MNet, data_loader, dataset_size, batch_size=200, 
                 pretraining_epoch=0, MaxIter1=50, MaxIter2=20, num_cluster=9, 
                 m=1.5, T1=2, T2=1, latent_size=10, zeta=0.8, gamma=0.2, 
                 dataset_name='zeisel', a=0.1):
        super(DeepScena, self).__init__()
        self.AE = AE
        self.MNet = MNet
        self.u_mean = torch.zeros([num_cluster, latent_size])
        self.batch_size = batch_size
        self.pretraining_epoch = pretraining_epoch
        self.MaxIter1 = MaxIter1
        self.MaxIter2 = MaxIter2
        self.num_cluster = num_cluster
        self.data_loader = data_loader
        self.dataset_size = dataset_size
        self.m = m
        self.T1 = T1
        self.T2 = T2
        self.latent_size = latent_size
        self.dataset_name = dataset_name
        self.zeta = zeta
        self.gamma = gamma
        self.a = a
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Kmeans_model_evaluation(self):
        self.AE.eval()
        datas = np.zeros([self.dataset_size, self.latent_size])
        label_true = np.zeros(self.dataset_size)
        ii = 0
        
        with torch.no_grad(): # FIXED: Prevent memory leak during eval
            for x, target, index in self.data_loader:
                x = x.to(self.device) # FIXED: Removed deprecated Variable()
                _mean, _disp, u, y = self.AE(x)
                u = u.cpu()
                
                current_batch_size = x.shape[0]
                datas[ii * min(self.batch_size, current_batch_size):(ii + 1) * min(self.batch_size, current_batch_size), :] = u.data.numpy()
                label_true[ii * min(self.batch_size, current_batch_size):(ii + 1) * min(self.batch_size, current_batch_size)] = target.numpy()
                ii += 1
                
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas)
        label_pred = kmeans.labels_
        print('ARI', ari(label_true, label_pred))

    def pretrain(self):
        self.AE.train()
        self.AE.to(self.device)
        for param in self.AE.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(self.AE.parameters())
        
        for T in range(0, self.pretraining_epoch):
            print('Pretraining Iteration: ', T + 1)
            for x, target, index in self.data_loader:
                optimizer.zero_grad()
                x = x.to(self.device)
                _mean, _disp, u, y = self.AE(x)
                loss = nn.MSELoss()(x, y)
                loss.backward()
                optimizer.step()
 
        # FIXED: Removed redundant duplicate save/load block
        torch.save(self.AE, f'AE_{self.dataset_name}_pretrain.pth')
        self.AE = torch.load(f'AE_{self.dataset_name}_pretrain.pth')
        return self.AE

    def initialization(self): 
        print("-----Start clustering--------")
        self.AE.to(self.device)
        datas = np.zeros([self.dataset_size, self.latent_size])
        ii = 0
        
        with torch.no_grad():
            for x, target, index in self.data_loader:
                x = x.to(self.device)
                _mean, _disp, u, y = self.AE(x)
                u = u.cpu()
                current_batch_size = x.shape[0]
                datas[ii * min(self.batch_size, current_batch_size):(ii + 1) * min(self.batch_size, current_batch_size)] = u.data.numpy()
                ii += 1
                
        kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(datas) 
        self.u_mean = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        return self.AE, self.u_mean

    def clustering_cost(self, x, y, u, p, u_means):
         return torch.matmul(p, torch.sum(torch.pow(x - y, 2), dim=1)) + self.a * torch.matmul(p, torch.sum(torch.pow(u - u_means, 2), dim=1))  
         
    def nb_loss(self, x, mean, disp):
        nb_loss_fn = NBLoss().to(self.device)  
        return nb_loss_fn(x, mean, disp)

    def update_cluster_centers(self):
        self.AE.eval()
        den = torch.zeros([self.num_cluster], device=self.device)
        num = torch.zeros([self.num_cluster, self.latent_size], device=self.device)
        
        with torch.no_grad(): # FIXED: Wrap in no_grad for speed and safety
            for x, target, index in self.data_loader:
                x = x.to(self.device)
                _mean, _disp, u, y = self.AE(x)
                current_batch_size = x.shape[0]
                
                p = torch.zeros([min(self.batch_size, current_batch_size), self.num_cluster], device=self.device)
                for j in range(0, self.num_cluster):
                    p[:, j] = torch.sum(torch.pow(u.unsqueeze(0).repeat(self.num_cluster, 1, 1)[j, :, :] - self.u_mean[j, :].unsqueeze(0).repeat(min(self.batch_size, current_batch_size), 1), 2), dim=1)
                
                p = torch.pow(p, -1 / (self.m - 1))
                sum1 = torch.sum(p, dim=1)
                p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
                p = torch.pow(p, self.m)
                
                for kk in range(0, self.num_cluster):
                    den[kk] += torch.sum(p[:, kk])
                    num[kk, :] += torch.matmul(p[:, kk].t(), u)
                    
            for kk in range(0, self.num_cluster):
                self.u_mean[kk, :] = torch.div(num[kk, :], den[kk])
                
        self.AE.train()
        return self.u_mean

    def model_evaluation(self, first_module):
        pred_labels = np.zeros(self.dataset_size)
        true_labels = np.zeros(self.dataset_size)
        ii = 0
        
        with torch.no_grad():
            for x, target, index in self.data_loader:
                x = x.to(self.device)
                _mean, _disp, u, y = self.AE(x)
                current_batch_size = x.shape[0]
                
                if first_module:
                    u_expanded = u.unsqueeze(0).repeat(self.num_cluster, 1, 1)
                    p = torch.zeros([min(self.batch_size, current_batch_size), self.num_cluster], device=self.device)
                    for j in range(0, self.num_cluster):
                        p[:, j] = torch.sum(torch.pow(u_expanded[j, :, :] - self.u_mean[j, :].unsqueeze(0).repeat(min(self.batch_size, current_batch_size), 1), 2), dim=1)
                    p = torch.pow(p, -1 / (self.m - 1))
                    sum1 = torch.sum(p, dim=1)
                    p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster))
                    pred_y = torch.argmax(p, dim=1)
                else:
                    q = self.MNet(u)
                    pred_y = torch.argmax(q, dim=1)
                    
                pred_labels[ii * min(self.batch_size, current_batch_size):(ii + 1) * min(self.batch_size, current_batch_size)] = pred_y.cpu().numpy()
                true_labels[ii * min(self.batch_size, current_batch_size):(ii + 1) * min(self.batch_size, current_batch_size)] = target.numpy()
                ii += 1

        NMI = nmi(true_labels, pred_labels)
        ARI = ari(true_labels, pred_labels)
        print('NMI', NMI)
        print('ARI', ARI)
        self.AE.train()
        return NMI, ARI

    def first_module(self):
        self.AE, self.u_mean = self.initialization()
        self.AE.to(self.device)
        self.AE.train()
        
        optimizer = optim.SGD(self.AE.parameters(), lr=0.000001, momentum=0.9)

        ARIlist, NMIlist = [], []
        for T in range(0, self.MaxIter1):
            print('First Module Iteration: ', T + 1)
            if T % self.T1 == 1:
                self.u_mean = self.update_cluster_centers()
                
            for x, target, index in self.data_loader:
                current_batch_size = x.shape[0]
                x = x.to(self.device)
                
                _mean, _disp, u1, y = self.AE(x)
                
                # Re-calculate P using the detached U
                u_detached = u1.detach()
                p = torch.zeros([min(self.batch_size, current_batch_size), self.num_cluster], device=self.device)
                for j in range(0, self.num_cluster):
                    p[:, j] = torch.sum(torch.pow(u_detached - self.u_mean[j, :].unsqueeze(0).repeat(min(self.batch_size, current_batch_size), 1), 2), dim=1)
                
                p = torch.pow(p, -1 / (self.m - 1))
                sum1 = torch.sum(p, dim=1)
                p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster)).detach()
                p = torch.pow(p.T, self.m)
                
                for i in range(0, self.num_cluster):
                    _mean, _disp, u1, y = self.AE(x) 
                    loss1 = self.clustering_cost(x.view(-1, 784), y.view(-1, 784), u1, p[i, :].unsqueeze(0), self.u_mean[i, :].unsqueeze(0).repeat(min(self.batch_size, current_batch_size), 1))
                    
                    x1 = torch.round(x)
                    loss2 = torch.matmul(p[i, :].unsqueeze(0), torch.sum(self.nb_loss(x1.view(-1, 784), _mean, _disp), dim=1))

                    beta = 0.1 
                    loss = loss1 + beta * loss2
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            NMI, ARI = self.model_evaluation(first_module=True)
            ARIlist.append(ARI)
            NMIlist.append(NMI)

        torch.save(self.AE, f'AE_First_module_{self.dataset_name}.pth')
        torch.save(self.u_mean, f'Centers_{self.dataset_name}.pth')
        pd.DataFrame(ARIlist).to_csv("ARI1.csv")
        pd.DataFrame(NMIlist).to_csv("NMI1.csv")

    def second_module(self):
        self.AE = torch.load(f'AE_First_module_{self.dataset_name}.pth')
        self.u_mean = torch.load(f'Centers_{self.dataset_name}.pth').to(self.device)

        AE_optimizer = optim.Adam(self.AE.parameters(), lr=0.000001)
        MNet_optimizer = optim.Adam(self.MNet.parameters())
        
        self.MNet.to(self.device)
        self.MNet.train()
        self.AE.to(self.device)
        self.AE.train()
        
        ARIlist2, NMIlist2 = [], []
        ARI_prev = 0.0
        
        for T in range(0, self.MaxIter2):
            print('Second Module Iteration: ', T + 1)
            for x, target, index in self.data_loader:
                current_batch_size = x.shape[0]
                x = x.to(self.device)
                
                _mean, _disp, u1, y = self.AE(x)
                u_detached = u1.detach()
                
                p = torch.zeros([min(self.batch_size, current_batch_size), self.num_cluster], device=self.device)
                for j in range(0, self.num_cluster):
                    p[:, j] = torch.sum(torch.pow(u_detached - self.u_mean[j, :].unsqueeze(0).repeat(min(self.batch_size, current_batch_size), 1), 2), dim=1)
                
                p = torch.pow(p, -1 / (self.m - 1))
                sum1 = torch.sum(p, dim=1)
                p = torch.div(p, sum1.unsqueeze(1).repeat(1, self.num_cluster)).detach()

                _mean, _disp, u, y = self.AE(x)
                q = self.MNet(u)
                
                if T <= self.T2:
                    multiply = torch.mm(p, p.T)
                else:
                    multiply = torch.mm(q, q.T)
                    
                q = torch.clamp(q, min=0.01, max=0.99)

                new_p1 = torch.where(multiply > self.zeta, 1 - torch.mm(q, q.T), torch.zeros_like(multiply))
                new_p2 = torch.where(multiply < self.gamma, torch.mm(q, q.T), torch.zeros_like(multiply))

                loss1 = 0.0001 * torch.sum(new_p1 + new_p2)

                MNet_optimizer.zero_grad()
                AE_optimizer.zero_grad()
                loss1.backward()
                MNet_optimizer.step()
                AE_optimizer.step()

            if T <= self.T2:
                self.u_mean = self.update_cluster_centers()
                
            NMI, ARI = self.model_evaluation(first_module=False)
            NMIlist2.append(NMI)
            ARIlist2.append(ARI)
            
            if ARI > ARI_prev:
                ARI_prev = ARI
                torch.save(self.AE, f'AE_Second_module_{self.dataset_name}.pth')
                torch.save(self.MNet, f'MNet_Second_module_{self.dataset_name}.pth')

        pd.DataFrame(ARIlist2).to_csv("ARI2.csv")
        pd.DataFrame(NMIlist2).to_csv("NMI2.csv")