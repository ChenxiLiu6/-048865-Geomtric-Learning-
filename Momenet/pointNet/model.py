import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.neighbors import KDTree


class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
       

    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))
      
        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix

    
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.second_order = Second_Order()
        
        self.conv1_1 = nn.Conv1d(3,64,1)
        self.conv1_2 = nn.Conv1d(6,64,1)
        self.conv1_3 = nn.Conv1d(18,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, input, channel):
        # input.shape == (B, 3, N)
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)            # xb.shape == (B, 3, N)
        
        if channel == "pointnet":
            xb = F.relu(self.bn1(self.conv1_1(xb)))
        elif channel == "vn":
            normals = self.second_order(input.transpose(1,2), channel).transpose(1,2)   # normals.shape == (B, 3, N)
            xb = torch.cat((xb, normals), dim=1)                                        # xb.shape == (B, 6, N)
            xb = F.relu(self.bn1(self.conv1_2(xb)))
        elif channel == "hp":
            harmonic = self.second_order(input.transpose(1,2), channel).transpose(1,2)  # harmonic.shape == (B, 15, N)
            xb = torch.cat((xb, harmonic), dim=1)                                        # xb.shape == (B, 18, N)
            xb = F.relu(self.bn1(self.conv1_3(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

    
class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, channel):
        xb, matrix3x3, matrix64x64 = self.transform(input, channel)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64

    
# Momenet Model#
class Spatial_Trans(nn.Module):
    def __init__(self):
        super().__init__()
        self.second_order = Second_Order()
        self.conv1_1 = nn.Conv2d(12, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv1_2 = nn.Conv2d(22, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv1_3 = nn.Conv2d(15, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv1_4 = nn.Conv2d(25, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv3 = nn.Conv2d(128, 1024, kernel_size=[1,1], stride=[1,1], bias=False)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3*3)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, input, channel):
        """
        # param: input: (B, N, C_in) -> [32, 1024, 3]
        # param: channel(str): "second", "third", "vertex" -> feature order
        
        @ return tensor: (B, 3, 3)
        """
        batch_size = input.size(0)
        num_points = input.size(1)
        
        edge_feature = self.second_order(input, channel)                 # (B, N, K, C_out) -> [32, 1024, 20, 12] ✅
        edge_feature_trans = edge_feature.transpose(1, 3)                # (B, C, K, N) -> [32, 12, 20, 1024] ✅
        #print("edge_feature_trans shape: ", edge_feature_trans.shape)
        
        if channel == "second":
            net0 = F.relu(self.bn1(self.conv1_1(edge_feature_trans)))    # (B, C, K, N) -> [32, 64, 20, 1024] ✅
        elif channel == "third":
            net0 = F.relu(self.bn1(self.conv1_2(edge_feature_trans)))
        elif channel == "vn2":
            net0 = F.relu(self.bn1(self.conv1_3(edge_feature_trans)))
        elif channel == "vn3":
            net0 = F.relu(self.bn1(self.conv1_4(edge_feature_trans)))
        
        #print("Spatial Trans net0 shape: ", net0.shape)
        
        #net0 = F.relu(self.bn1(self.conv1(edge_feature_trans)))          
        net1 = F.relu(self.bn2(self.conv2(net0)))                        # (B, C, K, N) -> [32, 128, 20, 1024]
        
        net2,_ = torch.max(net1, dim=-2, keepdim=True)                   # (B, C, K, N) -> [32, 128, 1, 1024]
        
        net3 = F.relu(self.bn3(self.conv3(net2)))                        # (B, C, K, N) -> [32, 1024, 1, 1024]
        
        net4 = nn.MaxPool2d(kernel_size=[1, net3.size(-1)], stride=[2, 2])(net3) # (B, C, K, N) -> [32, 1024, 1, 1]
        
        net4 = torch.reshape(net4, (batch_size, -1))                     # (B, C) -> [32, 1024]
        
        net5 = F.relu(self.bn4(self.fc1(net4)))                          # (B, C) -> [32, 512]
        net6 = F.relu(self.bn5(self.fc2(net5)))                          # (B, C) -> [32, 256]
        
        #initialize as identity
        init = torch.eye(3, requires_grad=True).repeat(batch_size,1,1)
        if input.is_cuda:
            init=init.cuda()
        matrix3x3 = self.fc3(net6).view(-1, 3, 3) + init                 # (B, C, C) -> [32, 3, 3]
        
        return edge_feature, matrix3x3

    
class Second_Order(nn.Module):
    def __init__(self):
        super().__init__()
    
    # Get 1, 2 order momenets
    def get_second(self, input):
        # input.shape == (B, N, 3)
        bs = input.size(0)
        second = torch.empty((input.size(0), input.size(1), 9))
        for i in range(bs):
            a = input[i]
            square = a**2

            xy = (a[:, 0] * a[:, 1]).reshape(-1, 1)
            xz = (a[:, 0] * a[:, 2]).reshape(-1, 1)
            yz = (a[:, 1] * a[:, 2]).reshape(-1, 1)

            b = torch.cat((a, square, xy, xz, yz), dim=1)
            second[i] = b
        # second.shape == (bs, n, 9)
        return second
    
    # Get 1, 2, 3 order momenets
    def get_third(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        num_points = input.size(1)
        third = torch.empty((bs, num_points, 19))
        for i in range(bs):
            a = input[i]
            square = a**2
            cube = a**3
            
            xy = (a[:, 0] * a[:, 1]).reshape(-1, 1)
            xz = (a[:, 0] * a[:, 2]).reshape(-1, 1)
            yz = (a[:, 1] * a[:, 2]).reshape(-1, 1)
            
            x2y = (a[:, 0] * a[:, 0] * a[:, 1]).reshape(-1, 1)
            x2z = (a[:, 0] * a[:, 0] * a[:, 2]).reshape(-1, 1)
            y2x = (a[:, 1] * a[:, 1] * a[:, 0]).reshape(-1, 1)
            y2z = (a[:, 1] * a[:, 1] * a[:, 2]).reshape(-1, 1)
            z2x = (a[:, 2] * a[:, 2] * a[:, 0]).reshape(-1, 1)
            z2y = (a[:, 2] * a[:, 2] * a[:, 1]).reshape(-1, 1)
            
            xyz = (a[:, 0] * a[:, 1] * a[:, 2]).reshape(-1, 1)
            
            b = torch.cat((a, square, cube, xy, xz, yz, x2y, x2z, y2x, y2z, z2x, z2y, xyz), dim=1)
            third[i] = b
        # third.shape == (bs, n, 19)
        return third
    
    # Estimate PointCloud Normals 
    def estimate_normals(self, input):
        # input.shape == (bs, n, 3)
        batch_size = input.size(0)
        num_points = input.size(1)
        
        # ------ (1) Find k==20 neighbors for each point --------------- #
        knn = self.KNN(input)                          # knn.shape == (B, N, K, C) -> [32, 1024, 20, 3]
        
        # ------ (2) Compute Centroid m == plane origin ---------------- #
        m = torch.mean(knn, dim=-2)                    # m.shape == (B, N, C) -> [32, 1024, 3]
        
        # ------ (3) Compute Y=(y1,... y20)-> yi = xi - m -------------- #
        M = m.unsqueeze(2).repeat(1, 1, 20, 1)  
        Y = knn - M                                    # Y.shape == (B, N, K, C) -> [32, 1024, 20, 3]
        S = torch.empty(batch_size, num_points, 3, 3)  # S.shape == (B, N, C, C) -> [32, 1024, 3, 3]
        for i in range(batch_size):
            mul = torch.bmm(Y[i].transpose(1,2), Y[i])
            S[i] = mul
        
        # ------ (4) Compute plane normal: smallest eigvec of each 3x3 matrix in S -------- #
        e, v = torch.symeig(S, eigenvectors=True)
        normals = v[:, :, :, 0]                        # n.shape == (B, N, C) -> [32, 1024, 3]
        
        return normals
        
    def get_hp(self, input):
        # input.shape == (B, N, 3)
        batch_size = input.size(0)
        num_points = input.size(1)
        
        # Add harmonic pre lifting:
        # sinx, sin2x, cosx, cos2x; siny, sin2y, cosy, cos2y; sinz, sin2z, cosz, cos2z; 
        extra_ebd = torch.empty(batch_size, num_points, 15)
        sin = torch.sin(input)         # sin.shape == (B, N, 3)
        cos = torch.cos(input)         # cos.shape == (B, N, 3)
        sin2 = torch.sin(2 * input)    # sin2.shape == (B, N, 3)
        cos2 = torch.cos(2 * input)    # cos2.shape == (B, N, 3)
        
        harmonic = torch.cat((input, sin, cos, sin2, cos2), dim=-1) # harmonic.shape == (B, N, C) -> [32, 1024, 15]
        
        return harmonic
            
        
    def get_vn2(self, input):
        # input.shape == (B, N, 3)
        batch_size = input.size(0)
        num_points = input.size(1)
        vn2 = torch.empty((batch_size, num_points, 12))
        
        # ------ (1) get 1 and 2 order momenets ------ #
        second = self.get_second(input)                # second.shape == (B, N, 9)
        
        # ------ (2) get estimated vertex normals ---- #
        normals = self.estimate_normals(input)         # normals.shape == (B, N, 3)
        
        # ------ (3) Concat momenets and normals ----- #
        vn2 = torch.cat((second, normals), dim=-1)     # vn2.shape == (B, N, 12)
        
        return vn2
    
    def get_vn3(self, input):
        # input.shape == (B, N, 3)
        batch_size = input.size(0)
        num_points = input.size(1)
        vn3 = torch.empty((batch_size, num_points, 22))
        
        # ------ (1) get 1, 2, 3 order momenets ------ #
        third = self.get_third(input)                  # third.shape == (B, N, 19)
        
        # ------ (2) get estimated vertex normals ---- #
        normals = self.estimate_normals(input)         # normals.shape == (B, N, 3)
        
        # ------ (3) Concat momenets and normals ----- #
        vn3 = torch.cat((third, normals), dim=-1)      # vn3.shape == (B, N, 22)
        
        return vn3
        
        
    def KNN(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        output = torch.empty(input.size(0), input.size(1), 20, 3)
        for i in range(bs):
            pcloud = input[i]                                      # pcloud.shape == (n, 3) -> [1024, 3]
            tree = KDTree(pcloud.detach().numpy(), leaf_size=512)              
            _, index = tree.query(pcloud.detach().numpy(), k=21)   # indices of 21 closest neighbors (including self)
            index = torch.from_numpy(index[:, 1:21])               # indices of 20 closest neighbors -> [1024, 20]
            for j in range(20):
                d = pcloud - pcloud[index[:, j], :]
                if j == 0:
                    out = d
                else:
                    out = torch.cat((out, d), dim=1)
            out = out.view(out.size(0), 20, 3)                     # out.shape == (n, k==20, 3) -> [1024, 20, 3]
            output[i] = out                                        # output.shape == (bs, n, k=20, 3)
                
        return output 
        
    def forward(self, input, channel):
        """
        # param: input.shape == (B, N, C_in) -> [32, 1024, 3]
        # param: channel(str): "second", "third", "vn2", "vn3" -> feature order
        
        @ return: edge_feature: (B, N, K, C_out)
          (1) second: (B, N, K, C_out) == [32, 1024, 20, 12]
          (2) third:  (B, N, K, C_out) == [32, 1024, 20, 22]
          (3) vn2: (B, N, K, C_out) == [32, 1024, 20, 15]
          (4) vn3: (B, N, K, C_out) == [32, 1024, 20, 25]
          (5) vn (for PointNet): (B, N, C) == [32, 1024, 3]
        """
        
        if channel == "second":
            nc = self.get_second(input)                       # nc.shape == (B, N, C_out) -> [32, 1024, 9]
        elif channel == "third":
            nc = self.get_third(input)
        elif channel == "vn2":
            nc = self.get_vn2(input)
        elif channel == "vn3":
            nc = self.get_vn3(input)
        elif channel == "vn":
            normals = self.estimate_normals(input)
            return normals
        elif channel == "hp":
            harmonic = self.get_hp(input)
            return harmonic
                                    
        nkc = nc.unsqueeze(2).repeat(1, 1, 20, 1)            # nkc.shape == (B, N, K, C) -> [32, 1024, 20, 9/19/12/22]
        
        knn = self.KNN(input)                                # knn.shape == (B, N, K, C) -> [32, 1024, 20, 3]
        
        edge_feature = torch.cat((nkc, knn), dim=3)          # edge_feature.shape == (B, N, K, C) -> [32, 1024, 20, 12/22/15/25]
        
        return edge_feature
        
        
class Momenet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.transform = Spatial_Trans()
        self.second_order = Second_Order()
        
        self.conv1_1 = nn.Conv2d(12, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv1_2 = nn.Conv2d(22, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv1_3 = nn.Conv2d(15, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        self.conv1_4 = nn.Conv2d(25, 64, kernel_size=[1,1], stride=[1,1], bias=False)
        
        
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input, channel):
        """
        # param: input(pointcloud tensor): (B, N, C) -> [32, 1024, 3]
        # param: channel(str): "second", "third", "vn2", "vn3" -> feature order
        
        @ return: output(classes): (B, C) -> [32, 10]
                  matrix3x3(Spatial Trans output): (B, C, C) -> [32, 3, 3]
        """
        batch_size = input.size(0)
        num_points = input.size(1)

        _, matrix3x3 = self.transform(input, channel)          # matrix3x3.shape == (B, C, C) -> [32, 3, 3]
            
        point_cloud_transformed = torch.bmm(input, matrix3x3)  # (B, N, C) -> [32, 1024, 3]
        
        # edge_feature.shape == (B, N, K, C) -> [32, 1024, 20, 12/22/15/25]
        edge_feature = self.second_order(point_cloud_transformed, channel) 
        net0 = edge_feature.transpose(1, 3)                    # net0.shape == (B, C, K, N) -> [32, 12, 20, 1024] ✅
        
        if channel == "second":
            net = F.relu(self.bn1(self.conv1_1(net0)))         # net.shape == (B, C, K, N) -> [32, 64, 20, 1024] ✅
        elif channel == "third":
            net = F.relu(self.bn1(self.conv1_2(net0)))
        elif channel == "vn2":
            net = F.relu(self.bn1(self.conv1_3(net0)))
        elif channel == "vn3":
            net = F.relu(self.bn1(self.conv1_4(net0)))
            
        
        net1,_ = torch.max(net, dim=-2, keepdim=True)          # net_.shape == (B, C, K, N) -> [32, 64, 1, 1024]                 
        
        net1 = torch.reshape(net1, (batch_size, 64, num_points)) # net1.shape == (B, C, N) -> [32, 64, 1024] 
        
        net2 = F.relu(self.bn2(self.conv2(net1)))             # net2.shape == (B, C, N) -> [32, 64, 1024] 
        net3 = F.relu(self.bn2(self.conv2(net2)))             # net3.shape == (B, C, N) -> [32, 64, 1024]
        net4 = F.relu(self.bn2(self.conv2(net3)))             # net4.shape == (B, C, N) -> [32, 64, 1024]
        net5 = F.relu(self.bn3(self.conv3(net4)))             # net5.shape == (B, C, N) -> [32, 128, 1024]
        net6 = self.bn4(self.conv4(net5))                     # net6.shape == (B, C, N) -> [32, 1024, 1024]
        
        net6 = nn.MaxPool1d(net6.size(-1))(net6)              # net6.shape == (B, C, N) -> [32, 1024, 1]
        net6 = torch.reshape(net6, (batch_size, -1))          # net6.shape == (B, C) -> [32, 1024]
        
        net7 = self.dropout(F.relu(self.bn5(self.fc1(net6)))) # net7.shape == (B, C) -> [32, 512]
        net8 = self.dropout(F.relu(self.bn6(self.fc2(net7)))) # net8.shape == (B, C) -> [32, 256]
        output = self.fc3(net8)                               # output.shape == (B, C) -> [32, 10]
        
        return self.logsoftmax(output), matrix3x3
        
       
        