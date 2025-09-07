import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGC_Layer(nn.Module):

    def __init__(self, in_features, out_features, num_nodes):

        super(DGC_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes

        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))

        self.A = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.A, gain=1.414)

    def forward(self, H):

        HW = torch.matmul(H, self.W)

        A_sym = (self.A + self.A.T) / 2
        A_norm = F.relu(A_sym)

        D = torch.sum(A_norm, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_matrix = torch.diag_embed(D_inv_sqrt)

        I = torch.eye(self.num_nodes, device=H.device).unsqueeze(0)
        L_norm = I - torch.matmul(torch.matmul(D_matrix, A_norm), D_matrix)

        output = torch.matmul(L_norm, HW)

        return F.relu(output)


class IDDA_Net(nn.Module):

    def __init__(self, in_features, dgc_out_features, num_nodes, num_classes, dropout=0.3):

        super(IDDA_Net, self).__init__()

        self.feature_extractor = DGC_Layer(
            in_features=in_features,
            out_features=dgc_out_features,
            num_nodes=num_nodes
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * dgc_out_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        features = self.feature_extractor(x)

        features_flat = self.flatten(features)

        output = self.classifier(features_flat)

        return output