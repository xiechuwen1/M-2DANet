import torch
import torch.nn as nn
import math

class SingleBranchTransformer(nn.Module):

    def __init__(self, input_dim=310, mlp_hidden_dims=[256, 128], model_dim=64,
                 seq_len=14, n_layers=3, n_heads=8, dim_feedforward=256):
        super().__init__()
        self.initial_mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dims[0]), nn.LeakyReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]), nn.LeakyReLU(),
            nn.Linear(mlp_hidden_dims[1], model_dim)
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len + 1, model_dim))
        nn.init.normal_(self.class_token, std=.02)
        nn.init.normal_(self.pos_embedding, std=.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=dim_feedforward,
            batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        x = self.initial_mlp(x)
        cls_token_expanded = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token_expanded, x), dim=1)
        x = x + self.pos_embedding
        transformer_output = self.transformer_encoder(x)
        return transformer_output[:, 0]


class TT_CDAN_FeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.de_branch = SingleBranchTransformer(**kwargs)
        self.psd_branch = SingleBranchTransformer(**kwargs)

    def forward(self, de_input, psd_input):
        de_features = self.de_branch(de_input)
        psd_features = self.psd_branch(psd_input)
        final_features = torch.cat((de_features, psd_features), dim=1)
        return final_features

class EmotionClassifier(nn.Module):

    def __init__(self, feature_dim=128, num_classes=3):
        super().__init__()
        self.classifier_layer = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        logits = self.classifier_layer(features)
        return logits


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ConditionalDomainDiscriminator(nn.Module):

    def __init__(self, feature_dim=128, num_classes=3, hidden_dim1=512, hidden_dim2=256):
        super().__init__()
        self.multilinear_map_dim = 1024

        self.discriminator_net = nn.Sequential(
            nn.Linear(self.multilinear_map_dim, hidden_dim1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )

        self.random_matrix = nn.Parameter(torch.randn(feature_dim * num_classes, self.multilinear_map_dim))

    def forward(self, features, classifier_softmax_output, grl_alpha=1.0):

        reversed_features = GradientReversalLayer.apply(features, grl_alpha)

        g = classifier_softmax_output
        f_g_outer = torch.bmm(reversed_features.unsqueeze(2), g.unsqueeze(1))
        f_g_flat = f_g_outer.view(f_g_outer.size(0), -1)

        conditioned_input = torch.mm(f_g_flat, self.random_matrix)

        domain_pred = self.discriminator_net(conditioned_input)

        return domain_pred.squeeze()
