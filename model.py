import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d, Conv2d, LSTM, BatchNorm1d
from torch_geometric.utils import to_dense_batch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SpatialEncoder(nn.Module):
    def __init__(self, channels=62, out_dim=128):
        super().__init__()
        self.conv1 = Conv1d(channels, 64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm1d(64)
        self.conv2 = Conv1d(64, out_dim, kernel_size=3, padding=1)
        self.bn2 = BatchNorm1d(out_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, channels=62, hidden_dim=128, out_dim=128):
        super().__init__()
        self.lstm = LSTM(input_size=channels, hidden_size=hidden_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.fc = Linear(hidden_dim, out_dim)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x_permuted)
        x = self.fc(h_n[-1])
        return x

class AttentionFrequencyEncoder(nn.Module):
    def __init__(self, channels=62, bands=5, samples_per_band=265, out_dim=128, nhead=4):
        super().__init__()
        self.bands = bands

        feature_dim = 64
        self.band_feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 7), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, feature_dim)
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=1)

        self.fc_out = nn.Linear(bands * feature_dim, out_dim)

    def forward(self, x):
        bs = x.shape[0]

        x = x.view(bs, 62, self.bands, 265)

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(bs * self.bands, 1, 62, 265)

        band_features = self.band_feature_extractor(x)

        sequence = band_features.view(bs, self.bands, -1)

        attended_sequence = self.transformer_encoder(sequence)

        flattened_output = attended_sequence.reshape(bs, -1)

        final_output = self.fc_out(flattened_output)

        return final_output


class CrossAttentionFuser(nn.Module):
    def __init__(self, feature_dim=128, nhead=4):
        super().__init__()
        self.feature_dim = feature_dim

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            kdim=feature_dim * 2,
            vdim=feature_dim * 2,
            num_heads=nhead,
            batch_first=True
        )

        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, spatial_feat, temporal_feat, frequency_feat):
        query = spatial_feat.unsqueeze(1)

        context = torch.cat([temporal_feat, frequency_feat], dim=1)
        context = context.unsqueeze(1)

        attn_output, _ = self.cross_attention(query=query, key=context, value=context)

        attended_feat = self.layer_norm1(query + attn_output)

        ffn_output = self.ffn(attended_feat)

        final_attended_feat = self.layer_norm2(attended_feat + ffn_output)

        final_attended_feat = final_attended_feat.squeeze(1)

        combined_features = torch.cat([spatial_feat, temporal_feat, frequency_feat, final_attended_feat], dim=1)

        return combined_features

class TriplePathExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(out_dim=128)
        self.temporal_encoder = TemporalEncoder(out_dim=128)
        self.frequency_encoder = AttentionFrequencyEncoder(out_dim=128)

        self.feature_fuser = CrossAttentionFuser(feature_dim=128)

        self.final_feature_dim = 128 * 4
    def forward(self, x, edge_index, batch):
        x_dense, _ = to_dense_batch(x, batch)
        spatial_feat = self.spatial_encoder(x_dense)
        temporal_feat = self.temporal_encoder(x_dense)
        frequency_feat = self.frequency_encoder(x_dense)
        combined_features = self.feature_fuser(spatial_feat, temporal_feat, frequency_feat)
        return combined_features

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        reversed_grad = grad_output.neg() * ctx.alpha
        return reversed_grad, None


class DomainClassifier(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = Linear(input_dim + num_classes, 1)

    def forward(self, features, class_one_hot):
        if class_one_hot is not None:
            combined_input = torch.cat([features, class_one_hot], dim=1)
        else:
            combined_input = features
        output = self.fc(combined_input)
        return torch.sigmoid(output.squeeze(-1))

class TriplePath_DANN_Model(torch.nn.Module):
    def __init__(self, classes=3):
        super().__init__()
        self.feature_extractor = TriplePathExtractor()
        feature_dim = self.feature_extractor.final_feature_dim
        self.grl_layer = GradientReversalLayer.apply
        self.global_domain_classifier = DomainClassifier(feature_dim, 0)
        self.conditional_domain_classifier = DomainClassifier(feature_dim, classes)

        self.emotion_classifier = Linear(feature_dim, classes)

    def forward(self, x, edge_index, batch, alpha=1.0):
        processed_features = self.feature_extractor(x, edge_index, batch)
        reversed_features = self.grl_layer(processed_features, alpha)
        global_domain_output = self.global_domain_classifier(reversed_features, None)
        class_output = self.emotion_classifier(processed_features)
        pred = F.softmax(class_output, dim=1)
        return class_output, pred, global_domain_output, processed_features