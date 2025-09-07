import torch
import torch.nn as nn

class SEDA_EEG_Backbone(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, feature_dim=128, num_classes=4, dropout_rate=0.25):
        super(SEDA_EEG_Backbone, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, feature_dim), nn.ReLU(), nn.Dropout(dropout_rate)
        )
        self.classifier_block = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier_block(features)
        return output, features


class GradientReversalLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainClassifier(nn.Module):

    def __init__(self, feature_dim=128, hidden_dim=256):
        super(DomainClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, features):
        return self.classifier(features)


class SEDA_EEG_Framework(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim=256, feature_dim=128, dropout_rate=0.25):
        super(SEDA_EEG_Framework, self).__init__()
        self.backbone = SEDA_EEG_Backbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        self.domain_classifier = DomainClassifier(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, x, alpha=1.0):

        emotion_logits, features = self.backbone(x)

        reversed_features = GradientReversalLayer.apply(features, alpha)
        domain_logits = self.domain_classifier(reversed_features)

        return emotion_logits, domain_logits, features
