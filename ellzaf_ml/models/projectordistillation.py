import torch
import torch.nn as nn

class ProjectorDistillationStudent(nn.Module):
    def __init__(self, encoder, out):
        super().__init__()
        self.encoder = encoder

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.num_features, out),
            nn.BatchNorm1d(out),
        )

    def forward(self, x):
        feat, preds = self.encoder(x)
        feat = self.projector(feat)
        return feat, preds

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}
    
class ProjectorDistillationTeacher(nn.Module):
    def __init__(self, encoder, out):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.projector = nn.Sequential(
            nn.BatchNorm1d(out),
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.projector(feat)
        return feat

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}