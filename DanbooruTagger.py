import torch.nn as nn
import torchvision.models as models

class DanbooruTagger(nn.Module):
    def __init__(self, num_classes):
        super(DanbooruTagger, self).__init__()
        self.base_model = models.efficientnet_v2_m(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.base_model.classifier[-1].in_features, out_features=num_classes,bias=False),
            nn.Sigmoid()
        )
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True
    def forward(self, x):
        return self.base_model(x)