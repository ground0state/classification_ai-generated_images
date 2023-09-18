#%%
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork



# class FPNPANClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(FPNPANClassifier, self).__init__()
#         self.backbone = models.resnet50(pretrained=True)
#         self.fpn_sizes = [self.backbone.layer1[-1].conv3.out_channels,
#                           self.backbone.layer2[-1].conv3.out_channels,
#                           self.backbone.layer3[-1].conv3.out_channels,
#                           self.backbone.layer4[-1].conv3.out_channels]

#         self.fpn = FeaturePyramidNetwork(in_channels_list=self.fpn_sizes, out_channels=256)

#         # PAN layers (bottom-up path)
#         self.pan_ups = nn.ModuleList([
#             nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1) for _ in range(len(self.fpn_sizes)-1)])

#         # Classifier
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         # Forward through backbone
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x1 = self.backbone.layer1(x)
#         x2 = self.backbone.layer2(x1)
#         x3 = self.backbone.layer3(x2)
#         x4 = self.backbone.layer4(x3)

#         # Forward through FPN
#         features = {f'0': x1, f'1': x2, f'2': x3, f'3': x4}
#         pyramid_features = self.fpn(features)

#         # Bottom-up path aggregation
#         for i in range(0, len(self.fpn_sizes)-1):
#             pyramid_features[f'{i+1}'] += self.pan_ups[i](pyramid_features[f'{i}'])

#         # Global Average Pooling and Classifier
#         x = nn.AdaptiveAvgPool2d(1)(pyramid_features['2']).view(x.size(0), -1)
#         x = self.fc(x)
#         return x


class FPNPANClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FPNPANClassifier, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.fpn_sizes = [self.backbone.layer1[-1].conv3.out_channels,
                          self.backbone.layer2[-1].conv3.out_channels,
                          self.backbone.layer3[-1].conv3.out_channels,
                          self.backbone.layer4[-1].conv3.out_channels]

        self.fpn = FeaturePyramidNetwork(in_channels_list=self.fpn_sizes, out_channels=256)

        # PAN layers (bottom-up path)
        self.pan_ups = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1) for _ in range(len(self.fpn_sizes)-1)])

        # Classifier
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        # Forward through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        # Forward through FPN
        features = {f'0': x1, f'1': x2, f'2': x3, f'3': x4}
        pyramid_features = self.fpn(features)

        # Bottom-up path aggregation
        for i in range(0, len(self.fpn_sizes)-1):
            pyramid_features[f'{i+1}'] += self.pan_ups[i](pyramid_features[f'{i}'])

        # Global Average Pooling and Classifier
        xs = []
        for i in range(0, len(self.fpn_sizes)):
            xs.append(nn.AdaptiveAvgPool2d(1)(pyramid_features[f'{i}']).view(batch_size, -1))
        xs = torch.cat(xs, dim=-1)
        print(xs.shape)
        x = self.fc(xs)
        return x



# %%
model = FPNPANClassifier(2)
# %%
x = torch.rand(1, 3, 256, 256)

# %%
model(x)
# %%
