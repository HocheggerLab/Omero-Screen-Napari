import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ROIBasedMobileNetV3Model(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ROIBasedMobileNetV3Model, self).__init__()

        # Pretrained MobileNetV3 model
        self.roi_model = models.mobilenet_v3_large(pretrained=True)
        self.roi_model.features[0][0] = nn.Conv2d(num_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust input channels
        self.roi_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Downsample output to a fixed size
        num_features_roi = self.roi_model.classifier[0].in_features
        self.roi_model.classifier = nn.Identity()  # Remove the final layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_features_roi, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, roi):
        # Feature extraction from the ROI
        roi_features = self.roi_model(roi)
        roi_features = roi_features.view(roi_features.size(0), -1)  # Flattening

        # Classification through fully connected layers
        x = torch.relu(self.fc1(roi_features))
        x = self.fc2(x)
        return x
    
class ROIBasedEfficientNetB0Model(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ROIBasedEfficientNetB0Model, self).__init__()

        # Pretrained EfficientNet-B0 model
        self.roi_model = models.efficientnet_b0(pretrained=True)
        self.roi_model.features[0][0] = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust input channels
        self.roi_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Downsample output to a fixed size
        num_features_roi = self.roi_model.classifier[1].in_features
        self.roi_model.classifier = nn.Identity()  # Remove the final layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_features_roi, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, roi):
        # Feature extraction from the ROI
        roi_features = self.roi_model(roi)
        roi_features = roi_features.view(roi_features.size(0), -1)  # Flattening

        # Classification through fully connected layers
        x = torch.relu(self.fc1(roi_features))
        x = self.fc2(x)
        return x
    
class ROIBasedShuffleNetV2Model(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ROIBasedShuffleNetV2Model, self).__init__()

        # Pretrained ShuffleNetV2 model
        self.roi_model = models.shufflenet_v2_x1_0(pretrained=True)
        self.roi_model.conv1[0] = nn.Conv2d(num_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)  # Adjust input channels
        self.roi_model.fc = nn.Identity()  # Remove the final layer

        num_features_roi = 1024  # Output features for ShuffleNetV2

        # Fully connected layers
        self.fc1 = nn.Linear(num_features_roi, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, roi):
        # Feature extraction from the ROI
        roi_features = self.roi_model(roi)
        roi_features = roi_features.view(roi_features.size(0), -1)  # Flattening

        # Classification through fully connected layers
        x = torch.relu(self.fc1(roi_features))
        x = self.fc2(x)
        return x

class ROIBasedDenseNetModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ROIBasedDenseNetModel, self).__init__()
        self.roi_model = models.densenet121(pretrained=True)
        self.roi_model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.roi_model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_features_roi = self.roi_model.classifier.in_features
        self.roi_model.classifier = nn.Identity()

        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, roi):
        roi_features = self.roi_model.features(roi)
        roi_features = torch.flatten(roi_features, 1)
        x = torch.relu(self.fc1(roi_features))
        x = self.fc2(x)
        return x
    
class ROIBasedResNeXtModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(ROIBasedResNeXtModel, self).__init__()

        # Pretrained ResNeXt model
        self.roi_model = models.resnext50_32x4d(pretrained=True)
        self.roi_model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Set the number of input channels to 2
        self.roi_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Downsample output to a fixed size
        num_features_roi = self.roi_model.fc.in_features
        self.roi_model.fc = nn.Identity()  # Remove the final layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, roi):
        # Feature extraction from the ROI
        roi_features = self.roi_model(roi)
        roi_features = roi_features.view(roi_features.size(0), -1)  # Flattening

        # Classification through fully connected layers
        x = torch.relu(self.fc1(roi_features))
        x = self.fc2(x)
        return x
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss