import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()

        self.num_classes = num_classes

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=False, num_classes=self.num_classes
        )

    def forward(self, x):
        x = self.model(x)

        return x


encoder_ch = [64, 128, 256, 512]

# from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65939


class CSE(nn.Module):
    def __init__(self, in_ch, r):
        super(CSE, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x)

        x = torch.mul(input_x, x)

        return x


class SSE(nn.Module):
    def __init__(self, in_ch):
        super(SSE, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = torch.mul(input_x, x)

        return x


class SCSE(nn.Module):
    def __init__(self, in_ch, r):
        super(SCSE, self).__init__()

        self.cSE = CSE(in_ch, r)
        self.sSE = SSE(in_ch)

    def forward(self, x):
        cSE = self.cSE(x)
        sSE = self.sSE(x)

        x = torch.add(cSE, sSE)

        return x


class Decoder(nn.Module):
    def __init__(self, in_ch, ch, out_ch, r):
        super(Decoder, self).__init__()

        self.cSE = CSE(out_ch, r)

        self.conv1 = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x, encoder=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)

        if encoder is not None:
            x = torch.cat([x, encoder], 1)

        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)

        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)

        x = self.cSE(x)

        return x


class ResUNet34(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ResUNet34, self).__init__()

        self.resnet = torchvision.models.resnet34(pretrained=True)

        self.encoder_1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )

        self.encoder_2 = nn.Sequential(
            self.resnet.layer1
        )

        self.encoder_3 = nn.Sequential(
            self.resnet.layer2
        )

        self.encoder_4 = nn.Sequential(
            self.resnet.layer3
        )

        self.encoder_5 = nn.Sequential(
            self.resnet.layer4
        )

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.decoder_1 = Decoder(256+512, 512, 64, 16)
        self.decoder_2 = Decoder(64+256, 256, 64, 16)
        self.decoder_3 = Decoder(64+128, 128, 64, 16)
        self.decoder_4 = Decoder(64+64, 64, 64, 16)
        self.decoder_5 = Decoder(64, 32, 64, 16)

        self.logit = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1)
        )

    def forward(self, x):
        encoder_1 = self.encoder_1(x)
        encoder_2 = self.encoder_2(encoder_1)
        encoder_3 = self.encoder_3(encoder_2)
        encoder_4 = self.encoder_4(encoder_3)
        encoder_5 = self.encoder_5(encoder_4)

        center = self.center(encoder_5)

        decoder_1 = self.decoder_1(center, encoder_5)
        decoder_2 = self.decoder_2(decoder_1, encoder_4)
        decoder_3 = self.decoder_3(decoder_2, encoder_3)
        decoder_4 = self.decoder_4(decoder_3, encoder_2)
        decoder_5 = self.decoder_5(decoder_4)

        logit = self.logit(decoder_5)

        return logit
