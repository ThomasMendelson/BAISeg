import torch
import torch.nn as nn



# class Dconv3DBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Dconv3DBlock, self).__init__()
#         self.deconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.bn1 = nn.BatchNorm3d(num_features=out_channels)
#         self.conv = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm3d(num_features=out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         res = self.relu(self.bn1(self.deconv(x)))
#         res = self.relu(self.bn2(self.conv(res)))
#         return res
#
# class DconvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DconvBlock, self).__init__()
#         self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
#         self.bn1 = nn.BatchNorm2d(num_features=out_channels)
#         self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(num_features=out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         res = self.relu(self.bn1(self.deconv(x)))
#         res = self.relu(self.bn2(self.conv(res)))
#         return res

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv2D, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),   # nn.ReLU(inplace=True) nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.GELU()
        )
    def forward(self, x):
        return self.conv(x)

class Up2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, final_layer=False):
        super(Up2DBlock, self).__init__()
        self.final_layer = final_layer
        self.deconv = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_channels, mid_channels)

    def forward(self, x1, x2):
        if x2 is not None:
            diffH = x2.size()[2] - x1.size()[2]
            diffW = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
            x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.final_layer:
            return x
        return self.deconv(x)

class Up2DBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, final_layer=False, scale_factor=2):
        super(Up2DBlock2, self).__init__()
        self.final_layer = final_layer
        # self.deconv = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = DoubleConv2D(in_channels, out_channels, mid_channels)

    def forward(self, x1, x2):
        if x2 is not None:
            diffH = x2.size()[2] - x1.size()[2]
            diffW = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
            x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.final_layer:
            return x
        return self.upsample(x)

class Dense2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super(Dense2DBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in_channels = in_channels + i * out_channels
            self.layers.append(nn.Sequential(
                nn.Conv2d(layer_in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = torch.cat(features, dim=1)
            out = layer(x)
            features.append(out)

        return out

class DenseUp2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, num_layers=3, final_layer=False):
        super(DenseUp2DBlock, self).__init__()
        self.final_layer = final_layer
        self.deconv = nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=2, stride=2)
        self.dense_block = Dense2DBlock(in_channels, mid_channels, num_layers)


    def forward(self, x1, x2):
        if x2 is not None:
            diffH = x2.size()[2] - x1.size()[2]
            diffW = x2.size()[3] - x1.size()[3]
            x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2])
            x1 = torch.cat([x2, x1], dim=1)
        x = self.dense_block(x1)
        if self.final_layer:
            return x
        return self.deconv(x)

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv3D, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.GELU(), #  nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.GELU(), #  nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Up3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, final_layer=False):
        super(Up3DBlock, self).__init__()
        self.final_layer = final_layer
        self.deconv = nn.ConvTranspose3d(mid_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, mid_channels)

    def forward(self, x1, x2):
        if x2 is not None:
            diffD = x2.size()[2] - x1.size()[2]
            diffH = x2.size()[3] - x1.size()[3]
            diffW = x2.size()[4] - x1.size()[4]
            x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2,
                                        diffD - diffD // 2])
            x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        if self.final_layer:
            return x
        return self.deconv(x)

class Dense3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3):
        super(Dense3DBlock, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_in_channels = in_channels + i * out_channels
            self.layers.append(nn.Sequential(
                nn.Conv3d(layer_in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ))
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = torch.cat(features, dim=1)
            out = layer(x)
            features.append(out)

        return out

class DenseUp3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=3, final_layer=False):
        super(DenseUp3DBlock, self).__init__()
        self.final_layer = final_layer
        self.dense_block = Dense3DBlock(in_channels, out_channels, num_layers)
        self.deconv = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        if x2 is not None:
            diffD = x2.size()[2] - x1.size()[2]
            diffH = x2.size()[3] - x1.size()[3]
            diffW = x2.size()[4] - x1.size()[4]
            x1 = nn.functional.pad(x1, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2,
                                        diffD - diffD // 2])
            x1 = torch.cat([x2, x1], dim=1)
        x = self.dense_block(x1)
        if self.final_layer:
            return x
        return self.deconv(x)






