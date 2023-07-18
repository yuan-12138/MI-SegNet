import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

    
class UpConv2d_s2(nn.Module):
    def __init__(self,in_channels,features):
        super(UpConv2d_s2, self).__init__()
        self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
class Res_block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Res_layer(nn.Module):
    def __init__(self, inplanes, planes, blocks, stride=1):
        super(Res_layer, self).__init__()
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(Res_block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(Res_block(planes, planes))
        
        self.res_layer = nn.Sequential(*layers)
    
    def forward(self,x):
        return self.res_layer(x)
    
class Mine_Conv(nn.Module):
    def __init__(self,in_channels_x,in_channels_y,inter_channels):
        super(Mine_Conv, self).__init__()
        
        self.ma_et=None
        
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels_x, max(in_channels_x//2,inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_x//2,inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            nn.Conv2d(max(in_channels_x//2,inter_channels), max(in_channels_x//4,inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_x//4,inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            nn.Conv2d(max(in_channels_x//4,inter_channels), inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.AvgPool2d(2,1),
            Flatten()
        )
        self.conv_y = nn.Sequential(
            nn.Conv2d(in_channels_y, max(in_channels_y//2,inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_y//2,inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            nn.Conv2d(max(in_channels_y//2,inter_channels), max(in_channels_y//4,inter_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(in_channels_y//4,inter_channels)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            nn.Conv2d(max(in_channels_y//4,inter_channels), inter_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.AvgPool2d(2,1),
            Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(inter_channels, inter_channels),
            nn.Linear(inter_channels, 1)
        )
        
    def forward(self, x,y):
        
        return self.fc(self.conv_x(x)+self.conv_y(y))

    
class Seg_encoder_LM(nn.Module):
    def __init__(self,in_channels=1,init_features=32,num_blocks=2):
        super(Seg_encoder_LM, self).__init__()
        
        features = init_features
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Res_layer(features,2*features,blocks=num_blocks,stride=1),
            Res_layer(2*features,4*features,blocks=num_blocks,stride=2),
            Res_layer(4*features,8*features,blocks=num_blocks,stride=2),
            Res_layer(8*features,16*features,blocks=num_blocks,stride=2)
        )

    def forward(self, input):
        
        
        return self.encoder(input)
    
class Seg_decoder_LM(nn.Module):
    def __init__(self,in_channels=1,init_features=32,num_blocks=2):
        super(Seg_decoder_LM, self).__init__()
        
        features = init_features
        
        self.decoder = nn.Sequential(
            
            UpConv2d_s2(16*features,8*features),
            Res_layer(8*features,8*features,blocks=num_blocks,stride=1),
            UpConv2d_s2(8*features,4*features),
            Res_layer(4*features,4*features,blocks=num_blocks,stride=1),
            UpConv2d_s2(4*features,2*features),
            Res_layer(2*features,2*features,blocks=num_blocks,stride=1),
            
            UpConv2d_s2(2*features,features),
            Res_layer(features,features,blocks=num_blocks,stride=1),
            UpConv2d_s2(features,features),
            Res_layer(features,features,blocks=num_blocks,stride=1),
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        
    def forward(self, input):
        
        return self.decoder(input)
    


class Recon_encoder_LM(nn.Module):
    def __init__(self,in_channels=1,init_features=32):
        super(Recon_encoder_LM, self).__init__()
        
        features = init_features
        
        self.encoder = nn.Sequential(
 			nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
 			nn.Conv2d(features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(2*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(4*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(8*features, 16*features, kernel_size=3, padding=1),
		)

    def forward(self, input):
        
        z = self.encoder(input)
        
        return z
    
class Recon_decoder_LM(nn.Module):
    def __init__(self,in_channels_a,in_channels_d,out_channels=1,init_features=32):
        super(Recon_decoder_LM, self).__init__()
        
        features = init_features
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels_a+in_channels_d, 8*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8*features, 4*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(4*features, 2*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2*features, features, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Sigmoid()
		)

        
    def forward(self, z_a, z_d):
        
        Recon_result = self.decoder(torch.cat([z_a,z_d],dim=1))
        
        return Recon_result