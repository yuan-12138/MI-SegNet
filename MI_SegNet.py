import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class Unflatten(nn.Module):
	def __init__(self, channel, height, width):
		super(Unflatten, self).__init__()
		self.channel = channel
		self.height = height
		self.width = width

	def forward(self, input):
		return input.view(input.size(0), self.channel, self.height, self.width)
     
class DoubleConv2d(nn.Module):
    def __init__(self,in_channels,features):
        super(DoubleConv2d, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                # nn.GroupNorm(num_groups=int(features/2), num_channels=features),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)
    
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
    
class Mine(nn.Module):
    def __init__(self,input_dim):
        super(Mine, self).__init__()
        
        self.ma_et=None
        
        self.fc1_x = nn.Linear(input_dim, 512)
        self.fc1_y = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,1)
        
    def forward(self, x,y):
        h1 = F.elu(self.fc1_x(x)+self.fc1_y(y))
        h2 = F.elu(self.fc2(h1))
        h3 = F.elu(self.fc3(h2))
        return h3
    
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

    
class DNL_block(nn.Module):

    def __init__(self, in_dim):
        super(DNL_block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.mask_conv = nn.Conv2d(in_dim, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B * C * H * W)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is H * W)
        """
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B * N * C
        proj_query -= proj_query.mean(1).unsqueeze(1)
        proj_key = self.key_conv(x).view(B, -1, H * W)  # B * C * N
        proj_key -= proj_key.mean(2).unsqueeze(2)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # B * N * N
        attention = attention.permute(0, 2, 1)
        proj_value = self.value_conv(x).view(B, -1, H * W)  # B * C * N
        proj_value = self.relu(proj_value)
        
        proj_mask = self.mask_conv(x).view(B, -1, H * W)  # B * 1 * N
        mask = self.softmax(proj_mask)
        mask = mask.permute(0, 2, 1)
        
        attention = attention+mask
        
        tissue = torch.bmm(proj_value, attention)
        tissue = tissue.view(B, C, H, W)

        out = x
        return out, tissue

class Seg_encoder_DNL(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=32):
        super(Seg_encoder_DNL, self).__init__()
        
        features = init_features
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        
        
        self.tad_1 = DNL_block(features)
        self.res_layer_1 = Res_layer(features,2*features,blocks=2,stride=1)
        
        self.tad_2 = DNL_block(2*features)
        self.res_layer_2 = Res_layer(2*features,4*features,blocks=2,stride=2)
        
        self.tad_3 = DNL_block(4*features)
        self.res_layer_3 = Res_layer(4*features,8*features,blocks=2,stride=2)
        
        self.tad_4 = DNL_block(8*features)
        self.res_layer_4 = Res_layer(8*features,16*features,blocks=2,stride=2)
        
        self.Flatten = Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(features*16*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
            nn.Tanh()
        )

    def forward(self, input):
        
        x1, tissue1=self.tad_1(self.conv_1(input))
        
        x2, tissue2 = self.tad_2(self.res_layer_1(x1))
        
        x3, tissue3 = self.tad_3(self.res_layer_2(x2))
        
        x4, tissue4 = self.tad_4(self.res_layer_3(x3))
        
        bottleneck = self.res_layer_4(x4)

        z = self.fc(self.Flatten(bottleneck))
        
        
        return z, tissue1, tissue2, tissue3, tissue4
    
class Seg_decoder_DNL(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=32):
        super(Seg_decoder_DNL, self).__init__()
        
        features = init_features
        
        self.Unflatten = Unflatten(features*16, 8, 8)
        
        self.upconv_4 = UpConv2d_s2(16*features,8*features)
        self.decoder_4 = DoubleConv2d(16*features,8*features)
        
        self.upconv_3 = UpConv2d_s2(8*features,4*features)
        self.decoder_3 = DoubleConv2d(8*features,4*features)
        
        self.upconv_2 = UpConv2d_s2(4*features,2*features)
        self.decoder_2 = DoubleConv2d(4*features,2*features)
        
        
        self.upconv_1 = nn.Conv2d(2*features, features, kernel_size=1)
        self.decoder_1 = DoubleConv2d(2*features,features)
        
        self.decoder_0 = nn.Sequential(
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=1),
            )
        
        self.fc = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features*16*8*8),
            nn.ReLU()
        )
        
    def forward(self, input):
        
        z, enc_1, enc_2, enc_3, enc_4 = input
        
        bottleneck_1 = self.Unflatten(self.fc(z))
        
        dec_4 = torch.cat([enc_4,self.upconv_4(bottleneck_1)],dim=1)
        dec_4 = self.decoder_4(dec_4)
        
        dec_3 = torch.cat([enc_3,self.upconv_3(dec_4)],dim=1)
        dec_3 = self.decoder_3(dec_3)
        
        dec_2 = torch.cat([enc_2,self.upconv_2(dec_3)],dim=1)
        dec_2 = self.decoder_2(dec_2)
        
        dec_1 = torch.cat([enc_1,self.upconv_1(dec_2)],dim=1)
        dec_1 = self.decoder_1(dec_1)
        
        Seg_result = torch.sigmoid(self.decoder_0(dec_1))
        
        return Seg_result

    
class Recon_encoder_fusion(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=32):
        super(Recon_encoder_fusion, self).__init__()
        
        features = init_features
        
        self.encoder = nn.Sequential(
 			nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
 			nn.Conv2d(features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2*features, 2*features, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(2*features, 2*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(2*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(4*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(8*features, 16*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features), num_channels=16*features),
            # nn.BatchNorm2d(16*features),
            nn.ReLU(inplace=True),

 			Flatten(),
            nn.Linear(16*features*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
            nn.Tanh()
		)

    def forward(self, input):
        
        z = self.encoder(input)
        
        return z
    
class Recon_decoder_fusion(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=32):
        super(Recon_decoder_fusion, self).__init__()
        
        features = init_features
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 16*features*8*8),
            nn.ReLU(),
            Unflatten(16*features, 8, 8),
            nn.ReLU(),
            # nn.ConvTranspose2d(16*features, 8*features, kernel_size=4, stride=4),
            nn.ConvTranspose2d(16*features, 8*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8*features, 4*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(4*features, 2*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2*features, features, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
		)

        
    def forward(self, z_a, z_d):
        
        Recon_result = self.decoder(z_a+z_d)
        
        return Recon_result
    



class Seg_encoder_N(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=32):
        super(Seg_encoder_N, self).__init__()
        
        features = init_features
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Res_layer(features,2*features,blocks=2,stride=1),
            Res_layer(2*features,4*features,blocks=2,stride=2),
            Res_layer(4*features,8*features,blocks=2,stride=2),
            Res_layer(8*features,16*features,blocks=2,stride=2),
            Flatten(),
            nn.Linear(features*16*8*8, 512),
            nn.ReLU(),
            nn.Linear(512, z_dim),
            nn.Tanh()
        )

    def forward(self, input):
        
        
        return self.encoder(input)
    
class Seg_decoder_N(nn.Module):
    def __init__(self,in_channels=1,z_dim=512,init_features=32):
        super(Seg_decoder_N, self).__init__()
        
        features = init_features
        
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, features*16*8*8),
            nn.ReLU(),
            
            Unflatten(features*16, 8, 8),
            
            UpConv2d_s2(16*features,8*features),
            DoubleConv2d(8*features,8*features),
            UpConv2d_s2(8*features,4*features),
            DoubleConv2d(4*features,4*features),
            UpConv2d_s2(4*features,2*features),
            DoubleConv2d(2*features,2*features),
            nn.Conv2d(2*features, features, kernel_size=1),
            DoubleConv2d(features,features),
            
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features, features, kernel_size=2, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        
    def forward(self, input):
        
        return self.decoder(input)
    
    
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
            # DoubleConv2d(8*features,8*features),
            Res_layer(8*features,8*features,blocks=num_blocks,stride=1),
            UpConv2d_s2(8*features,4*features),
            # DoubleConv2d(4*features,4*features),
            Res_layer(4*features,4*features,blocks=num_blocks,stride=1),
            UpConv2d_s2(4*features,2*features),
            # DoubleConv2d(2*features,2*features),
            Res_layer(2*features,2*features,blocks=num_blocks,stride=1),
            # nn.Conv2d(2*features, features, kernel_size=1),
            # DoubleConv2d(features,features),
            
            UpConv2d_s2(2*features,features),
            Res_layer(features,features,blocks=num_blocks,stride=1),
            # nn.BatchNorm2d(features),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(features, features, kernel_size=3, padding=1),
            # nn.BatchNorm2d(features),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(features, features, kernel_size=3, padding=1),
            # nn.BatchNorm2d(features),
            # nn.ReLU(inplace=True),
            UpConv2d_s2(features,features),
            Res_layer(features,features,blocks=num_blocks,stride=1),
            # nn.BatchNorm2d(features),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(features, features, kernel_size=3, padding=1),
            # nn.BatchNorm2d(features),
            # nn.ReLU(inplace=True),
            nn.Conv2d(features, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        
    def forward(self, input):
        
        return self.decoder(input)
    
class Seg_decoder_LM_(nn.Module):
    def __init__(self,in_channels=1,init_features=32):
        super(Seg_decoder_LM_, self).__init__()
        
        features = init_features
        
        self.decoder = nn.Sequential(
            
            UpConv2d_s2(16*features,8*features),
            DoubleConv2d(8*features,8*features),
            UpConv2d_s2(8*features,4*features),
            DoubleConv2d(4*features,4*features),
            UpConv2d_s2(4*features,2*features),
            DoubleConv2d(2*features,2*features),
            nn.Conv2d(2*features, features, kernel_size=1),
            DoubleConv2d(features,features),
            
            nn.ConvTranspose2d(features,features, kernel_size=2, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features,features, kernel_size=2, stride=2),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
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
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
 			nn.Conv2d(features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            # nn.Conv2d(2*features, 2*features, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(2*features, 2*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(2*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            
            nn.Conv2d(4*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
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
            # nn.ConvTranspose2d(16*features, 8*features, kernel_size=4, stride=4),
            nn.ConvTranspose2d(in_channels_a+in_channels_d, 8*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(8*features, 8*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/2), num_channels=8*features),
            # nn.BatchNorm2d(8*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8*features, 4*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*features, 4*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/4), num_channels=4*features),
            # nn.BatchNorm2d(4*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(4*features, 2*features, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*features, 2*features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/8), num_channels=2*features),
            # nn.BatchNorm2d(2*features),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(2*features, features, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=int(features/16), num_channels=features),
            # nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, out_channels, kernel_size=1),
            nn.Sigmoid()
		)

        
    def forward(self, z_a, z_d):
        
        Recon_result = self.decoder(torch.cat([z_a,z_d],dim=1))
        
        return Recon_result