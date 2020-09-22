"""
Code modified from the following publication:
    
Shi, Z., Chen, C., Xiong, Z., Liu, D., & Wu, F. (2018). 
Hscnn+: Advanced cnn-based hyperspectral recovery from rgb images. 
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 939-947).
"""

import torch
import torch.nn as nn
from math import sqrt
import numpy as np
from torch.autograd import Variable

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=True)

class conv_relu_res_block(nn.Module):
    def __init__(self):
        super(conv_relu_res_block, self).__init__()
        self.conv1 = conv3x3(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(256, 256)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.add(out,residual) 
        return out
    
class conv_relu_res_block_scale01(nn.Module):
    def __init__(self):
        super(conv_relu_res_block_scale01, self).__init__()
        self.conv1 = conv3x3(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(256, 256)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = torch.mul(out,0.1) 
        out = torch.add(out,residual) 
        return out
    
class conv_relu_res_relu_block(nn.Module):
    def __init__(self):
        super(conv_relu_res_relu_block, self).__init__()
        self.conv1 = conv3x3(256, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(256, 256)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = torch.add(out,residual) 
        out = self.relu2(out)
        return out

class conv_relu_res_relu_block_scale01(nn.Module):
    def __init__(self):
        super(conv_relu_res_relu_block_scale01, self).__init__()
        self.conv1 = conv3x3(256, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(256, 256)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = torch.mul(out,0.1) 
        out = torch.add(out,residual) 
        out = self.relu2(out)
        return out
    
class resblock(nn.Module):
    def __init__(self, block, block_num, input_channel, output_channel):
        super(resblock, self).__init__()

        self.in_channels = input_channel
        self.out_channels = output_channel
        self.input_conv = conv3x3(self.in_channels, out_channels=256)  
        self.conv_seq = self.make_layer(block, block_num)
        self.conv = conv3x3(256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.output_conv = conv3x3(in_channels=256,  out_channels=self.out_channels)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,sqrt(2./n))# the devide  2./n  carefully  
                
    def make_layer(self,block,num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block()) # there is a () 
        return nn.Sequential(*layers)   
    
    def forward(self, x):
       
        out = self.input_conv(x)
        residual = out
        out = self.conv_seq(out)
        out = self.conv(out)
        out = torch.add(out,residual)  
        out = self.relu(out)
        out = self.output_conv(out)
        return out

def get_reconstruction(input, num_split, dimension, model):
    """As the limited GPU memory split the input."""
    
    input_split = torch.split(input,  int(input.shape[3]/num_split), dim=dimension)
    output_split = []
    for i in range(num_split):
        var_input = Variable(input_split[i].cuda(),volatile=True)
        var_output = model(var_input)
        output_split.append(var_output.data)
        if i == 0:
            output = output_split[i]
        else:
            output = torch.cat((output, output_split[i]), dim=dimension)
    
    return output

def reconstruction(rgb, model):
    """Output the final reconstructed hyperspectral images."""
    img_res = get_reconstruction(torch.from_numpy(rgb).float(),87, 3, model)
    img_res = img_res.cpu().numpy()*4095
    img_res = np.transpose(np.squeeze(img_res))
    
    return img_res