import torch.nn as nn
from torch import nn
import torch

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU()
    )

def upsample_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Upsample(scale_factor= 2), #upsample multiply by 2 on hight and width, but the num of color channle remains the same
        nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace = True)
    )

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        print('Initialize ConvBlock...')

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        print('Initialize UNet...')

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = conv_block(in_ch=3, out_ch=16)
        self.conv2 = conv_block(in_ch=16, out_ch=32)
        self.conv3 = conv_block(in_ch=32, out_ch=64)
        self.conv4 = conv_block(in_ch=64, out_ch=128)

        self.upsample1 = upsample_block(in_ch=128, out_ch=64)
        self.up_conv1 = conv_block(in_ch=128, out_ch=64)

        self.upsample2 = upsample_block(in_ch=64, out_ch=32)
        self.up_conv2 = conv_block(in_ch=64, out_ch=32)

        self.upsample3 = upsample_block(in_ch=32, out_ch=16)
        self.up_conv3 = conv_block(in_ch=32, out_ch=16)

        self.conv_1x1 = nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0) #number of output classes

    def forward(self, input_img_4d_tensor):
        conv1_in = input_img_4d_tensor
        conv1_out = self.conv1(conv1_in)
        #print('conv1_in: ', conv1_in.size())
        #print('conv1_out: ', conv1_out.size())

        conv2_in = self.maxpool1(conv1_out)
        conv2_out = self.conv2(conv2_in)
        #print('conv2_in: ', conv2_in.size())
        #print('conv2_out: ', conv2_out.size())

        conv3_in = self.maxpool2(conv2_out)
        conv3_out = self.conv3(conv3_in)
        #print('conv3_in: ', conv3_in.size())
        #print('conv3_out: ', conv3_out.size())

        conv4_in = self.maxpool3(conv3_out)
        conv4_out = self.conv4(conv4_in)
        #print('conv4_in: ', conv4_in.size())
        #print('conv4_out: ', conv4_out.size())

        deconv1_in = conv4_out
        deconv1_out = self.upsample1(deconv1_in)
        #print('deconv1_in: ', deconv1_in.size())
        #print('deconv1_out: ', deconv1_out.size())

        cat_c3_d1 = torch.cat((conv3_out, deconv1_out), dim=1)
        deconv2_in = self.up_conv1(cat_c3_d1)
        deconv2_out = self.upsample2(deconv2_in)
        #print('deconv2_in: ', deconv2_in.size())
        #print('deconv2_out: ', deconv2_out.size())

        cat_c2_d2 = torch.cat((conv2_out, deconv2_out), dim=1)
        deconv3_in = self.up_conv2(cat_c2_d2)
        deconv3_out = self.upsample3(deconv3_in)
        #print('deconv3_in: ', deconv3_in.size())
        #print('deconv3_out: ', deconv3_out.size())

        cat_c1_d3 = torch.cat((conv1_out, deconv3_out), dim=1)
        deconv4_in = self.up_conv3(cat_c1_d3)
        deconv4_out = self.conv_1x1(deconv4_in)
        #print('deconv4_in: ', deconv4_in.size())
        #print('deconv4_out: ', deconv4_out.size())

        return deconv4_out
