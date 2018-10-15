import torch.nn as nn
import torch.nn.functional as F
import math
import torch

class resBlock(nn.Module):
    expansion = 4

    def __init__(self,inplanes, planes, kernelsize, padding, stride = 1):
        super(resBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=kernelsize,
        padding=padding, bias=False)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=kernelsize,
        padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = False

        if inplanes is not planes:
            self.downsample = True

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.conv1(residual)

        out += residual

        out = self.bn3(out)
        out = self.relu(out)

        return out

class segmentNetwork(nn.Module):
    def __init__(self):
        super(segmentNetwork,self).__init__()

        self.res1 = resBlock(1,8,7,3)
        self.res2 = resBlock(8,8,3,1)
        self.res3 = resBlock(8,16,3,1)
        self.res4 = resBlock(16,32,3,1)
        self.res5 = resBlock(32,64,3,1)

        self.resa = resBlock(64,64,1,0)
        self.resb = resBlock(64,64,3,1)
        self.resc = resBlock(64,64,1,0)
        self.resd = resBlock(64,64,3,1)
        self.rese = resBlock(64,64,1,0)

        self.res6 = resBlock(64,3,1,0)

        # self.multires =

        self.avgpool = nn.AvgPool2d(2,stride=2)

        self.deconv = nn.ConvTranspose2d(3,3,16,16)

    def forward(self,x):

        x = self.res1(x)
        #import IPython; IPython.embed()
        x = self.res2(x)
        x = self.avgpool(x)
        #import IPython; IPython.embed()
        x = self.res3(x)
        x = self.avgpool(x)
        #import IPython; IPython.embed()
        x = self.res4(x)
        x = self.avgpool(x)
        #import IPython; IPython.embed()
        x = self.res5(x)
        x = self.avgpool(x)
        #import IPython; IPython.embed()

        x = self.rese(self.resd(self.resc(self.resb(self.resa(x)))))

        x = self.res6(x)
        #import IPython; IPython.embed()
        x = self.deconv(x)

        return x


class criticNetwork(nn.Module):
    def __init__(self):
        super(criticNetwork,self).__init__()

    def forward(self):
        pass
