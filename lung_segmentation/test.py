import argparse
import os

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import lung_segmentation.importAndProcess as iap
from models import model
from models.unet_models import unet11, unet16


parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('-m', '--model', choices=['unet11', 'unet16', 'resnet'], default='unet16')
parser.add_argument('-r', '--resume-from', help='resume from a specific savepoint', required=True)
parser.add_argument('-t', '--input-type', choices=['dicom', 'png'], default='dicom')
parser.add_argument('--non-montgomery', action='store_true', help='toggle this flag if you are working on a non-montgomery dataset')
parser.add_argument('--no-normalize', action='store_true')
args = parser.parse_args()

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

if args.model == 'resnet':
    model = model.segmentNetwork().cuda()
    resize_dim = (400, 400)
    convert_to = 'L'
elif args.model == 'unet11':
    model = unet11(out_filters=3).cuda()
    resize_dim = (224, 224)
    convert_to = 'RGB'
elif args.model == 'unet16':
    model = unet16(out_filters=3).cuda()
    resize_dim = (224, 224)
    convert_to = 'RGB'

if args.no_normalize:
    transforms = Compose([Resize(resize_dim),ToTensor()])
else:
    transforms = Compose([Resize(resize_dim),ToTensor(),normalize])
convert_to = 'RGB'

if args.input_type == 'dicom':
    dataset = iap.DicomSegment(args.img_path, transforms, convert_to)
elif args.input_type == 'png' and args.non_montgomery:
    dataset = iap.LungTest(args.img_path, transforms, convert_to)
elif args.input_type == 'png':
    dataset = iap.lungSegmentDataset(
        os.path.join(args.img_path, "CXR_png"),
        os.path.join(args.img_path, "ManualMask/leftMask/"),
        os.path.join(args.img_path, "ManualMask/rightMask/"),
        imagetransform=transforms,
        labeltransform=Compose([Resize((224, 224)),ToTensor()]),
        convert_to='RGB',
    )
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.resume_from))
show = iap.visualize(dataset)

with torch.no_grad():
    for i, sample in enumerate(dataloader):
        img = torch.autograd.Variable(sample['image']).cuda()
        mask = model(img)
        if not args.non_montgomery:
            show.ImageWithGround(i,True,True,save=True)

        show.ImageWithMask(i, sample['filename'][0], mask.squeeze().cpu().numpy(), True, True, save=True)
