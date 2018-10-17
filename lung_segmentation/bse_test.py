import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.utils import save_image

import lung_segmentation.importAndProcess as iap
from models import model
from models.unet_models import unet11, unet16

parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('-m', '--model', choices=['unet11', 'unet16'])
parser.add_argument('-o', '--out-dir', default='image/')
parser.add_argument('-r', '--resume-from', help='resume from a specific savepoint', required=True)
parser.add_argument('--no-normalize', action='store_true')
args = parser.parse_args()

if args.no_normalize:
    transforms = Compose([Resize((224, 224)),ToTensor()])
else:
    norm_means = [0.485, 0.456, 0.406]
    norm_stds = [0.229, 0.224, 0.225]
    normalize = Normalize(norm_means, norm_stds)
    transforms = Compose([Resize((224, 224)),ToTensor(),normalize])

if args.model == 'unet11':
    model = unet11(out_filters=3).cuda()
elif args.model == 'unet16':
    model = unet16(out_filters=3).cuda()

dataset = iap.LungTest(args.img_path, transforms, convert_to='RGB')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.resume_from))

with torch.no_grad():
    for i,sample in enumerate(dataloader):
        img = torch.autograd.Variable(sample['image']).cuda()
        bse = model(img).squeeze()
        # de-normalize so that it doesn't look unintelligible
        if not args.no_normalize:
            for i in range(3):
                bse[i] = bse[i] * norm_stds[i]
                bse[i] += norm_means[i]
        save_image(bse, os.path.join(os.path.dirname(__file__), 'image/', sample['filename'][0]))
