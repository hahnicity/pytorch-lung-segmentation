import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

import lung_segmentation.importAndProcess as iap
from models import model
from models.unet_models import unet11, unet16

parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('-m', '--model', choices=['unet11', 'unet16'])
parser.add_argument('-o', '--out-dir', default='images/')
parser.add_argument('-r', '--resume-from', help='resume from a specific savepoint', required=True)
args = parser.parse_args()

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

if args.model == 'unet11':
    model = unet11(out_filters=3).cuda()
elif args.model == 'unet16':
    model = unet16(out_filters=3).cuda()

dataset = iap.LungTest(
    args.img_path,
    Compose([Resize((224, 224)),ToTensor(),normalize]),
    convert_to='RGB',
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(args.resume_from))

with torch.no_grad():
    for i,sample in enumerate(dataloader):
        img = torch.autograd.Variable(sample['image']).cuda()
        bse = model(img)
        import IPython; IPython.embed()
        plt.imshow(bse)
        plt.show()
