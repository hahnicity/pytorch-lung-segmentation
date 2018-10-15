"""
bse_elimination
~~~~~~~~~~~~~~~

Stands for bone shadow elimination. We will try to create a learning
regressor to remove bone shadow from lung images
"""
import argparse

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from lung_sementation.importAndProcess import JSRTBSE
from models.unet_models import unet11, unet16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsrt-path', default='/fastdata/jsrt/All247images/')
    parser.add_argument('--bse-path', default='/fastdata/jsrt/bone_shadow_eliminated/')
    parser.add_argument('-r', '--resume-from')
    parser.add_argument('-m', '--model', choices=['unet11', 'unet16'], default='unet16')
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=int, default=40)
    args = parser.parse_args()

    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transforms = Compose([Resize((224, 224)), ToTensor(), normalize])
    dataset = JSRTBSE(args.jsrt_path, args.bse_path, transforms, 'RGB')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = unet16(pretrained=True, out_filters=3).cuda()
    if args.resume_from:
        model.load_state_dict(torch.load(args.resume_from))
        init_epochs = int(os.path.basename(args.resume_from).replace('{}_bse_'.format(args.model), ''))
    else:
        init_epochs = 0
    model = torch.nn.DataParallel(model)
    optimizer = Adam(model.parameters(), lr=0.0002)
    criterion = MSELoss()

    with torch.enable_grad():
        for ep_offset in range(args.epochs-init_epochs):
            eps = ep_offset + init_epochs

            for input, target in dataloader:
                img = torch.autograd.Variable(input).cuda()
                ground = torch.autograd.Variable(target).cuda()
                # the output mask seems to have 1 channel, which is different than
                # the output mask for Sam's resnet which has 3 channels
                mask = model(img)
                optimizer.zero_grad()
                loss = criterion(mask, ground)
                loss.backward()
                optimizer.step()
                print(loss.cpu().detach().numpy().item())

            if((eps+1) % 20 == 0):
                torch.save(model.state_dict(), "{}_bse_{}".format(args.model, eps+1))


if __name__ == "__main__":
    main()
