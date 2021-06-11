import argparse
import logging
import os
import sys
from datetime import datetime
import csv

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, CarvanaDataset
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import hiddenlayer as hl

# stamp
now = datetime.now()
timestamp = str(now.year).zfill(4) + str(now.month).zfill(2) + str(now.day).zfill(2) + str(now.hour).zfill(2) + str(now.minute).zfill(2)

# constants & variables
dirname = os.path.dirname(__file__)
outputDir = f"models/{timestamp}"
dir_img = os.path.join(dirname, 'data/x.npz')          # Images
dir_mask = os.path.join(dirname, 'data/ykid.npz')        # kidney .. ykid.npz | tumor .. ytum.npz
dir_checkpoint = os.path.join(dirname, outputDir, 'checkpoints/')
tensorboardPath = os.path.join(dirname, outputDir, 'runs/')

def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              save_cp=True,
              img_scale = 1):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)


    os.makedirs(tensorboardPath, exist_ok=True)
    writer = SummaryWriter(tensorboardPath)
    global_step = 0

    fileDir = os.path.join(dirname, outputDir, 'parameters')
    file = open(fileDir, "a")
    note = ('Starting training with Histograms Equalization:')
    L1 = [f'''{note}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {img_scale}
    '''] 
    file.writelines(L1)
    file.close()

    logging.info(f'''{note}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9) 
    # optimizer = optim.Adadelta(net.parameters(), lr=1.0, rho=0.98, eps=1e-06, weight_decay=0)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-07, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    file = open(fileDir, "a")
    L2 = [f'Optimizer: {optimizer.defaults}'] 
    file.writelines(L2)
    file.close()

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # summary(net, (1, 512, 512))
    # # Build HiddenLayer graph
    # hl_graph = hl.build_graph(net, torch.zeros([1, 1, 512, 512]).to(device))
    # # Use a different color theme
    # hl_graph.theme = hl.graph.THEMES["blue"].copy()  # Two options: basic and blue
    # hl_graph.save(path=os.path.join(dirname, outputDir) , format="png")

    csvPath = os.path.join(dirname, outputDir, 'loss.csv')
    csv_to_write = open(csvPath, 'w', newline='')
    csvwriter = csv.writer(csv_to_write, delimiter=' ')
        
    for epoch in range(epochs):
       
        
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:

            for batch in train_loader:
                content = {
                    'train_loss' : [],
                    'train_score' : [],
                    'test_loss' : [],
                    'test_score' : []
                }
                imgs = batch['image'].unsqueeze(1)
                true_masks = batch['mask'].unsqueeze(1)
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                content['train_loss'].append(loss.item())

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)
                        content['test_score'].append(val_score)
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

                for loss, score in zip(content['train_loss'], content['test_score']):
                    # csv_to_write.write(str(epoch) + ',' + str(loss) + ',' + str(score))
                    csvwriter.writerow(str(epoch + 1) + ' ' + str(loss) + ' ' + str(score))

            
        #writer.add_graph(net, imgs)
        writer.close()

        if save_cp:
            try:
                # create output dir
                os.makedirs(dir_checkpoint, exist_ok=True)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    csv_to_write.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=4,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') # cuda:0 for 1st GPU, cuda:1 for 2nd GPU
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
