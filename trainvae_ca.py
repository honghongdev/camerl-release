import argparse
from os.path import join, exists
from os import mkdir
import os

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision.utils import save_image

from models.vae import VAE
from utils.misc import save_checkpoint
from utils.misc import LSIZE, RED_SIZE
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from data.loaders import _RolloutCADataset
from torch.utils.tensorboard import SummaryWriter



def visualize_reconstruction(model, dataloader, vae_dir, epoch=None, device=None):
    model.eval()
    with torch.no_grad():
        data_iter = iter(dataloader)
        data, target = next(data_iter)
        data   = data.to(device)
        target = target.to(device)
        recon, _, _ = model(data)

        # Three rows: input depth (top), reconstructed CA (middle), target CA (bottom)
        comparison = torch.cat([data.cpu(), recon.cpu(), target.cpu()])

        recon_dir = os.path.join(vae_dir, 'reconstructions')
        os.makedirs(recon_dir, exist_ok=True)

        save_path = os.path.join(recon_dir, f'reconstruction_epoch_{epoch}.png')
        save_image(comparison, save_path, nrow=data.size(0))
        print(f'[vis] input/recon/CA-target saved to: {save_path}')


def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss: MSE reconstruction (against CA target) + KL divergence. """
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    loss = BCE + KLD
    return loss, BCE, KLD


def train(epoch, writer=None):
    model.train()
    dataset_train.load_next_buffer()
    train_loss = train_bce_sum = train_kld_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data   = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, target, mu, logvar)
        loss.backward()

        train_loss    += loss.item()
        train_bce_sum += bce.item()
        train_kld_sum += kld.item()

        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))

    if writer:
        writer.add_scalar('Loss/train', train_loss / len(train_loader.dataset), epoch)


def test():
    model.eval()
    dataset_test.load_next_buffer()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data   = data.to(device)
            target = target.to(device)
            recon_batch, mu, logvar = model(data)
            t_loss, _, _ = loss_function(recon_batch, target, mu, logvar)
            test_loss += t_loss.item()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


parser = argparse.ArgumentParser(description='VAE Trainer (CA target)')
parser.add_argument('--batch-size', type=int, default=32,   metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs',     type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--vae-dir',    type=str, default='./saved/vae/camerl-1',
                    help='Output directory for VAE weights and logs')
parser.add_argument('--dataset',    type=str, default='./saved/dataset-ca',
                    help='CA dataset directory produced by ca_proc/generate.py')

args = parser.parse_args()
vae_dir         = args.vae_dir
CA_DATASET_PATH = args.dataset
cuda = torch.cuda.is_available()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")

dataset_train = _RolloutCADataset(CA_DATASET_PATH, train=True)
dataset_test  = _RolloutCADataset(CA_DATASET_PATH, train=False)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_loader  = torch.utils.data.DataLoader(
    dataset_test,  batch_size=args.batch_size, shuffle=True, num_workers=1)

model      = VAE(1, LSIZE).to(device)
enc_params = sum(p.numel() for p in model.encoder.parameters())
optimizer  = optim.Adam(model.parameters())
scheduler  = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-5)
earlystopping = EarlyStopping('min', patience=50)

if not exists(vae_dir):
    os.makedirs(vae_dir)

writer  = SummaryWriter(log_dir=vae_dir)
cur_best = None

for epoch in range(1, args.epochs + 1):
    train(epoch, writer)
    test_loss = test()
    writer.add_scalar('Loss/test', test_loss, epoch)

    scheduler.step(test_loss)
    cur_lr = optimizer.param_groups[0]['lr']
    print(f"[Epoch {epoch}] LR: {cur_lr:.6g}")
    writer.add_scalar('LR/epoch', cur_lr, epoch)

    earlystopping.step(test_loss)

    best_filename = join(vae_dir, 'best.tar')
    filename      = join(vae_dir, 'checkpoint.tar')
    is_best       = not cur_best or test_loss < cur_best
    if is_best:
        cur_best = test_loss

    save_checkpoint({
        'epoch':        epoch,
        'state_dict':   model.state_dict(),
        'precision':    test_loss,
        'optimizer':    optimizer.state_dict(),
        'scheduler':    scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)

    if epoch % 10 == 0:
        visualize_reconstruction(model, test_loader, vae_dir, epoch=epoch, device=device)

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break

writer.close()
