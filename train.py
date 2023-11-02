from __future__ import print_function

import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from clearml import Dataset
from clearml import Task
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='path to dataset dir or name dataset in clearml mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--modelName', type=str, help='model name', default="")
parser.add_argument('--modelSize', type=str, default="medium", help='choose model')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=1, help='num channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--clearml', action='store_true', default=False, help='enables clearml mode')
parser.add_argument('--projectName', type=str, help='project name for clearml', default="")
parser.add_argument('--taskName', type=str, help='task name for clearml', default="")
parser.add_argument('--outputUri', action='store_true', default=False, help='save on clearml')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')

opt = parser.parse_args()
print(opt)

writer = SummaryWriter('runs')
task = None
dataset = None

if opt.clearml:
    task = Task.init(project_name=opt.projectName, task_name=opt.taskName, output_uri=opt.outputUri)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.modelSize == "large":
    from model_large import Generator, Discriminator, weights_init
elif opt.modelSize == "big":
    from model_big import Generator, Discriminator, weights_init
elif opt.modelSize == "medium":
    from model_medium import Generator, Discriminator, weights_init
else:
    from model_small import Generator, Discriminator, weights_init

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if torch.backends.mps.is_available() and not opt.mps:
    print("WARNING: You have mps device, to enable macOS GPU run with --mps")
dataset_path = ""

if opt.clearml:
    clearml_dataset = Dataset.get(
        dataset_name=opt.dataset,
    )
    dataset_path = clearml_dataset.get_local_copy()
else:
    dataset_path = opt.dataset

nc = int(opt.nc)

if nc == 3:
    norm = (0.5, 0.5, 0.5)
    dataset = dset.ImageFolder(root=dataset_path,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(norm, norm),
                               ]))
elif nc == 1:
    norm = (0.5)
    dataset = dset.ImageFolder(root=dataset_path,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.Grayscale(1),
                                   transforms.ToTensor(),
                                   transforms.Normalize(norm, norm),
                               ]))
else:
    raise ValueError("Number channel can be 1 or 3")

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
use_mps = opt.mps and torch.backends.mps.is_available()
if opt.cuda:
    device = torch.device("cuda:0")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

# custom weights initialization called on netG and netD
netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(ngpu, nc, ndf).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dictet(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

if opt.dry_run:
    opt.niter = 1

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        writer.add_scalar('Losses/Loss_D', errD.item(), epoch)
        writer.add_scalar('Losses/Loss_G', errG.item(), epoch)
        writer.add_scalar('Train/D(x)', D_x, epoch)
        writer.add_scalar('Train/D(G(z))_z1', D_G_z1, epoch)
        writer.add_scalar('Train/D(G(z))_z2', D_G_z2, epoch)

        if i % 100 == 0:
            vutils.save_image(real_cpu,
                              'real_samples.png',
                              normalize=True)

            fake = netG(fixed_noise)

            vutils.save_image(fake.detach(),
                              'fake_samples_epoch_%03d.png' % epoch,
                              normalize=True)
            if opt.clearml:
                task.upload_artifact("Real samples", "real_samples.png")
                task.upload_artifact("Fake samples epoch %03d" % epoch, "fake_samples_epoch_%03d.png" % epoch)

        if opt.dry_run:
            break
    # do checkpointing
    torch.save(netG.state_dict(), '%s/%s_netG_epoch_%d.pth' % (opt.outf, opt.modelName, epoch))
    torch.save(netD.state_dict(), '%s/%s_netD_epoch_%d.pth' % (opt.outf, opt.modelName, epoch))
