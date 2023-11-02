
# SyntGAN

Script for training dcGAN to generate synthetic data.


| model   | input image size |
|---------|--------------|
| small   | 64           |
| medium  | 128          |
| big     | 256          |
| medium  | 512          |

## Train

```
usage: train.py [-h] --dataset DATASET [--workers WORKERS]
                [--modelName MODELNAME] [--modelSize MODELSIZE]
                [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nc NC]
                [--nz NZ] [--ngf NGF] [--ndf NDF] [--niter NITER] [--lr LR]
                [--beta1 BETA1] [--cuda] [--clearml]
                [--projectName PROJECTNAME] [--taskName TASKNAME]
                [--outputUri] [--ngpu NGPU] [--netG NETG] [--netD NETD]
                [--outf OUTF] [--manualSeed MANUALSEED] [--mps]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     path to dataset dir or name dataset in clearml mode
  --workers WORKERS     number of data loading workers
  --modelName MODELNAME
                        model name
  --modelSize MODELSIZE
                        choose model
  --batchSize BATCHSIZE
                        input batch size
  --imageSize IMAGESIZE
                        the height / width of the input image to network
  --nc NC               num channels
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --clearml             enables clearml mode
  --projectName PROJECTNAME
                        project name for clearml
  --taskName TASKNAME   task name for clearml
  --outputUri           save on clearml
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTF           folder to output images and model checkpoints
  --manualSeed MANUALSEED
                        manual seed
  --mps                 enables macOS GPU training

```
