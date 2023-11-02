import argparse
import os

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torchvision.utils as vutils
from PIL import Image
from clearml import Task, InputModel, Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, required=True, help='path to weights or name weights in clearml mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--clearml', action='store_true', default=False, help='enables clearml mode')
parser.add_argument('--projectName', type=str, help='project name for clearml', default="")
parser.add_argument('--taskName', type=str, help='task name for clearml', default="")
parser.add_argument('--modelSize', type=str, default="medium", help='choose model')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--threshold', type=int, default=5000, help="threshold")
parser.add_argument('--amount', type=int, default=100, help="generate amount image")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--datasetName', default='dataset', help='folder to output generated dataset')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--nc', type=int, default=1, help='num channels')

opt = parser.parse_args()
print(opt)
BATCH = 8
dataset = None

if opt.clearml:
    task = Task.init(project_name=opt.projectName, task_name=opt.taskName, task_type=Task.TaskTypes.inference)

try:
    os.makedirs(opt.datasetName)
except OSError:
    pass

if opt.modelSize == "large":
    from model_large import Generator, weights_init
elif opt.modelSize == "big":
    from model_big import Generator, weights_init
elif opt.modelSize == "medium":
    from model_medium import Generator, weights_init
else:
    from model_small import Generator, weights_init

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if torch.backends.mps.is_available() and not opt.mps:
    print("WARNING: You have mps device, to enable macOS GPU run with --mps")

if opt.cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

ngpu = int(opt.ngpu)
ngf = int(opt.ngf)
nc = int(opt.nc)
nz = int(opt.nz)

if opt.clearml:
    dataset = Dataset.create(
        dataset_name=str(opt.datasetName),
        dataset_project='SyntGAN',
        dataset_tags=['synt']
    )

netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.apply(weights_init)

print(netG)

model_weights = ""

if opt.clearml:
    model = InputModel(name=opt.weights, project="SyntGAN")
    print(model.name)
    model_weights = model.get_weights(raise_on_error=True)
else:
    model_weights = opt.weights

print("Using weights %s" % model_weights)

netG.load_state_dict(torch.load(model_weights))

generated = 0
ims = int(opt.imageSize)
while generated < opt.amount:
    noise = torch.randn(BATCH, nz, 1, 1, device=device)
    fake = netG(noise)

    vutils.save_image(fake.detach(),
                      'fake.png',
                      normalize=True
                      )

    image = Image.open("fake.png")
    for i in range(BATCH):
        left = 2 + (ims * i) + (2 * i)
        top = 2
        right = left + ims
        bottom = top + ims

        cropped_image = image.crop((left, top, right, bottom))

        pixels = sum([sum(i) for i in list(cropped_image.getdata())]) // 1000

        if pixels < opt.threshold:
            continue

        cropped_image.save('%s/fake%i.png' % (opt.datasetName, generated))
        generated += 1
        print("Generated %s / %s images" % (generated, opt.amount))

if opt.clearml:
    dataset.add_files(str(opt.datasetName))
    dataset.finalize(auto_upload=True)
