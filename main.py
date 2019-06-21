import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import csv
import copy
import PIL

parser = argparse.ArgumentParser()
#parser.add_argument('--train_dir', default='~/manga/train')
#parser.add_argument('--val_dir', default='~/manga/valid')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10	, type=int)
parser.add_argument('--num_epochs2', default=150, type=int)
#parser.add_argument('--use_gpu', action='store_true')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main(args):
  dtype = torch.cuda.FloatTensor
  train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20, resample= PIL.Image.BILINEAR),
    T.ToTensor(),            
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  
  train_dset = ImageFolder('train', transform=train_transform)
  train_loader = DataLoader(train_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True)
  val_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  val_dset = ImageFolder('valid', transform=val_transform)
  val_loader = DataLoader(val_dset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

  model = torchvision.models.resnet50(pretrained=True)

  num_classes = len(train_dset.classes)
  model.fc = nn.Sequential(nn.Dropout(p = 0.5), nn.Linear(model.fc.in_features, num_classes))
  model.type(dtype)
  loss_fn = nn.CrossEntropyLoss().type(dtype)
  for param in model.parameters():
    param.requires_grad = False
  for param in model.fc.parameters():
    param.requires_grad = True

  optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

  for epoch in range(args.num_epochs1):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
    run_epoch(model, loss_fn, train_loader, optimizer, dtype)

    train_acc = check_accuracy(model, train_loader, dtype)
    val_acc = check_accuracy(model, val_loader, dtype)
    print('Train accuracy: ', train_acc)
    print('Val accuracy: ', val_acc)
    print()

  for param in model.parameters():
    param.requires_grad = True
  

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
  max_acc = 0

  for epoch in range(args.num_epochs2):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
    run_epoch(model, loss_fn, train_loader, optimizer, dtype)

    train_acc = check_accuracy(model, train_loader, dtype)
    print('Train accuracy: ', train_acc)
    if (epoch%3 == 0):
      val_acc = check_accuracy(model, val_loader, dtype)
      print('Val accuracy: ', val_acc)
      if (val_acc > max_acc):
        max_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "resnet50.pth")
        print('Saving model')    

    print()

  print('loading a model')
  """
  loda the best performing model
  model.load_state_dict(torch.load("resnet50.pth"))
  write some test function to test your model
  test(model, dtype)
  """



def run_epoch(model, loss_fn, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  model.train()
  for x, y in loader:
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).long())
    scores = model(x_var)
    loss = loss_fn(scores, y_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

@ torch.no_grad()
def check_accuracy(model, loader, dtype):
  model.eval()
  num_correct, num_samples = 0, 0
  for x, y in loader:
    x_var = x.type(dtype)
    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)
    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  acc = float(num_correct) / num_samples
  return acc

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
