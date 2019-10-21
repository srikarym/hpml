import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim import SGD, Adagrad, Adadelta, Adam
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from resnet import ResNet18



def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--data-dir', type=str,
						metavar='', help='Data directory root',
						default='./data')

	parser.add_argument('--optimizer', type=str,
						metavar='', help='Optimizer', default='sgd')

	parser.add_argument('--num-workers', type=int,
						metavar='', help='Number of data loader workers',
						default=2)

	parser.add_argument('--batch-size', type=int,
						metavar='', help='Batch Size', default=128)
	parser.add_argument('--epochs', type=int,
						metavar='', help='Number of epochs', default=5)

	parser.add_argument('--lr', type=float,
						metavar='', help='Learning rate', default=0.1)

	parser.add_argument('--momentum', type=float,
						metavar='', help='Momentum', default=0.9)

	parser.add_argument('--weight-decay', type = float,
						metavar='',help = 'Weight decay', default=5e-4)

	parser.add_argument('--cuda', type=int, default=0,
						help='Flag 1/0 to enable/disable CUDA')

	args = parser.parse_args()

	# Make file paths absolute
	args.data_dir = os.path.abspath(args.data_dir)

	args.cuda = bool(args.cuda)
	args.cuda = args.cuda and torch.cuda.is_available()
	device = 'cuda' if args.cuda else 'cpu'
	args.device = torch.device(device)

	print(args.device)

	return args


def precision(k, output, target, device, mean=True):
	topk_labels = output.detach().topk(k)[1]
	rows = torch.arange(topk_labels.shape[0]).unsqueeze(-1).repeat(1, topk_labels.shape[1]).to(device)
	indices = torch.cat((rows.reshape(-1, 1), topk_labels.reshape(-1, 1)), dim=1)

	labels_predicted = torch.zeros_like(target)
	labels_predicted[indices[:, 0], indices[:, 1]] = 1

	precision_k = torch.sum(target * labels_predicted, dim=1).float() / float(k)

	if mean:
		return precision_k.mean()

	return precision_k.sum()


def main():
	args = parse_args()

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

	testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

	net = ResNet18().to(args.device)

	if args.optimizer == 'sgd':
		optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer == 'nesterov':
		optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
						weight_decay=args.weight_decay,nesterov=True)
	elif args.optimizer == 'adagrad':
		optimizer = Adagrad(net.parameters())
	elif args.optimizer == 'adadelta':
		optimizer = Adadelta(net.parameters())
	elif args.optimizer == 'adam':
		optimizer = Adam(net.parameters())
	else:
		raise Exception('Invalid optimizer specified.')

	n_batches = len(trainloader)

	for epoch in range(args.epochs):
		# Statistics Tracking Variables
		epoch_loss = 0.0
		epoch_precision1 = 0.0

		epoch_loader_time = 0.0
		epoch_minibatch_time = 0.0

		iter_loader = iter(trainloader)
		net.train()

		epoch_start = time.perf_counter()

		for mini_batch in range(n_batches):
			load_start = time.perf_counter()

			image, labels = next(iter_loader)

			load_end = time.perf_counter()

			image = image.to(args.device)
			labels = labels.to(args.device).reshape(image.shape[0], -1)

			out = net(image)
			loss = F.cross_entropy(out, labels.float())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			batch_end = time.perf_counter()

			precision_1 = precision(1, out, labels, args.device, mean=False)

			epoch_loss += loss.detach().cpu().item()
			epoch_precision1 += precision_1.cpu().item()

			epoch_loader_time += (load_end - load_start)
			epoch_minibatch_time += (batch_end - load_start)

		epoch_time = time.perf_counter() - epoch_start

		epoch_precision1 /= len(trainset)
		epoch_loss /= n_batches

		report = """Epoch {}
	Aggregates:
		DataLoader Time: {:10.4f}
		Mini-batch Time: {:10.4f}
		Epoch Time: {:10.4f}
	Averages:
		Loss: {:10.4f}
		Precision@1: {:10.4f}
	""".format(epoch + 1, epoch_loader_time, epoch_minibatch_time, epoch_time,
				epoch_loss, epoch_precision1)

		print(report)


if __name__ == '__main__':
	main()