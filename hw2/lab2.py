import argparse
import os
import time
import torch
from torch.utils.data import Dataset
from torch.optim import SGD, Adagrad, Adadelta, Adam
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


	return args


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


def precision(k, output, target):

	batch_size = target.size(0)
	_, pred = output.topk(k,1,True,True)
	pred = pred.t()
	correct = pred.eq(target.view(1,-1).expand_as(pred))

	correct_k = correct[:k].view(-1).float().sum(0,keepdim = True)
	res = correct_k.mul_(100.0 / batch_size)
	return res

def train(epoch):
	
	n_batches = len(trainloader)
		# Statistics Tracking Variables
	epoch_loss = 0.0

	epoch_loader_time = 0.0
	epoch_minibatch_time = 0.0

	iter_loader = iter(trainloader)
	net.train()
	epoch_start = time.perf_counter()
	print(f'Epoch {epoch}')
	for mini_batch in range(n_batches):
		load_start = time.perf_counter()

		image, labels = next(iter_loader)

		load_end = time.perf_counter()

		image = image.to(args.device)
		labels = labels.to(args.device)

		out = net(image)
		loss = F.cross_entropy(out, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_end = time.perf_counter()


		epoch_loss += loss.detach().cpu().item()

		epoch_loader_time += (load_end - load_start)
		epoch_minibatch_time += (batch_end - load_start)

		print(f'Minibatch {mini_batch} / {n_batches}\n \t loss: {loss.detach().cpu().item()}\
			\n \t Accumulated loss: {epoch_loss}')

	epoch_time = time.perf_counter() - epoch_start

	epoch_loss /= n_batches

	print(f'Aggregates: \n \t DataLoader Time: {epoch_loader_time} \n \t \
		Mini-batch Time:{epoch_minibatch_time} \n \t Training Epoch Time:{epoch_time}\n \
		Averages: \n \t Training Loss: {epoch_loss}')


def test():
	n_batches = len(testloader)
		# Statistics Tracking Variables
	epoch_loss = 0.0
	epoch_precision1 = 0.0

	iter_loader = iter(testloader)
	net.eval()


	for mini_batch in range(n_batches):

		image, labels = next(iter_loader)


		image = image.to(args.device)
		labels = labels.to(args.device)

		out = net(image)
		precision_1 = precision(1, out, labels)

		epoch_precision1 += precision_1.cpu().item()

	epoch_precision1 /= len(testset)

	report = "\tTesting precision@1: {:10.4f}".format(epoch_precision1)
	print(report)

if __name__ == '__main__':
	
	for epoch in range(args.epochs):
		train(epoch)
		test()