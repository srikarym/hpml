import os
import time
import torch
from torch.utils.data import Dataset
from torch.optim import SGD, Adagrad, Adadelta, Adam
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from resnet import *
from arguments import parse_args

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])





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
	res = correct_k.mul_(100.0)
	return res


def train(epoch):
	global avg_dataloading_time
	n_batches = len(trainloader)
		# Statistics Tracking Variables
	epoch_loss = 0.0

	epoch_loader_time = 0.0
	epoch_minibatch_time = 0.0

	iter_loader = iter(trainloader)
	net.train()
	epoch_start = time.perf_counter()
	print(f'Epoch {epoch}')
	precision_1 = 0.0

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

		loss_scalar = loss.detach().cpu().item()
		epoch_loss += loss_scalar

		precision_1 = precision(1, out, labels).cpu().item() / image.shape[0]

		epoch_loader_time += (load_end - load_start)
		epoch_minibatch_time += (batch_end - load_start)

		print(f'\t Minibatch {mini_batch+1} / {n_batches}, loss: {loss_scalar:.2f}, \
			\t Avg Accumulated loss: {epoch_loss / (mini_batch+1):.2f}, \
			\t Precision: {precision_1:.2f}')

	epoch_time = time.perf_counter() - epoch_start

	epoch_loss /= n_batches

	print(f'\nAggregates: \
			\n \t DataLoader Time: {epoch_loader_time:.4f} \
			\n \t Mini-batch Time:{epoch_minibatch_time:.4f} \
			\n \t Training Epoch Time:{epoch_time:.4f}\n \
			\n Averages: \
			\n \t Training Loss: {epoch_loss:.4f}')

	avg_dataloading_time += epoch_loader_time

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

	report = f"\t Testing precision@1: {epoch_precision1:.4f} \n".format(epoch_precision1)
	print(report)

if __name__ == '__main__':
	avg_dataloading_time = 0.0
	
	os.system("grep 'model name' /proc/cpuinfo |head -1")
	os.system('nvidia-smi')

	for epoch in range(args.epochs):
		train(epoch)
		test()

	print(f'Average dataloader time is {avg_dataloading_time/(args.epochs):.2f}')