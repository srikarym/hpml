import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from resnet import *
from arguments import parse_args
from utils import *



args = parse_args()
transform_train, transform_test = get_transforms()

trainset = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

net = ResNet18().to(args.device)

optimizer = get_optimizer(net)


def train(epoch):

	print('\nEpoch: %d' % epoch)

	global avg_dataloading_time
	n_batches = len(trainloader)

	preprocess_start = time.perf_counter()

	net.train()
	train_loss = 0
	correct = 0
	total = 0

	epoch_loader_time = 0.0
	epoch_minibatch_time = 0.0
	print_time = 0.0

	metrics_time = 0.0

	iter_loader = iter(trainloader)

	preprocess_time = time.perf_counter() - preprocess_start

	for batch_idx in range(n_batches):
		load_start = time.perf_counter()

		inputs, targets = next(iter_loader)

		load_end = time.perf_counter()

		inputs, targets = inputs.to(args.device), targets.to(args.device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = F.cross_entropy(outputs, targets)
		loss.backward()
		optimizer.step()

		batch_end = time.perf_counter()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		
		metrics_end = time.perf_counter()

		epoch_loader_time += (load_end - load_start)
		epoch_minibatch_time += (batch_end - load_start)

		metrics_time += metrics_end - batch_end

		print_start = time.perf_counter()
		print('Loss: %.3f | Precision@1: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
		print_end = time.perf_counter()

		print_time += print_end - print_start

	epoch_end = time.perf_counter()

	epoch_time =  epoch_end - preprocess_start + print_time

	train_loss /= n_batches

	print(f'\nAggregates: \
			\n \t DataLoader Time: {epoch_loader_time:.4f} \
			\n \t Mini-batch Time: {epoch_minibatch_time:.4f} \
			\n \t Training Epoch Time: {epoch_time:.4f}\n \
			\n Averages: \
			\n \t Training Loss: {train_loss:.4f}')

	avg_dataloading_time += epoch_loader_time

	

def test():
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.to(args.device), targets.to(args.device)
			outputs = net(inputs)
			loss = F.cross_entropy(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	print('\n Test set Averages: Loss: %.3f | Precision@1: %.3f%% (%d/%d)'
		% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == '__main__':
	avg_dataloading_time = 0.0
	
	os.system("grep 'model name' /proc/cpuinfo |head -1")
	os.system('nvidia-smi')

	for epoch in range(args.epochs):
		train(epoch)
		test()

	print(f'Average dataloader time is {avg_dataloading_time/(args.epochs):.2f}')