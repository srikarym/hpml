import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from resnet import *
from arguments import parse_args
from utils import get_optimizer, get_transforms


args = parse_args()

device = f'cuda:{args.gpu_id[0]}'
args.device = torch.device(device)

print(args)
print(torch.cuda.get_device_name(0))

transform_train, transform_test = get_transforms()

trainset = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

net = ResNet18().to(args.device)

if len(args.gpu_id) > 1:
	ids = args.gpu_id.split(',')
	ids = list(map(int,ids))
	net = torch.nn.DataParallel(net, device_ids = ids)


optimizer = get_optimizer(net)


def train(epoch):

	n_batches = len(trainloader)
	print(f'\nEpoch: {epoch}, num mini batches: {n_batches}')

	preprocess_start = time.perf_counter()

	net.train()
	train_loss = 0
	correct = 0
	total = 0

	data_loader_time = 0.0
	epoch_minibatch_time = 0.0

	data_movement_time = 0.0
	compute_time = 0.0


	metrics_time = 0.0

	iter_loader = iter(trainloader)

	for batch_idx in range(n_batches):
		load_start = time.perf_counter()

		inputs, targets = next(iter_loader)

		load_end = time.perf_counter()

		inputs, targets = inputs.to(args.device), targets.to(args.device)
		movement_end = time.perf_counter()

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

		data_loader_time += (load_end - load_start)
		epoch_minibatch_time += (batch_end - load_start)

		compute_time += batch_end - load_end

		data_movement_time += movement_end - load_end
		metrics_end = time.perf_counter()
		metrics_time += metrics_end - batch_end

	epoch_end = time.perf_counter()

	epoch_time =  epoch_end - preprocess_start - metrics_time

	train_loss /= n_batches

	if epoch != 0:
		print(f'\nAggregates: \
				\n \t DataLoader Time: {data_loader_time:.4f} \
				\n \t Mini-batch Time: {epoch_minibatch_time:.4f} \
				\n \t Compute time: {compute_time:.4f} \
				\n \t \t Movement time: {data_movement_time:.4f}\
				\n \t Training Epoch Time: {epoch_time:.4f}\n \
				\n Averages: \
				\n \t Training Loss: {train_loss:.4f}\
				\n \t Training accuracy: {(100* correct/total):.4f}')


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
	if args.print_logs:
		print('\n Test set Averages: Loss: %.3f | Precision@1: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == '__main__':
	
	# os.system("grep 'model name' /proc/cpuinfo |head -1")
	# os.system('nvidia-smi')

	for epoch in range(args.epochs):
		train(epoch)
		test()
