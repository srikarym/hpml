import argparse
import os
import torch

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

	parser.add_argument('--batch-size','-bs', type=int,
						metavar='', help='Batch Size', default=128)

	parser.add_argument('--epochs', type=int,
						metavar='', help='Number of epochs', default=5)

	parser.add_argument('--lr', type=float,
						metavar='', help='Learning rate', default=0.1)

	parser.add_argument('--momentum', type=float,
						metavar='', help='Momentum', default=0.9)

	parser.add_argument('--weight-decay', type = float,
						metavar='',help = 'Weight decay', default=5e-4)

	parser.add_argument('--gpu-id', type=str, default='0',
						help='GPU device id')

	parser.add_argument('--batch-norm', type=int, default=1,
						help='Flag 1/0 to enable/disable Batch-norm')

	parser.add_argument('--print-logs',type=bool, default=False,
						help = 'Flag to enable/disable printing training loss / acc')

	args = parser.parse_args()

	# Make file paths absolute
	args.data_dir = os.path.abspath(args.data_dir)


	return args