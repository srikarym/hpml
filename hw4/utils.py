import torchvision.transforms as transforms
from torch.optim import SGD, Adagrad, Adadelta, Adam
from arguments import parse_args
import time
args = parse_args()

def get_transforms():
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
	return transform_train, transform_test


def get_optimizer(net):
	if args.optimizer == 'sgd':
		optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
	elif args.optimizer == 'nesterov':
		optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
						weight_decay=args.weight_decay,nesterov=True)
	elif args.optimizer == 'adagrad':
		optimizer = Adagrad(net.parameters(), weight_decay=args.weight_decay)
	elif args.optimizer == 'adadelta':
		optimizer = Adadelta(net.parameters(), weight_decay=args.weight_decay)
	elif args.optimizer == 'adam':
		optimizer = Adam(net.parameters(), weight_decay=args.weight_decay)
	else:
		raise Exception('Invalid optimizer specified.')

	return optimizer

# class Timer:
#     """With block timer.
#     with Timer() as t:
#             foo = blah()
#     print('Request took %.03f sec.' % t.interval)
#     """

#     def __enter__(self):
#         self.start = time.perf_counter()
#         return self

#     def __exit__(self, *args):
#         self.end = time.perf_counter()
#         self.interval = self.end - self.start
