import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torch.utils.data
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, print_tensor_dict
from torch.backends import cudnn
from models import get_model
from torchvision.transforms import ColorJitter

cudnn.benchmark = True

# Configuration parser
parser = argparse.ArgumentParser(description='Generic Training Script for CIFAR-10')
# Model configuration
parser.add_argument('--model', default='wrn', type=str, help='Model architecture')
parser.add_argument('--depth', default=40, type=int, help='Depth for WRN')
parser.add_argument('--width', default=4, type=float, help='Width multiplier for WRN')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='Dataset name')

# Data configuration
parser.add_argument('--data_source', required=True, choices=['dpmepf', 'dpsda', 'privimage'],
                   help='Data source type')
parser.add_argument('--train_datapath', required=True, type=str, help='Path to training data')
parser.add_argument('--test_datapath', default='./dataset_cifar10', type=str, help='Path to test data')
parser.add_argument('--nthread', default=4, type=int, help='Number of data workers')

# Training configuration
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--epochs', default=200, type=int, help='Total training epochs')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay')
parser.add_argument('--lr_schedule', default='[60,120,160]', type=str, help='LR decay schedule')
parser.add_argument('--lr_decay', default=0.2, type=float, help='LR decay ratio')

# Experiment management
parser.add_argument('--save_dir', default='./logs', type=str, help='Root directory for saving results')
parser.add_argument('--exp_name', required=True, type=str, help='Experiment name')
parser.add_argument('--seed', default=1, type=int, help='Random seed')
parser.add_argument('--gpu', default='0', type=str, help='GPU IDs')

def create_dataset(args, is_train):
    """Create dataset based on data source type"""
    if args.data_source == 'dpmepf':
        return DPMEPFDataset(args.train_datapath, is_train)
    elif args.data_source == 'dpsda':
        return DPSDADataset(args.train_datapath, is_train)
    elif args.data_source == 'privimage':
        return PrivImageDataset(args.train_datapath, is_train)
    else:
        raise ValueError(f"Unknown data source: {args.data_source}")

class BaseDataset(Dataset):
    """Base dataset class with common transformations"""
    def __init__(self, data_path, is_train):
        self.is_train = is_train
        self.transform = self.build_transform(is_train)
        self.load_data(data_path)

    def build_transform(self, is_train):
        transforms = []
        if is_train:
            transforms.extend([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return T.Compose(transforms)

    def load_data(self, path):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.transform(self.samples[idx]), self.labels[idx]

class DPMEPFDataset(BaseDataset):
    """Dataset loader for DP-MEPF format"""
    def load_data(self, path):
        data = np.load(path)
        self.samples = data['x']
        self.labels = data['y'] if 'y' in data else np.zeros(len(self.samples))

class DPSDADataset(BaseDataset):
    """Dataset loader for DPSDA format"""
    def load_data(self, path):
        data = np.load(path)
        self.samples = data['samples']
        self.labels = np.repeat(np.arange(10), 5000)

class PrivImageDataset(datasets.ImageFolder):
    """Dataset loader for PrivImage format"""
    def __init__(self, data_path, is_train):
        transform = self.build_transform(is_train)
        super().__init__(root=data_path, transform=transform)

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    
    # Create model
    model = get_model(args.model, in_channel=3)
    if torch.cuda.is_available():
        model.cuda()
    
    # Create data loaders
    train_loader = DataLoader(
        create_dataset(args, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.nthread
    )
    test_loader = DataLoader(
        datasets.CIFAR10(args.test_datapath, train=False, download=True,
                        transform=BaseDataset.build_transform(False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.nthread
    )
    
    # Training setup
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    engine = TrainingEngine(model, optimizer, args)
    engine.run(train_loader, test_loader)

class TrainingEngine:
    """Generic training engine"""
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.epoch = 0
        
        # Setup metrics
        self.meter_loss = tnt.meter.AverageValueMeter()
        self.classacc = tnt.meter.ClassErrorMeter(accuracy=True)
        self.setup_save_dir()

    def setup_save_dir(self):
        self.save_path = os.path.join(self.args.save_dir, 
                                     f"{self.args.data_source}_{self.args.exp_name}")
        os.makedirs(self.save_path, exist_ok=True)

    def train_step(self, sample):
        inputs, targets = map(cast, (sample[0], sample[1]))
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss, outputs

    def run(self, train_loader, test_loader):
        engine = Engine()
        engine.hooks.update({
            'on_sample': self.on_sample,
            'on_forward': self.on_forward,
            'on_start_epoch': self.on_start_epoch,
            'on_end_epoch': self.on_end_epoch,
        })
        engine.train(self.train_step, train_loader, self.args.epochs, self.optimizer)

    # Engine hooks implementation
    def on_sample(self, state):
        state['sample'].append(state['train'])

    def on_forward(self, state):
        self.meter_loss.add(float(state['loss']))
        self.classacc.add(state['output'].data, state['sample'][1])

    def on_start_epoch(self, state):
        self.meter_loss.reset()
        self.classacc.reset()
        state['iterator'] = tqdm(state['iterator'], dynamic_ncols=True)

    def on_end_epoch(self, state):
        # Test and save model
        test_acc = self.evaluate(test_loader)
        self.save_checkpoint(state)
        print(f"Epoch {state['epoch']} - Test Acc: {test_acc:.2f}%")

    def evaluate(self, loader):
        self.model.eval()
        self.classacc.reset()
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = self.model(inputs.cuda())
                self.classacc.add(outputs.data, targets.cuda())
        return self.classacc.value()[0]

    def save_checkpoint(self, state):
        torch.save({
            'epoch': state['epoch'],
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.save_path, 'checkpoint.pth'))

if __name__ == '__main__':
    main()