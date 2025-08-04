import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='训练的轮数')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--train_path', type=str, default='/root/Project/datasets/200kMDID/Train', help='训练数据集路径')
parser.add_argument('--val_path', type=str, default='/root/Project/datasets/200kMDID/Val', help='评估数据集路径')
parser.add_argument('--model_path', type=str, default='/root/Project/weights/CNN/200kMDID', help='训练权重存储路径')
parser.add_argument('--dataset_name', type=str, default='200kMDID', help='dataset_name')

args = parser.parse_args()

# Paths
TRAIN_DIR = args.train_path
VAL_DIR = args.val_path
# TEST_DIR = '/root/Project/datasets/Celeb_V2/Test'
CHECKPOINT_PATH = args.model_path + '/' + args.dataset_name + '_CNN+Transformer_'

# Hyperparameters
BATCH_SIZE = 32
LR = args.lr
EPOCHS = args.epochs
SEED = 42

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')