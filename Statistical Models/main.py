# main.py
from src.train import train_loop
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='训练的轮数')
# parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--train_path', type=str, default='/root/Project/datasets/200kMDID/Train', help='训练数据集路径')
parser.add_argument('--val_path', type=str, default='/root/Project/datasets/200kMDID/Val', help='评估数据集路径')
parser.add_argument('--model_path', type=str, default='/root/Project/weights/CNN/200kMDID', help='训练权重存储路径')
parser.add_argument('--dataset_name', type=str, default='200kMDID', help='数据集名称')

args = parser.parse_args()

if __name__ == "__main__":
    train_loop(args.train_path + "/real", args.train_path + "/fake",
               args.val_path + "/real", args.val_path + "/fake",
               args.model_path, args.epochs, args.dataset_name)