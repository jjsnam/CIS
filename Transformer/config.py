# config.py
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs', type=int, default=10, help='训练的轮数')
# parser.add_argument('--lr', type=float, default=0.001, help='学习率')
# parser.add_argument('--train_path', type=str, default='/root/Project/datasets/200kMDID/Train', help='训练数据集路径')
# parser.add_argument('--val_path', type=str, default='/root/Project/datasets/200kMDID/Val', help='评估数据集路径')
# parser.add_argument('--model_path', type=str, default='/root/Project/weights/CNN/200kMDID', help='训练权重存储路径')
# parser.add_argument('--dataset_name', type=str, default='200kMDID', help='dataset_name')

# args = parser.parse_args()

class Config:
    image_size = 224
    batch_size = 32
    # num_epochs = args.epochs
    # lr = args.lr
    model_name = 'vit_base_patch16_224'
    num_classes = 2
    pretrained = True
    device = 'cuda'
    checkpoint_path = '/root/Project/Transformer/weights/vit_base_patch16_224.pth'
    # train_path = args.train_path
    # val_path = args.val_path
    # model_path = args.model_path
    # dataset_name = args.dataset_name