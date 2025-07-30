# config.py
class Config:
    image_size = 224
    batch_size = 32
    num_epochs = 10
    lr = 3e-4
    model_name = 'vit_base_patch16_224'
    num_classes = 2
    pretrained = True
    device = 'cuda'
    checkpoint_path = '/root/Project/Transformer/weights/vit_base_patch16_224.pth'