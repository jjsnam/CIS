# model.py
# import timm
# import torch.nn as nn
# from config import Config

# def get_model():
#     model = timm.create_model(Config.model_name, pretrained=Config.pretrained, num_classes=Config.num_classes)
#     return model
# model.py
# import timm
# import torch
# from config import Config

# def get_model():
#     model = timm.create_model(Config.model_name, pretrained=False, num_classes=Config.num_classes)
#     if Config.pretrained:
#         state_dict = torch.load(Config.checkpoint_path, map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model
# model.py
import timm
import torch
from config import Config

def get_model():
    # model = timm.create_model(Config.model_name, pretrained=False, num_classes=Config.num_classes)
    model = timm.create_model(Config.model_name, pretrained=False, num_classes=Config.num_classes, drop_rate=0.1)
    """ , in_chans=6 """
    if Config.pretrained:
        state_dict = torch.load(Config.checkpoint_path, map_location='cpu')

        # 删除分类头（全连接层）的权重
        for key in ['head.weight', 'head.bias']:
            if key in state_dict:
                del state_dict[key]

        model.load_state_dict(state_dict, strict=False)  # 忽略分类头的 shape mismatch
    return model