from torch.utils.data import DataLoader
import torch
from project import PNGDataset
from project import CropDataset
from project import FSRCNN
import pandas as pd

ENVIRONMENT = 'LOCAL'
if ENVIRONMENT == 'LOCAL':
    PATH_TR_LR = r'/Users/randle_h/Downloads/DIV2K_train_LR_bicubic/X2'
    PATH_TR_HR = r'/Users/randle_h/Downloads/DIV2K_train_HR'
    PATH_VA_LR = r'/Users/randle_h/Downloads/DIV2K_valid_LR_bicubic/X2'
    PATH_VA_HR = r'/Users/randle_h/Downloads/DIV2K_valid_HR'
    PATH_MODEL = r'/Users/randle_h/Desktop/model.pt'
    PATH_OPTIM = r'/Users/randle_h/Desktop/optim.pt'
    PATH_CSV   = r'/Users/randle_h/Desktop/loss.csv'
else:
    PATH_TR_LR = r'/gdrive/MyDrive/Colab_Notebooks/DIV2K_train_LR_bicubic/X2'
    PATH_TR_HR = r'/gdrive/MyDrive/Colab_Notebooks/DIV2K_train_HR'
    PATH_VA_LR = r'/gdrive/MyDrive/Colab_Notebooks/DIV2K_valid_LR_bicubic/X2'
    PATH_VA_HR = r'/gdrive/MyDrive/Colab_Notebooks/DIV2K_valid_HR'
    PATH_MODEL = r'/gdrive/MyDrive/Colab_Notebooks/model.pt'
    PATH_OPTIM = r'/gdrive/MyDrive/Colab_Notebooks/optim.pt'
    PATH_CSV   = r'/gdrive/MyDrive/Colab_Notebooks/loss.csv'

# 制作训练集
valid_set = PNGDataset ( PATH_VA_LR, PATH_VA_HR )                 # 完整的图像测试集
patch_set = CropDataset( PATH_TR_LR, PATH_TR_HR, (128, 128), 2 )  # 从图像中随机裁剪128x128像素截图作为训练集

# 转为DataLoader
loader_valid = DataLoader(dataset=valid_set, shuffle=False, batch_size=1 )
loader_patch = DataLoader(dataset=patch_set, shuffle=False, batch_size=10)

# 构建模型
model = FSRCNN()

# 初次训练时, 注释本行
# model.load_state_dict(torch.load(PATH_MODEL))

# 开始训练, 返回每次验证后保存的PSNR和LOSS值
list_psnr, list_loss = model.start( loader_patch, loader_valid, n_epoch=10, validate=True )

# 保存学习曲线数据
pd.DataFrame( data={'PSNR': list_psnr, 'LOSS': list_loss} ).to_csv( PATH_CSV )

# 保存模型参数
model.save_param( PATH_OPTIM, PATH_MODEL)





