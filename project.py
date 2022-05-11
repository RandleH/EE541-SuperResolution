import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import tqdm

def psnr( clean, noisy, max_value=1.0, on_gpu=True):
    if on_gpu:
        np_ground = np.array(clean.cpu().detach().numpy(), dtype='float')
        np_compressed = np.array(noisy.cpu().detach().numpy(), dtype='float')
    else:
        np_ground = np.array(clean.detach().numpy(), dtype='float')
        np_compressed = np.array(noisy.detach().numpy(), dtype='float')
    mse  = np.mean((np_ground - np_compressed)**2)
    psnr = np.log10(max_value**2/mse) * 10
    return psnr

class PNGDataset(data.Dataset):
    def __init__(self, lr_dir, hr_dir):
        super(PNGDataset, self).__init__()
        self.img_dir_lr   = lr_dir
        self.img_dir_hr   = hr_dir
        self.img_names_lr = [ n for n in os.listdir( lr_dir ) ]
        self.img_names_hr = [ n for n in os.listdir( hr_dir ) ]
        self.img_names_lr.sort()
        self.img_names_hr.sort()

    def __getitem__(self, idx):
        img_lr = cv2.imread( os.path.join( self.img_dir_lr, self.img_names_lr[idx]) )
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        # assert( img_lr!=None )
        img_hr = cv2.imread( os.path.join( self.img_dir_hr, self.img_names_hr[idx]) )
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        # assert( img_hr!=None )
        return transforms.ToTensor()(img_lr), transforms.ToTensor()(img_hr)

    def __len__(self):
        assert( len(self.img_names_lr)==len(self.img_names_hr) )
        return len( self.img_names_lr )

class CropDataset(PNGDataset):
    def __init__(self, lr_dir, hr_dir, dim, scale):
        super(CropDataset, self).__init__( lr_dir, hr_dir)
        self.scale = scale
        self.dim   = dim

    def __getitem__(self, idx):
        img_lr = cv2.imread( os.path.join( self.img_dir_lr, self.img_names_lr[idx]) )
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        xs = random.randrange( 0, img_lr.shape[0]-self.dim[0] )
        ys = random.randrange( 0, img_lr.shape[1]-self.dim[1] )
        img_lr = img_lr[ xs:xs+self.dim[0], ys:ys+self.dim[1], : ]
        assert( img_lr.shape == ( self.dim[0], self.dim[1], 3) )

        img_hr = cv2.imread( os.path.join( self.img_dir_hr, self.img_names_hr[idx]) )
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        xs *= self.scale
        ys *= self.scale
        img_hr = img_hr[ xs:xs+self.scale*self.dim[0], ys:ys+self.scale*self.dim[1], : ]
        assert( img_hr.shape == ( self.scale*self.dim[0], self.scale*self.dim[1], 3) )
        return transforms.ToTensor()(img_lr), transforms.ToTensor()(img_hr)

    def __len__(self):
        assert( len(self.img_names_lr) == len(self.img_names_hr) )
        return len( self.img_names_lr )

def psnr( clean, noisy, max_value=1.0, on_gpu=True):
    if on_gpu:
        np_ground = np.array(clean.cpu().detach().numpy(), dtype='float')
        np_compressed = np.array(noisy.cpu().detach().numpy(), dtype='float')
    else:
        np_ground = np.array(clean.detach().numpy(), dtype='float')
        np_compressed = np.array(noisy.detach().numpy(), dtype='float')
    mse  = np.mean((np_ground - np_compressed)**2)
    psnr = np.log10(max_value**2/mse) * 10
    return psnr

class BASE_NN(nn.Module):
    def __init__(self, lr=0.0005):
        super(BASE_NN, self).__init__()
        self.learning_rate = lr
        self.loss  = nn.MSELoss()

    def info(self):
        print("Model's state_dict:")
        for param_tensor in self.state_dict():
            print(param_tensor, "\t", self.state_dict()[param_tensor].size())

    def show_filter(self, layer):
        t = 0
        for param in self.parameters():
            if( t==layer ):
                n_pic_h = 4
                n_pic_w = param.size()[0]//n_pic_h + 1
                cnt     = 1
                for i, weights in enumerate( param ):
                    plt.subplot( n_pic_h, n_pic_w, cnt )
                    w = weights.detach().numpy()
                    if( w.shape==() ):
                        w = w.reshape((1,1))
                    else:
                        w = np.average( w, 0 )
                    plt.imshow( w, cmap='viridis' )
                    plt.title( f'{cnt}' )
                    plt.axis('off')
                    cnt += 1
                break
            t += 1
        plt.show()

    def start(self, train_loader, valid_loader, n_epoch=10, validate=False):
        list_psnr = []  # Peak Signal Noise Ratio
        list_loss = []  # Loss function ie. MSE
        for epoch in range(n_epoch):
            self.train()  # Shift to training mode
            with tqdm( train_loader, unit='batch' ) as loader:
                loader.set_description(f"Epoch {epoch}\t TRAIN")
                for lr, hr in loader:
                    outputs = self(lr)
                    loss    = self.loss( outputs, hr)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

            if validate:
                self.eval()
                with tqdm( valid_loader, unit='batch' ) as loader:
                    tmp_psnr = []
                    tmp_loss = []
                    loader.set_description(f"Epoch {epoch}\t VALID")
                    for lr, hr in loader:
                        y = self(lr)
                        tmp_psnr.append( psnr( hr, y, max_value=1.0, on_gpu=True ) )
                        tmp_loss.append( self.loss( hr, y ).detach().numpy() )
                    list_psnr.append( np.average(tmp_psnr) )
                    list_loss.append( np.average(tmp_loss) )

        return list_psnr, list_loss

    def save_param(self, path_optim, path_model):
        torch.save( self.state_dict()    , path_model )
        torch.save( self.optim.state_dict(), path_optim )

    def infer(self, x):
        self.eval()
        y_ = self(x)[0].squeeze().detach().numpy().transpose(1,2,0)
        return np.interp( y_, (y_.min(), y_.max()), (0, 255)).astype(np.uint8)

class SRCNN(BASE_NN):
    def __init__(self, lr=0.0005):
        super(SRCNN, self).__init__(lr)
        self.conv1 = nn.Conv2d(  3, 64, (9,9), stride=1, padding='same')
        self.conv2 = nn.Conv2d( 64, 32, (1,1), stride=1, padding='same')
        self.conv3 = nn.Conv2d( 32,  3, (5,5), stride=1, padding='same')
        self.relu  = nn.ReLU()
        self.optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        pass

    def forward(self, x):
        x = F.interpolate( x, scale_factor=2, mode='bicubic' )
        x = self.relu( self.conv1(x) )
        x = self.relu( self.conv2(x) )
        x = self.relu( self.conv3(x) )
        return x

class FSRCNN(BASE_NN):
    def __init__(self, lr=0.0005):
        super(FSRCNN, self).__init__(lr)
        self.conv1 = nn.Conv2d(  3, 56, (5,5), stride=1, padding='same')
        self.conv2 = nn.Conv2d( 56, 12, (1,1), stride=1, padding='same')
        self.conv3 = nn.Conv2d( 12, 12, (3,3), stride=1, padding='same')  # x 4
        self.conv4 = nn.Conv2d( 12, 56, (1,1), stride=1, padding='same')
        self.relu  = nn.ReLU()
        self.upsample = nn.ConvTranspose2d( 56, 3, (9,9), stride=2, padding=(4,4), output_padding=1)
        self.optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        pass

    def forward(self, x):
        x = self.relu( self.conv1(x) )
        x = self.relu( self.conv2(x) )

        x = self.relu( self.conv3(x) )
        x = self.relu( self.conv3(x) )
        x = self.relu( self.conv3(x) )
        x = self.relu( self.conv3(x) )

        x = self.relu( self.conv4(x) )

        x = self.relu( self.upsample(x) )
        return x

class BTSRN(BASE_NN):
    def __init__(self, lr=0.0005):
        super(BTSRN, self).__init__(lr)
        self.conv1 = nn.Conv2d( 3, 64, (1,1), stride=1, padding='same' )

        self.conv_res1 = nn.Conv2d( 64,  32, (1,1), stride=1, padding='same' )
        self.conv_res2 = nn.Conv2d(  32, 64, (3,3), stride=1, padding='same' )
        self.deconv    = nn.ConvTranspose2d( 64, 64, (2,2), stride=2, padding=0, output_padding=0)
        self.conv2     = nn.Conv2d( 64,   3, (1,1), stride=1, padding='same' )

        self.relu  = nn.ReLU()
        self.optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x_bicubic = x.clone().detach()
        x_bicubic = F.interpolate( x_bicubic, scale_factor=2, mode='bicubic' )

        x = self.conv1(x)

        # LR Stage
        for i in range(6):
            x_ = x.clone().detach()
            x  = self.relu(x)
            x  = self.relu( self.conv_res1(x) )
            x  = self.relu( self.conv_res2(x) )
            x += x_

        # Up-sampling
        x_ = x.clone().detach()
        x_ = F.interpolate( x_, scale_factor=2, mode='nearest' )
        x  = self.deconv(x)
        x += x_

        # HR Stage
        for i in range(4):
            x_ = x.clone().detach()
            x  = self.relu(x)
            x  = self.relu( self.conv_res1(x) )
            x  = self.relu( self.conv_res2(x) )
            x += x_

        x  = self.conv2(x)

        x += x_bicubic
        return x
