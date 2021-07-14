import cv2
import torch.utils.data as data
from matplotlib import pyplot as plt
import numpy as np
import os

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", "JPG"])

class HazyData(data.Dataset):
    def __init__(self, root,alpha,beta,bias,random=True,extraAugment=True,crop_size=480):
        """
        :param root: Root path of RGB images and their depth maps
        :param alpha: Atmospheric light intensity (0,1]
        :param beta: Atmospheric diffuse reflection coefficient (0,inf)
        :param bias: Background retention factor [0,1)
        :param random: Whether to make actual alpha and beta random from (0,alpha) (0,beta)
        :param extraAugment: Whether to take randomly crop,flip and rotate
        :param crop_size: the image size after crop, only valid when extraAugment==True
        """
        self.root = root
        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.random = random
        self.extAug = extraAugment
        self.crop_size = crop_size
        self.target_dir = os.path.join(self.root, 'gt')
        self.depth_dir = os.path.join(self.root, 'depth')
        self.target_ids = [x for x in sorted(os.listdir(self.target_dir)) if is_image_file(x)]
        self.target_ids.sort(key=lambda x:int(x.split('.')[0]))


    def __len__(self):
        return len(self.target_ids)

    def __getitem__(self, index):
        #targetImage
        name = self.target_ids[index]
        target_image = plt.imread(os.path.join(self.target_dir, name))[:,:,:3]
        if target_image.dtype==np.uint8:
            target_image = np.asarray(target_image, np.float32)
            target_image /= 255
        target_image = np.expand_dims(target_image,0)
        #depthImage
        dep = plt.imread(os.path.join(self.depth_dir,name))
        dep = cv2.cvtColor(dep,cv2.COLOR_RGB2GRAY)
        dep = dep[None,:,:,None]
        if self.extAug:
            #randomly crop
            hw = self.crop_size
            _,h,w,c = target_image.shape
            i = np.random.randint(h-hw+1)
            j = np.random.randint(w-hw+1)
            target_image = target_image[:,i:i+hw,j:j+hw]
            dep = dep[:, i:i + hw, j:j + hw]
            #randomly flip
            flip_channel,rotate_degree = np.random.randint(3,size=2)
            if flip_channel:
                target_image=np.flip(target_image,flip_channel)
                dep = np.flip(dep,flip_channel)
            #randomly rotate
            if rotate_degree:
                target_image=np.rot90(target_image,rotate_degree,(1,2))
                dep=np.rot90(dep,rotate_degree,(1,2))
        #hazyImage
        if self.random:
            beta = np.random.rand(1)*self.beta
            alpha = np.random.rand(1)*self.alpha
        else:
            beta = self.beta
            alpha = self.alpha
        trans = np.zeros_like(dep,np.float32)+self.bias
        np.exp(-beta / dep, out=trans, where=dep > self.bias)
        hazy = (target_image*trans + alpha*(1-trans)).astype(np.float32)
        hazy = np.transpose(hazy,[0,3,1,2])
        target_image = np.transpose(target_image,[0,3,1,2])
        return hazy.copy(), target_image.copy(), name



