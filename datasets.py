import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from typing import List, Tuple
import random
from PIL import Image
from tqdm import tqdm



'''def getImages(part):
    pathsofglory = sorted(glob.glob('../data/' + part + '/*.jpg', recursive=True))
    images = []
    for filepath in tqdm(pathsofglory):
        image = Image.open(filepath)
        temp = image.copy()
        images.append(temp)
        image.close()
    return images'''

class getImages():
    def __init__(self,part):
        self.pathsofglory = sorted(glob.glob('../data/' + part + '/*.jpg', recursive=True))
    def __len__(self):
        return len(self.pathsofglory)
    def __getitem__(self, idx):
        image = Image.open(self.pathsofglory[idx])
        return image




class PadCenterImageTensor:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def padShitRight(size, image: Image) -> Image:
        sh, sw = image.size
        mh, mw = size

        assert sh <= mh and sw <= mw, "The upscaling boundary specified is smaller than an image"

        pleft = (mw - sw) // 2
        ptop = (mh - sh) // 2
        # to make sure the resulting image has the specified size
        pright = mw - sw - pleft
        pbot = mh - sh - ptop

        padding = (pleft, ptop, pright, pbot)
        return F.pad(image, padding, 0, 'constant'), (pleft, ptop)

    def __call__(self, image: Image) -> Image:
        return PadCenterImageTensor.padShitRight(self.size, image)[0]



class ImageWidiSet(Dataset):
    def __init__(self, imagesRAW=[],
                 centerList=None, cropsizeList= None,dataset=None,
                 imageBox: Tuple = (100, 100),
                 tf = transforms.ToTensor(),mode = None): # make a default transform
        # call the super constructor just in case
        super(ImageWidiSet, self).__init__()
        self.mode = mode
        self.dataset = dataset
        # assume we have a numpy array in the input list, transform it to tensor
        #TODO: check if we are doing the train or test set set and apply different transform for the test set
        #self.trans = transforms.Compose([transforms.Resize(size=(100,100),interpolation=Image.LANCZOS), #TODO: specify the fixed aspect ratio or this cocksucker is gonna bitch about sizes in assert
                                   #PadCenterImageTensor((100, 100)),#imageBox),
                                    #transforms.ToTensor()]) # TODO: add the posibility for a custom transform
        if self.mode is not None:
                self.imgtens = [transforms.functional.to_tensor(img) for img in imagesRAW]

        # check what kind of stuff
        if centerList is not None and cropsizeList is not None:
            # if we are here, just assign the centerList and cropSizeList
            self.crops = cropsizeList
            self.centers = centerList
        elif centerList is None and cropsizeList is None:
            # generate centers and crops
            # TODO: make a more exciting way of generating the training data
            self.crops, self.centers = generate_stuff()

        else:
            raise ValueError("Should specify neither or both arguments for cropping.")

    def __len__(self):
        if self.mode is not None:
            return len(self.imgtens)
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        self.trans = transforms.Compose([transforms.Resize(size=(random.randrange(70,100), random.randrange(70,100)),
                                                           interpolation=Image.LANCZOS),  # TODO: specify the fixed aspect ratio or this cocksucker is gonna bitch about sizes in assert
         transforms.ToTensor()]) # TODO: add the posibility for a custom transform
        if self.mode is not None:
            X = self.imgtens[idx]
            #X[:] /= 255.
            #Xmean = X.mean()
            #Xstd = X.std()
            #X = X - X.mean()
            #X = X / X.std()
        else:
            X = self.trans(self.dataset.__getitem__(idx))
            #X[:]/=255.
            #X = X-X.mean()
            #X = X/X.std()
        center, crop = self.centers[idx], self.crops[idx]

        #compute the crop boundaries
        Hs, He = center[0] - crop[0], center[0] + crop[0]
        Ws, We = center[1] - crop[1], center[1] + crop[1]

        # produce the masks
        # assume x is of shape [1, nchn, h, w]
        nchcn, h, w = X.shape
        mask = torch.zeros(1, h, w)
        mask[0, Hs:He, Ws:We] = 1

        # remove the part that we want to predict
        X.requires_grad= False
        mask.requires_grad= False
        Y = X*mask
        Y.requires_grad=False
        X = X*(1-mask)

        # add the mask channel to the image (concatenate on the channel dimension)
        #X = torch.cat((X, mask), dim=0)

        return X, Y, mask


def validate_input(crop_size, crop_center):  # validates the input of the ex4 function
    """
    :param image_array: A numpy array containing the image data in an arbitrary datatype and shape (X, Y).
    :param crop_size:A tuple containing 2 odd int values. These two values specify the size
           of the rectangle that should be cropped-out in pixels for the two spatial dimensions X and Y.
    :param crop_center:A tuple containing 2 int values. These two values are the position of the
           center of the to-be cropped-out rectangle in pixels for the two spatial dimensions X and Y.
    """
    valid = True
    distance = 20
    x, y = crop_center
    dx, dy = crop_size
    # integer division
    dx, dy = dx // 2, dy // 2
    dimx, dimy = (100, 100)
    if any(x < distance for x in [x - dx, dimx - (x + dx + 1), y - dy, dimy - (y + dy + 1)]):
        valid = False
    return valid

def generate_stuff():
    crop_sizelist = []
    crop_centerlist = []
    for i in range(5, 22, 2):
        for j in range(5, 22, 2):
            for k in range(70):
                for l in range(100):
                    ik = random.randrange(5, 22, 2)
                    jk = random.randrange(5, 22, 2)
                    kk = random.randrange(0, 100)
                    lk = random.randrange(0, 100)
                    valid = validate_input((ik, jk), (kk, lk))
                    if valid:
                        crop_centerlist.append((kk, lk))
                        crop_sizelist.append((ik, jk))
    return crop_sizelist, crop_centerlist

from matplotlib import pyplot as plt

if __name__=="__main__":
    imgs = getImages()
    ds = ImageWidiSet(imgs)

    X, Y, m = ds[1]
    plt.imshow(X[0].numpy())
    plt.show()
    plt.imshow(Y[0].numpy())
    plt.show()
    plt.imshow(m[0].numpy())
    plt.show()
