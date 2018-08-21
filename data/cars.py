import torch.utils.data as data
from PIL import Image
import cv2
import os
import numpy
import numpy.random as rand
import numpy as np
import torch
# from natsort import natsorted

def imageload(path1):
    img1 = cv2.imread(path1)
    # img1 = np.swapaxes(img1,0,2)
    return img1

class Cars(data.Dataset):
    def __init__(self,path,transform=None,seed=100):
        files = os.listdir(path)
        names = []
        for file in files:
            ful_file = os.path.join(path,file)
            names.append(ful_file)
        # print path
        # print names
        self.names = names
        self.transform = transform
        self.seed = seed
        rand.seed(self.seed)

    def __getitem__(self,index):
    
        image=[]

        for i in range(4):
            key = rand.random_integers(0,63)
            x = self.names[index].split('/')[-1] + '-' + str(key) + '.png'
            name1 = os.path.join(self.names[index],x)
            image1 = imageload(name1)
            image.append(image1)

        image = np.array(image)/255.0
        # print image.shape
        image = torch.from_numpy(image)
        if self.transform is not None:
            image = self.transform(image)
                
        return image

   
    def __len__(self):
 	    return len(self.names)   
#return len(self.images)

