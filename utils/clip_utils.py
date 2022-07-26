import torch
import torch.nn as nn
import torchvision.transforms as T

class CLIP_aug(object) :
    def __init__(self):
        self.n = 10
        self.augmentation = T.Compose([
            T.Resize((224,224)),
            T.RandomPerspective(0.5,1),
            T.RandomVerticalFlip(0.5),
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=(0,90), scale=(0.8, 1.2))
        ])
    def __call__(self,x) :
        out = []
        for _ in range(self.n) :
            out.append(self.augmentation(x))
        out = torch.cat(out,dim=0)
        return out

class VGG_aug(object) :
    def __init__(self):
        self.n = 25
        self.augmentation = T.Compose([
            T.Resize((224,224)),
            T.RandomPerspective(0.5,1),
            T.RandomVerticalFlip(0.5),
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(degrees=(0,90), scale=(0.8, 1.2))
        ])
    def __call__(self,x) :
        out = []
        for _ in range(self.n) :
            out.append(self.augmentation(x))
        out = torch.cat(out,dim=0)
        return out
