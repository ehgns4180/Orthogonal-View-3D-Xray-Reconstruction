import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

class VGG_aug(object) :
    def __init__(self):
        self.n = 5
        self.augmentation = T.Compose([
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

def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a, b, c * d)
    
    G = torch.einsum('abc,acd -> abd',features, features.transpose(1,2))
    G = G.div(c * d)
    return G

# Style / Content 손실 계산을 원하는 계층
# [0,5,10,19,21,28]
def get_features(x, model, layers):
    features = []
    for name, layer in enumerate(model.children()): # 0, conv
        x = layer(x)
        if str(name) in layers:
            features.append(x)
    return features
