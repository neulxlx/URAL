import random
import torchvision.transforms.functional as TF
from elasticdeform import deform_random_grid
import torch

def random_flip_rotate(img, label, p=0.5, degrees=15):

    if random.random() < p:
        img = TF.hflip(img)
        label = TF.hflip(label)
        
    if random.random() < p:
        angle = random.randint(-degrees, degrees)
        img = TF.rotate(img, angle, fill=0)
        label = TF.rotate(label, angle)

    return img, label


def random_flip_rotate_corrupt(img, label, label_clear, p=0.5, degrees=15):

    if random.random() < p:
        img = TF.hflip(img)
        label = TF.hflip(label)
        label_clear = TF.hflip(label_clear)
        
    if random.random() < p:
        angle = random.randint(-degrees, degrees)
        img = TF.rotate(img, angle, fill=0)
        label = TF.rotate(label, angle)
        label_clear = TF.rotate(label_clear, angle)

    return img, label, label_clear


def random_label_corrupt(img, label, p=0, num_classes=7):
    
    if random.random() < p:
        img, label = deform_random_grid([img, label], order=0, sigma=5, points=3, axis=(1, 2))
        label[label>(num_classes-1)] = num_classes-1
        label[label<0] = 0
        
    return label.astype(int)

def random_flip_rotate_elastic_deform(img, label, num_classes=7, p=0.5, degrees=15):

    if random.random() < p:
        img, label = deform_random_grid([img, label], order=0, sigma=1, points=3, axis=(1, 2))
        label[label>(num_classes-1)] = num_classes-1
        label[label<0] = 0

    img, label = to_tensor(img, label)

    if random.random() < p:
        img = TF.hflip(img)
        label = TF.hflip(label)
        
    if random.random() < p:
        angle = random.randint(-degrees, degrees)
        img = TF.rotate(img, angle, fill=0)
        label = TF.rotate(label, angle)

    return img, label    


def img_to_tensor(img):
    img = torch.from_numpy(img.copy())
    # img = repeat(img, '1 h w -> 3 h w')
    return img.to(dtype=torch.float32).div(255)

def label_to_tensor(label):
    label = torch.from_numpy(label.copy())
    return label.to(dtype=torch.long)

def to_tensor(img, label):
    img = img_to_tensor(img)
    label = label_to_tensor(label)
    return img, label

