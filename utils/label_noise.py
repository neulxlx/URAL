import random
from einops import rearrange
from elasticdeform import deform_random_grid
import cv2

def apply_elastic_deform(vols, p=0, beta=3):
    output = []
    for vol in vols:
        if random.random() < p:
            vol = deform_random_grid(vol.squeeze(), order=0, sigma=2, points=beta)
            vol = rearrange(vol, 'h w -> 1 h w')
        output.append(vol)
    return output


def apply_erode_dilate(vols, p=0, beta=3):
    output = []
    kernel = np.ones((beta, beta),np.uint8)
    for vol in vols:
        if random.random() <= p:
            if random.random() <= 0.5:
                vol = cv2.erode(vol,kernel,iterations=1)
            else:
                vol = cv2.dilate(vol, kernel, iterations=1)
        output.append(vol)
    return output


def apply_label_noise(vols, p=0, beta=3):
    output = []
    kernel = np.ones((beta, beta),np.uint8)
    p_ = random.random()
    for vol in vols:
        if random.random() <= p:
            if p_ <= 0.33:
                vol = cv2.erode(vol,kernel,iterations=1)
            elif p_<= 0.66:
                vol = cv2.dilate(vol, kernel, iterations=1)
            else:
                vol = deform_random_grid(vol.squeeze(), order=0, sigma=2, points=beta)
                vol = rearrange(vol, 'h w -> 1 h w')
        output.append(vol)
    return output    