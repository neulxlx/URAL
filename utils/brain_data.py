import cv2 as cv
import nibabel as nib
import numpy as np
import math
import os.path

def to_uint8(vol):
    vol = vol.astype(float)
    vol[vol < 0] = 0    
    return ((vol - vol.min()) * 255.0 / vol.max()).astype(np.uint8)


def histeq_vol(vol):
    for ind in range(vol.shape[0]):
        vol[ind, :, :] = cv.equalizeHist(vol[ind, :, :])
    return vol


def read_vol(PATH):
    return nib.load(PATH).get_data().transpose(2, 0, 1)


def stack_vol(vol, stack_num):
    assert stack_num % 2 == 1, 'stack numbers must be odd!'
    vol = np.expand_dims(vol, axis=1)
    N = range(stack_num // 2, -(stack_num // 2 + 1), -1)
    stacked_vol = np.roll(vol, N[0], axis=0)
    for n in N[1:]:
        stacked_vol = np.concatenate((stacked_vol, np.roll(vol, n, axis=0)), axis=1)
    return stacked_vol


# crop

def calc_ceil_pad(x, devider):
    return math.ceil(x / float(devider)) * devider


def crop_vol(vol, crop_region):
    l_range, r_range, c_range = crop_region
    cropped_vol = vol[l_range[0]: l_range[1], r_range[0]: r_range[1],
                  c_range[0]: c_range[1]]
    return cropped_vol


def get_mask_region(vols, scale1, scale2):
    mask = np.zeros(vols[0].shape[:])
    for vol in vols:
        l, r, c = np.where(vol > 0)
        mask[l, r, c] = 1

    l, r, c = np.where(mask > 0)
    min_l, min_r, min_c = l.min(), r.min(), c.min()
    max_l, max_r, max_c = l.max(), r.max(), c.max()
    max_r = min_r + calc_ceil_pad(max_r - min_r, scale1)
    max_c = min_c + calc_ceil_pad(max_c - min_c, scale1)
    max_l = min_l + calc_ceil_pad(max_l - min_l, scale2)

    pad_r = 0
    pad_c = 0
    if (max_r - min_r) < 96:
        pad_r = (96 - (max_r - min_r))//2
    if (max_c - min_c) < 128:
        pad_c = (128 - (max_c - min_c))//2

    return [(min_l, max_l), (min_r - pad_r, max_r + pad_r), (min_c - pad_c, max_c + pad_c)]


class BrainData:
    def __init__(self):
        self.vols = []

    def read(self, PATH, IDs, suffix):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_vol(os.path.join(PATH, id_vol + '_' + suffix))))

    def histeq(self):
        self.vols = [histeq_vol(vol) for vol in self.vols]

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region[ind])
            self.vols[ind] = cropped_vol

    def stack(self, stack_num):
        self.vols = [stack_vol(vol, stack_num) for vol in self.vols]


    def split(self):
        temp = []
        for vol in self.vols:
            temp.extend(np.split(vol, vol.shape[0], axis=0))
        self.vols = temp


class Brain_Img(BrainData):
    def __init__(self):
        super().__init__()
    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

class Brain_Label(Brain_Img):
    def __init__(self):
        super().__init__()
    def trans_vol_label(self, label_s, label_t):
        assert len(label_s) == len(label_t), 'length must be same!'
        self.num_classes = max(label_t) + 1
        for i in range(len(self.vols)):
            vol = np.zeros(self.vols[i].shape)
            for j in range(len(label_s)):
                l, r, c = np.where(self.vols[i] == label_s[j])
                vol[l, r, c] = label_t[j]
            self.vols[i] = vol
    def split(self):
        vols = []
        for vol in self.vols:
            vols.extend(np.split(vol, vol.shape[0], axis=0))
        self.vols = vols


def read_IBSR_vol(path):
    vol = nib.load(path).get_fdata().squeeze().transpose(2, 1, 0)
    return np.flip(vol, axis=2)

# def read_IBSR_vol(path):
#     vol = nib.load(path).get_fdata().transpose(1, 2, 0)
#     return np.flip(vol, axis=1)

class IBSR_Img(Brain_Img):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_IBSR_vol(os.path.join(
                PATH, '{}_img.nii'.format(id_vol)))))


class IBSR_Label(Brain_Label):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_IBSR_vol(os.path.join(PATH, '{}_seg_6.nii'.format(id_vol))))


def read_MALC_vol(path):
    vol = nib.load(path).get_data().transpose(2, 1, 0)
    return vol


class MALC_Img(Brain_Img):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_MALC_vol(os.path.join(
                PATH, '{}_img.nii'.format(id_vol)))))


class MALC_Label(Brain_Label):
    def __init__(self):
        super().__init__()
    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_MALC_vol(os.path.join(PATH, '{}_seg_6.nii'.format(id_vol))))