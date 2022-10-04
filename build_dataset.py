from torch.utils import data
from einops import rearrange
from yacs.config import CfgNode as CN
from skimage.metrics import structural_similarity as ssim
from utils.transforms import to_tensor, label_to_tensor
from utils.brain_data import *
from label_noise import apply_label_noise

IBSR = CN()
IBSR.name = 'IBSR'
IBSR.PATH = '/IBSR_18'
IBSR.num_classes = 7

MALC = CN()
MALC.name = 'MALC'
MALC.PATH = '/MALC'
MALC.num_classes = 7

Dataset_dict = {'IBSR':IBSR, 'MALC': MALC}


class Dataset(data.Dataset):
    def __init__(self, img, label, transform=None):
        self.img = img
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):

        img, label = to_tensor(self.img[index], self.label[index])
        label = rearrange(label, '1 h w -> h w')
        return img, label


class Dataset_train(data.Dataset):
    def __init__(self, img, label, label_clean, meta_eval_img, meta_eval_label, transform=None):
        self.img = img
        self.label = label
        self.label_clean = label_clean
        self.meta_eval_img = meta_eval_img
        self.meta_eval_label = meta_eval_label
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):

        img, label, label_clean = self.img[index], self.label[index], self.label_clean[index]
        img, label = to_tensor(img, label)
        label_clean = label_to_tensor(label_clean)
        if self.transform:
            img, label, label_clean = self.transform(img, label, label_clean)
        label = rearrange(label, '1 h w -> h w')
        label_clean = rearrange(label_clean, '1 h w -> h w')
        meta_eval_img, meta_eval_label = to_tensor(self.meta_eval_img[index], self.meta_eval_label[index])

        return img, label, label_clean, meta_eval_img, meta_eval_label


def get_meta_eval(train_img_vols, train_label_vols, meta_eval_img_vols, meta_eval_label_vols):
    
    output_img = []
    output_label = []

    for img, label in zip(train_img_vols, train_label_vols):
        img, label = choose_atlas(img, meta_eval_img_vols, meta_eval_label_vols)
        output_img.append(img)
        output_label.append(label)
        
    return output_img, output_label


def choose_atlas(img, atlas_img, atlas_label):
    num_atlas = len(atlas_img)
    s = np.zeros(num_atlas)
    for i in range(num_atlas):
        atlas_img_current = atlas_img[i]
        s[i] = ssim(img.squeeze(), atlas_img_current.squeeze())
    ind = np.argsort(s)
    output_img = atlas_img[ind[-1]]
    output_label = atlas_label[ind[-1]]
    return output_img, output_label



def build_dataset(dataset, train_config, noise_rate=0, beta=3, transform=None):

    dataset_config = Dataset_dict[dataset]
    train_img, train_label = globals()[dataset_config.name + '_Img'](), globals()[dataset_config.name + '_Label']()
    train_label_clean = globals()[dataset_config.name + '_Label']()
    test_img, test_label = globals()[dataset_config.name + '_Img'](), globals()[dataset_config.name + '_Label']()
    eval_img, eval_label = globals()[dataset_config.name + '_Img'](), globals()[dataset_config.name + '_Label']()
    meta_eval_img, meta_eval_label = globals()[dataset_config.name + '_Img'](), globals()[dataset_config.name + '_Label']()
    
    train_img.read(dataset_config.PATH, train_config.IDs_train)
    train_label.read(dataset_config.PATH, train_config.IDs_train)
    train_label_clean.read(dataset_config.PATH, train_config.IDs_train)
    train_mask_region = get_mask_region(train_label.vols, 16, 1)
    train_img.crop(train_mask_region)
    train_label.crop(train_mask_region)
    train_label_clean.crop(train_mask_region)

    train_img.histeq()
    train_img.split()
    train_label.split() 
    train_label_clean.split()
    
    train_label.vols = apply_label_noise(train_label.vols, noise_rate, beta)

    meta_eval_img.read(dataset_config.PATH, train_config.IDs_meta_eval)
    meta_eval_label.read(dataset_config.PATH, train_config.IDs_meta_eval)

    meta_eval_img.crop(train_mask_region)
    meta_eval_label.crop(train_mask_region)

    meta_eval_img.histeq()
    meta_eval_img.split()
    meta_eval_label.split()

    train_img.vols = train_img.vols + meta_eval_img.vols
    train_label.vols = train_label.vols + meta_eval_label.vols
    train_label_clean.vols = train_label_clean.vols + meta_eval_label.vols

    meta_eval_img_vols, meta_eval_label_vols = get_meta_eval(train_img.vols, train_label.vols, meta_eval_img.vols, meta_eval_label.vols)

    eval_img.read(dataset_config.PATH, train_config.IDs_eval)
    eval_label.read(dataset_config.PATH, train_config.IDs_eval)

    eval_img.crop(train_mask_region)
    eval_label.crop(train_mask_region)

    eval_img.histeq()
    eval_img.split()
    eval_label.split()

    test_img.read(dataset_config.PATH, train_config.IDs_test)
    test_label.read(dataset_config.PATH, train_config.IDs_test)

    test_img.crop(train_mask_region)
    test_label.crop(train_mask_region)

    test_img.histeq()
    test_img.split()
    test_label.split()

    train_dataset = Dataset_train(train_img.vols, train_label.vols, train_label_clean.vols, meta_eval_img_vols, meta_eval_label_vols, transform)
    test_dataset  = Dataset(test_img.vols, test_label.vols)
    eval_dataset = Dataset(eval_img.vols, eval_label.vols)
    
    return train_dataset, test_dataset, eval_dataset


