from glob import glob
from sre_constants import IN
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from data.imagenet_dic import IMAGENET_DIC
from torch.utils.data import Dataset
import os

IN_NUM_CLASSES = 1000

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    # import pdb; pdb.set_trace()
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img
    

@register_dataset(name='imagenet')
class IMAGENET_dataset(Dataset):
    def __init__(self, root: str,transforms: Optional[Callable]=None, mode: Optional[Callable]="val", dataset_size: Optional[Callable]=1000 ):
        super().__init__()
        
        self.image_paths = list()
        self.y = list()
        for cl in range(IN_NUM_CLASSES):
            data_dir_batch = os.path.join(root, mode, IMAGENET_DIC[str(cl)][0], "*.JPEG")
            image_paths_batch = sorted(glob(data_dir_batch))
            y_batch = [cl for _ in range(len(image_paths_batch))]

            self.image_paths += image_paths_batch
            self.y += y_batch
            if len(self.y)>dataset_size:
                break

        self.transform = transforms
    def __getitem__(self, index):
        f = self.image_paths[index]
        lab = self.y[index]
        pil_image = Image.open(f)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image, lab

    def getImagePaths(self):
        return self.image_paths

    def __len__(self):
        return len(self.image_paths)