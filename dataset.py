from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from configs.config import CFGS

class myDataset(Dataset):
    def __init__(self, root_img):
        
        self.img_size = CFGS["model"]["image_size"]
        self.root_img = root_img
        self.transform = Compose([
                                Resize((self.img_size, self.img_size)),
                                ToTensor(),
                                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]
                            )
        self.images = os.listdir(root_img)
        self.length_dataset = len(self.images)
        
    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        img = self.images[index % self.length_dataset]
    
        img_path = os.path.join(self.root_img, img)
        
        img = Image.open(img_path)
        w, _ = img.size
        img = np.array(img)
        width_cutoff = w // 2
        s1 = Image.fromarray(img[:, :width_cutoff])
        s2 = Image.fromarray(img[:, width_cutoff:])
        
        if self.transform:
            A_img = self.transform(s1)
            B_img = self.transform(s2)

        return A_img, B_img
    
        
        