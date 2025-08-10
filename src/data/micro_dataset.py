import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class MicroGTDataset(Dataset):
    def __init__(self, root, image_size=256):
        self.paths = sorted(
            glob.glob(os.path.join(root, "*.png"))
            + glob.glob(os.path.join(root, "*.jpg"))
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
        x = TF.to_tensor(img) * 2.0 - 1.0
        return {"img": x, "path": p}


def build_image_dataset(root, image_size):
    return MicroGTDataset(root=root, image_size=image_size)
