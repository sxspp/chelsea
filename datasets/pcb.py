import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PCB(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        if train:
            self.normal_path = path + 'rotation_resize/0'
            self.abnormal_path = path + 'rotation_resize/1'
        else:
            self.normal_path = path + 'rotation_resize/0'
            self.abnormal_path = path + 'rotation_resize/1'
        
        self.normal_img_list = glob.glob(self.normal_path + '/*.JPG')
        self.abnormal_img_list = glob.glob(self.abnormal_path + '/*.jpg')

        self.transform = transform

        self.img_list = self.normal_img_list + self.abnormal_img_list
        self.class_list = [0] * len(self.normal_img_list) + [1] * len(self.abnormal_img_list) 
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label = self.class_list[idx]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = PCB(path='./', train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset,
                        batch_size=1,
                        shuffle=True,
                        drop_last=False)

    for epoch in range(2):
        print(f"epoch : {epoch} ")
        for batch in dataloader:
            img, label = batch
            print(img.size(), label)