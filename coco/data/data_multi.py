import torch
from torch.utils import data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from torchvision import transforms

import os

class CocoClsDataset(data.Dataset):
    def __init__(self, root_dir, ann_file, img_dir,  phase):
        self.ann_file = os.path.join(root_dir, ann_file)
        self.img_dir = os.path.join(root_dir, img_dir)
        self.coco = COCO(self.ann_file)
        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                ])
            

        cat_ids = self.coco.getCatIds()
        categories = self.coco.dataset['categories']
        self.id2cat = dict()
        for category in categories:
            self.id2cat[category['id']] = category['name']
        self.id2label = {category['id'] : label for label, category in enumerate(categories)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.label2cat = {v: self.id2cat[k] for k, v in self.id2label.items()}
        self.img_names, self.img_labels = self._load_infos()

    def _load_infos(self):
        img_names = list()
        img_labels = dict()
        ann_ids = self.coco.getAnnIds()
        for ann_id in ann_ids:
            ann = self.coco.loadAnns([ann_id])[0]
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            if ann['area'] <= 0 or w < 1 or h < 1 or ann['iscrowd']:
                continue
            img_meta = self.coco.loadImgs(ann['image_id'])[0]
            file_name = img_meta['file_name']
            label = self.id2label[ann['category_id']]
            if file_name in img_labels:
                img_labels[file_name].append(label)
            else:
                img_labels[file_name] = [label]
                img_names.append(file_name)
            img_labels[file_name] = list(set(img_labels[file_name]))
        return img_names, img_labels



    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        file_name = self.img_names[idx]
        labels = self.img_labels[file_name]
        img_path = os.path.join(self.img_dir, file_name)

        img = Image.open(img_path).convert('RGB')
        labels = np.unique(np.array(labels))
        onehot = torch.zeros(80).float()
        onehot[labels] = 1
        onehot = onehot.float()
        img = self.transform(img)
        return img, onehot

if __name__ == '__main__':
    print('length: ', len(CocoClsDataset))