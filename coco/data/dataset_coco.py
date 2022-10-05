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
                transforms.RandomHorizontalFlip(),
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
        self.id2cat[0] = 'background'
        self.id2label = {category['id'] : label + 1 for label, category in enumerate(categories)}
        self.id2label[0] = 0
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.labels = []

        tmp_ann_ids = self.coco.getAnnIds()
        self.ann_ids = []

        for i in tmp_ann_ids:
            for ann_id in tmp_ann_ids:
                tmp_list= []
                if i == self.coco.loadAnns([ann_id])[0]['image_id']and self.coco.loadAnns([ann_id])[0]['category_id'] not in tmp_list:
                    tmp_list.append(self.coco.loadAnns([ann_id])[0]['category_id'])
            self.ann_ids.append(ann_id)
            self.labels.append(tmp_list)
        
        self._cal_num_dict()


        print('total_length of dataset:', len(self))


    def __len__(self):
        return len(self.ann_ids) 


    def __getitem__(self, idx):
        # if idx < len(self.ann_ids):
        #     ann = self.coco.loadAnns([self.ann_ids[idx]])[0]

        #     cat_id = ann['category_id']
        #     label = self.id2label[cat_id]

        #     img_meta = self.coco.loadImgs(ann['image_id'])[0]
        #     img_path = os.path.join(self.img_dir, img_meta['file_name'])
        # else:
        #     ann = self.bg_anns[idx - len(self.ann_ids)]

        #     label = 0

        #     img_path = os.path.join(self.img_dir, ann['file_name'])
        ann = self.coco.loadAnns([self.ann_ids[idx]])[0]
        label = self.labels[idx]
        img_meta = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_meta['file_name'])
        x, y, w, h = ann['bbox']

        x, y, w, h = int(x), int(y), int(w), int(h)
        img = img.crop((x, y, x + w - 1, y + h - 1))

        # save_img = img.resize((224, 224), Image.BILINEAR)
        # save_img.save('test.jpg')

        try:
            img = self.transform(img)
        except:
            print(img.mode)
            exit(0)
        return img, label



if __name__ == '__main__':
    coco_cls = CocoClsDataset(root_dir='/mnt/storage1/dataset/coco/', 
                              ann_file='annotations/instances_train2017.json',
                              img_dir='train2017',
                              phase='train')
    print('length: ', len(coco_cls))
    from pprint import pprint
    pprint(coco_cls.num_dict)

# if __name__ == '__main__':
#     coco_cls = CocoClsDataset(root_dir='/home1/share/coco/', 
#                               ann_file='annotations/instances_train2017.json',
#                               img_dir='images/train2017',
#                               bg_bboxes_file='./bg_bboxes/coco_train_bg_bboxes.log',
#                               phase='train')
#     print('length: ', len(coco_cls))
#     from pprint import pprint
#     pprint(coco_cls.num_dict)