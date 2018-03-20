#import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class FrameRecord(object):
    def __init__(self, row):
        """
        frame level record
        :param row:
        """
        self._data = row

    @property
    def path(self):
        return "/".join(self._data[0].split('/')[3:]).replace(".mp4","")    #dataset_name/video_name/imgage_name.jpg
    @property
    def video_name(self):
        return self._data[0].split('/')[-2]
    @property
    def label(self):
        return int(self._data[1])
    @property
    def dataset_name(self):
        return self._data[0].split('/')[-3]

class TSNDataSet_frame(object):
    def __init__(self, root_path, list_file,modality='RGB',transform=None,
                 num_segments=3, new_length=1,
                 force_grayscale=False, test_mode=False):

        self.root_path = root_path  # video path: /workspace/run/UCF_frames
        self.list_file = list_file  # .txt, row: image_path, label
        self.modality = modality
        self.transform = transform
        self.num_segments = num_segments
        self.new_length = new_length
        self.frames_per_video = self.num_segments * self.new_length
        self.test_mode = test_mode

        self._parse_list()
        self.video_list = self._parse_frame_list()

    def _load_image(self, img_path):
        img_path = os.path.join(root_path,img_path)
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(img_path).convert('RGB')]
            except:
                print("Couldn't load image:{}".format(img_path))
                return None
        elif self.modality == 'Flow':
            x_img = Image.open(img_path).convert('L')
            y_img = Image.open(img_path).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.frame_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _parse_frame_list(self):
        video_list = []                 # [[image1,image2,image3,...,label],...]
        Cur_video_label = self.frame_list[0].label
        self.dataset_name = self.frame_list[0].dataset_name
        Cur_video_list = []
        for record in self.frame_list:
            if record.label == Cur_video_label:
                Cur_video_list.append(record.path)
            else:
                if len(Cur_video_list) != self.frames_per_video:
                    Cur_video_list = [record.path]
                    Cur_video_label = record.label
                    continue
                Cur_video_list.append(Cur_video_label)
                video_list.append(Cur_video_list)
                Cur_video_list = [record.path]
                Cur_video_label = record.label
        if len(Cur_video_list) == self.frames_per_video:
            Cur_video_list.append(Cur_video_label)
            video_list.append(Cur_video_list)
        return video_list
    def __getitem__(self, index=-1):
        """
        getitem through index or video_id
        :param index:
        :return:
        """
        assert index != -1, "Please give a valid index"
        index = index % self.__len__()
        process_data,label = self.get(index)
        return process_data,label

    def get(self,index):
        """
        :param index:       video_index
        :return:
        """
        record = self.video_list[index]  # [image1,image2,..., label]
        images = list()
        for image_path in record[:-1]:
            image = self._load_image(image_path)
            images.append(image)
        while None in images:
            return self.get(index+1)
        if self.transform:
            process_data = self.transform(images)
        else:
            process_data = images
        return process_data, record[-1]

    def __len__(self):
        return len(self.video_list)
if __name__ == '__main__':
    root_path = "/workspace/run/ActivityNet/data"
    listfile = "/workspace/run/ActivityNet/data/ucf101_rgb_train_split_1_frame.txt"
    test = TSNDataSet_frame(root_path=root_path,list_file=listfile,num_segments=3,
                            new_length=1,modality="RGB",transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]))
    """
    TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])
    """
    data,label = test.__getitem__(index=10)
    print(data,label)