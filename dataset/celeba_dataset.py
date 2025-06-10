import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image
import glob
import os
import numpy as np
from torchvision import transforms

attributes = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
    "Young"
]  # 40种属性

PIL_INTERPOLATION = {
    "linear": Image.Resampling.BILINEAR,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}
VALID_IMG_FORMATS = ("jpeg", "png", "jpg", "webp")

# root_path = "/media/inspur/disk/yychang_workspace/data/celeba_hq_latent"
root_path = "/media/inspur/disk/yychang_workspace/data/celeba_hq_1024"
label_file_path = '/media/inspur/disk/yychang_workspace/data/celeba_hq_1024_attr.txt'


class celeba_Dataset(Dataset):
    def __init__(self,
                 attribute="Smiling",
                 image_root_path=root_path,
                 label_path=label_file_path,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 ):
        self.attribute = attribute
        self.size = int(size)
        self.flip_p = flip_p
        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.image_root_path = image_root_path
        self.label_path = label_path
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        self.image_paths = []
        for root, dirs, files in os.walk(self.image_root_path):
            for file in sorted(files):
                file_extension = file.split(".")[-1]  # 分割文件名，取最后一个部分作为文件扩展名。
                if file_extension in VALID_IMG_FORMATS:
                    self.image_paths.append(os.path.join(root, file))

        self.labels_dict = []
        with open(self.label_path, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.strip('[]')
                label_per_image = [int(item) for item in line.split(",")]
                self.labels_dict.append(label_per_image)

        try:
            index_in_list = attributes.index(self.attribute)
            self.attribute_index = index_in_list
        except ValueError:
            print(f'Attribute "{attribute}" error!')

        assert len(self.image_paths) == len(self.labels_dict), "The image does not match the label."

        print(f"CelebaHQ_1024 init successfully with attribution {self.attribute}!")

    def __getitem__(self, index):
        data_entry = dict()
        image = Image.open(self.image_paths[index])
        image = image.convert("RGB") if image.mode != "RGB" else image

        image_processed = image.resize((self.size, self.size), resample=self.interpolation)
        image_processed = self.flip_transform(image_processed)  # 以一定概率水平翻转图像
        image_processed = np.array(image_processed).astype(np.uint8)
        image_processed = (image_processed / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
        image_processed = torch.from_numpy(image_processed).permute(2, 0, 1)  # HWC->CHW

        data_entry["img_pixel_values"] = image_processed
        data_entry["img_path"] = self.image_paths[index]
        data_entry["img_label"] = self.labels_dict[index][self.attribute_index]
        return data_entry

    def __len__(self):
        return len(self.image_paths)


class celeba_Dataset_RA(Dataset):
    def __init__(self,
                 attribute="Mustache",
                 image_root_path=root_path,
                 label_path=label_file_path,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 is_train=True
                 ):
        self.attribute = attribute
        self.size = int(size)
        self.flip_p = flip_p
        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.image_root_path = image_root_path
        self.label_path = label_path
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.is_train = is_train

        self.image_paths = []
        for root, dirs, files in os.walk(self.image_root_path):
            for file in sorted(files):
                file_extension = file.split(".")[-1]  # 分割文件名，取最后一个部分作为文件扩展名。
                if file_extension in VALID_IMG_FORMATS:
                    self.image_paths.append(os.path.join(root, file))

        self.labels_dict = []
        with open(self.label_path, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.strip('[]')
                label_per_image = [int(item) for item in line.split(",")]
                self.labels_dict.append(label_per_image)

        try:
            index_in_list = attributes.index(self.attribute)
            self.attribute_index = index_in_list
            self.attribute_index_single = [sublist[self.attribute_index] for sublist in self.labels_dict]
        except ValueError:
            print(f'Attribute "{attribute}" error!')

        self.pos_list = []
        self.neg_list = []
        for index, label in enumerate(self.attribute_index_single):
            if label == 0:
                self.neg_list.append(index)
            elif label == 1:
                self.pos_list.append(index)

        self.length = min(len(self.pos_list), len(self.neg_list))

        assert len(self.image_paths) == len(self.labels_dict), "The image does not match the label."

        print("CelebaHQ_1024 for relative attribution init successfully!")

    def __getitem__(self, index):
        data_entry = dict()

        image_1 = Image.open(self.image_paths[self.pos_list[index]])
        image_1 = image_1.convert("RGB") if image_1.mode != "RGB" else image_1

        image_processed_1 = image_1.resize((self.size, self.size), resample=self.interpolation)
        image_processed_1 = self.flip_transform(image_processed_1)  # 以一定概率水平翻转图像
        image_processed_1 = np.array(image_processed_1).astype(np.uint8)
        image_processed_1 = (image_processed_1 / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
        img_1 = torch.from_numpy(image_processed_1).permute(2, 0, 1)  # HWC->CHW
        img_path_1 = self.image_paths[self.pos_list[index]]

        image_2 = Image.open(self.image_paths[self.neg_list[index]])
        image_2 = image_2.convert("RGB") if image_2.mode != "RGB" else image_2

        image_processed_2 = image_2.resize((self.size, self.size), resample=self.interpolation)
        image_processed_2 = self.flip_transform(image_processed_2)  # 以一定概率水平翻转图像
        image_processed_2 = np.array(image_processed_2).astype(np.uint8)
        image_processed_2 = (image_processed_2 / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
        img_2 = torch.from_numpy(image_processed_2).permute(2, 0, 1)  # HWC->CHW
        img_path_2 = self.image_paths[self.neg_list[index]]

        difference_label = -1

        if random.random() > 0.5 and self.is_train:  # 训练时有 50% 的概率交换
            img_1, img_2 = img_2, img_1
            img_path_1, img_path_2 = img_path_2, img_path_1
            difference_label = 1
        data_entry["img_pixel_values_1"] = img_1
        data_entry["img_pixel_values_2"] = img_2
        data_entry["img_path_1"] = img_path_1
        data_entry["img_path_2"] = img_path_2
        data_entry["difference_label"] = difference_label
        return data_entry

    def __len__(self):
        return self.length


class celeba_Dataset_single(Dataset):
    def __init__(self,
                 attribute="Mustache",
                 image_root_path=root_path,
                 label_path=label_file_path,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 is_train=True,
                 get_label=0,
                 ):
        self.attribute = attribute
        self.size = int(size)
        self.flip_p = flip_p
        self.interpolation = PIL_INTERPOLATION[interpolation]
        self.image_root_path = image_root_path
        self.label_path = label_path
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.is_train = is_train
        self.get_label = get_label

        self.image_paths = []
        for root, dirs, files in os.walk(self.image_root_path):
            for file in sorted(files):
                file_extension = file.split(".")[-1]  # 分割文件名，取最后一个部分作为文件扩展名。
                if file_extension in VALID_IMG_FORMATS:
                    self.image_paths.append(os.path.join(root, file))

        self.labels_dict = []
        with open(self.label_path, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.strip('[]')
                label_per_image = [int(item) for item in line.split(",")]
                self.labels_dict.append(label_per_image)

        try:
            index_in_list = attributes.index(self.attribute)
            self.attribute_index = index_in_list
            self.attribute_index_single = [sublist[self.attribute_index] for sublist in self.labels_dict]
        except ValueError:
            print(f'Attribute "{attribute}" error!')

        self.pos_list = []
        self.neg_list = []
        for index, label in enumerate(self.attribute_index_single):
            if label == 0:
                self.neg_list.append(index)
            elif label == 1:
                self.pos_list.append(index)
        if self.get_label == 0:
            self.final_list = self.neg_list
        else:
            self.final_list = self.pos_list
        self.length = len(self.final_list),

        assert len(self.image_paths) == len(self.labels_dict), "The image does not match the label."

    def __getitem__(self, index):
        data_entry = dict()
        image_1 = Image.open(self.image_paths[self.final_list[index]])
        image_1 = image_1.convert("RGB") if image_1.mode != "RGB" else image_1

        image_processed_1 = image_1.resize((self.size, self.size), resample=self.interpolation)
        image_processed_1 = self.flip_transform(image_processed_1)  # 以一定概率水平翻转图像
        image_processed_1 = np.array(image_processed_1).astype(np.uint8)
        image_processed_1 = (image_processed_1 / 127.5 - 1.0).astype(np.float32)  # Pixel values between -1, 1
        img_1 = torch.from_numpy(image_processed_1).permute(2, 0, 1)  # HWC->CHW
        img_path_1 = self.image_paths[self.final_list[index]]

        data_entry["img_pixel_values"] = img_1
        data_entry["img_path"] = img_path_1

        return data_entry

    def __len__(self):
        return len(self.final_list)
        # return 1000


"""
5_o_Clock_Shadow:正样本数量4491，比例0.1497
Arched_Eyebrows:正样本数量11020，比例0.36733333333333335
Attractive:正样本数量17218，比例0.5739333333333333
Bags_Under_Eyes:正样本数量8634，比例0.2878
Bald:正样本数量712，比例0.023733333333333332
Bangs:正样本数量5425，比例0.18083333333333335
Big_Lips:正样本数量10890，比例0.363
Big_Nose:正样本数量9734，比例0.3244666666666667
Black_Hair:正样本数量6592，比例0.21973333333333334
Blond_Hair:正样本数量5126，比例0.17086666666666667
Blurry:正样本数量113，比例0.003766666666666667
Brown_Hair:正样本数量6925，比例0.23083333333333333
Bushy_Eyebrows:正样本数量5676，比例0.1892
Chubby:正样本数量2102，比例0.07006666666666667
Double_Chin:正样本数量1786，比例0.059533333333333334
Eyeglasses:正样本数量1468，比例0.048933333333333336
Goatee:正样本数量2290，比例0.07633333333333334
Gray_Hair:正样本数量1242，比例0.0414
Heavy_Makeup:正样本数量13708，比例0.45693333333333336
High_Cheekbones:正样本数量13847，比例0.4615666666666667
Male:正样本数量11057，比例0.36856666666666665
Mouth_Slightly_Open:正样本数量14139，比例0.4713
Mustache:正样本数量1735，比例0.057833333333333334
Narrow_Eyes:正样本数量3516，比例0.1172
No_Beard:正样本数量24328，比例0.8109333333333333
Oval_Face:正样本数量6243，比例0.2081
Pale_Skin:正样本数量1533，比例0.0511
Pointy_Nose:正样本数量9506，比例0.3168666666666667
Receding_Hairline:正样本数量2530，比例0.08433333333333333
Rosy_Cheeks:正样本数量3379，比例0.11263333333333334
Sideburns:正样本数量2430，比例0.081
Smiling:正样本数量14092，比例0.46973333333333334
Straight_Hair:正样本数量6444，比例0.2148
Wavy_Hair:正样本数量10723，比例0.3574333333333333
Wearing_Earrings:正样本数量7944，比例0.2648
Wearing_Hat:正样本数量1070，比例0.035666666666666666
Wearing_Lipstick:正样本数量16859，比例0.5619666666666666
Wearing_Necklace:正样本数量5085，比例0.1695
Wearing_Necktie:正样本数量2162，比例0.07206666666666667
Young:正样本数量23368，比例0.7789333333333334

1. 5_o_Clock_Shadow - 五点钟阴影（指脸上的胡须阴影）
2. Arched_Eyebrows - 拱形眉毛
3. Attractive - 有吸引力的
4. Bags_Under_Eyes - 眼袋
5. Bald - 秃顶的
6. Bangs - 刘海
7. Big_Lips - 大嘴唇
8. Big_Nose - 大鼻子
9. Black_Hair - 黑发
10. Blond_Hair - 金发
11. Blurry - 模糊的
12. Brown_Hair - 棕发
13. Bushy_Eyebrows - 浓密的眉毛
14. Chubby - 圆胖的
15. Double_Chin - 双下巴
16. Eyeglasses - 眼镜
17. Goatee - 山羊胡
18. Gray_Hair - 灰发
19. Heavy_Makeup - 浓妆
20. High_Cheekbones - 高颧骨
21. Male - 男性
22. Mouth_Slightly_Open - 微张的嘴
23. Mustache - 小胡子
24. Narrow_Eyes - 窄眼睛
25. No_Beard - 没有胡须
26. Oval_Face - 椭圆形脸
27. Pale_Skin - 苍白的皮肤
28. Pointy_Nose - 尖鼻子
29. Receding_Hairline - 后退的发际线
30. Rosy_Cheeks - 红润的脸颊
31. Sideburns - 鬓角
32. Smiling - 微笑
33. Straight_Hair - 直发
34. Wavy_Hair - 波浪发
35. Wearing_Earrings - 戴耳环
36. Wearing_Hat - 戴帽子
37. Wearing_Lipstick - 涂口红
38. Wearing_Necklace - 戴项链
39. Wearing_Necktie - 戴领带
40. Young - 年轻的
"""

