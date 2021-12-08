import os
import shutil

label_dir = '../data/labels_no_chinese/'
image_dir = '/data/zzengae/jwwang/final_project/formula_images_processed/'
output_dir ='/data/zzengae/jwwang/final_project/formula_images/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

label_name_list = os.listdir(label_dir)

for i in range(len(label_name_list)):
    label_name_list[i] = label_name_list[i][:-4]

# print(label_list)

image_name_list = os.listdir(image_dir)

for image_name in image_name_list:
    if image_name[:-4] in label_name_list:
        #根据图片找formula可能找不到 有对应一个or多个的formula才选入图片
        print(image_name)
        shutil.copy(image_dir + image_name, output_dir + image_name)