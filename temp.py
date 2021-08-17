import os
from glob import glob

# img_file_list = glob('./data/food_datasets/food_023/*.jpg')
# img_file_list = img_file_list[:3]
#
# print(len(img_file_list))
# for index, img_file in enumerate(img_file_list):
#     os.rename(img_file, './data/food_datasets/food_023/' + str(index) + '.jpg')

nums1 = [1, 3, 5, 7]
nums2 = [2, 4, 6, 8]

for index, [num1, num2] in enumerate(zip(nums1, nums2)):
    print(index, num1, num2)
