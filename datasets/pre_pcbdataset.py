import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


normal_path = '/home/sxspp/sxspp/chohj/PCB/0'
abnormal_path = '/home/sxspp/sxspp/chohj/PCB/1'

normal_img = glob.glob('/home/sxspp/sxspp/chohj/PCB/0/*.JPG')
abnormal_img = glob.glob('/home/sxspp/sxspp/chohj/PCB/1/*.jpg')

#normal img save

for i in normal_img:
    img = cv2.imread(i)
    # Image load (cv2.imread는 BGR로 load 하기에 RGB로 변환)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 가로, 세로에 대해 부족한 margin 계산
    height, width = image.shape[0:2]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    # 부족한 길이가 절반으로 안 떨어질 경우 +1
    if np.abs(height-width) % 2 != 0:
        margin[0] += 1

    # 가로, 세로 가운데 부족한 쪽에 margin 추가
    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]

    # color 이미지일 경우 color 채널 margin 추가
    if len(image.shape) == 3:
        margin_list.append([0,0])

    # 이미지에 margin 추가
    output = np.pad(image, margin_list, mode='constant')

    resize_img = cv2.resize(output,dsize=(1024, 1024))
    rename = i[30:]
    cv2.imwrite("/home/sxspp/sxspp/chelsea/data/pcb/0/"+rename,resize_img)

#abnormal img save

for i in abnormal_img:
    img = cv2.imread(i)
    # Image load (cv2.imread는 BGR로 load 하기에 RGB로 변환)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 가로, 세로에 대해 부족한 margin 계산
    height, width = image.shape[0:2]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    # 부족한 길이가 절반으로 안 떨어질 경우 +1
    if np.abs(height-width) % 2 != 0:
        margin[0] += 1

    # 가로, 세로 가운데 부족한 쪽에 margin 추가
    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]

    # color 이미지일 경우 color 채널 margin 추가
    if len(image.shape) == 3:
        margin_list.append([0,0])

    # 이미지에 margin 추가
    output = np.pad(image, margin_list, mode='constant')

    resize_img = cv2.resize(output,dsize=(1024, 1024))
    rename = i[30:]
    cv2.imwrite("/home/sxspp/sxspp/chelsea/data/pcb/1/"+rename,resize_img)

    