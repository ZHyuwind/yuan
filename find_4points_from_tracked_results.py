import numpy as np
import os
import json
import matplotlib.pyplot as plt
import cv2



def get_file(path):
    with open(path, 'r') as f:
        file = json.load(f)
        # print(file['199.jpg'])
        file_name = sorted([int(name.split('.')[0]) for name in file.keys()])
        file_name_list = []
        for i in file_name:
            file_name_list.append(str(i) + '.jpg')
    return file,file_name_list

    # #### oneimage-onefile transfer to oneidx-onefile [logistic transform]
def statistic_idx(file):
    #file,file_name_list = get_file(path)
    index = {}
    for i,img_name in enumerate(list(file.keys())):
        for j in range(len(file[img_name])):

            index[file[img_name][j]['idx']] = index.get(file[img_name][j]['idx'],0)+1

    return index
    # print(index)
def fix_each_idx_keypoints(file):

    index = len(statistic_idx(file))
    idx_keypoints_name = {}
    for i in range(1,index+1):
        idx_keypoints_name[('idx{}'.format(i)+'_keypoints')] = {}

    for j in file.keys():
        n = len(file[j])
        # a = n
        for i in idx_keypoints_name:

            key = int(list(i.split('_')[0])[-1])
            # if a > 0:
            for k in range(n):

                if file[j][k]['idx'] == key:
                    j_name = j.split('.')[0]
                    idx_keypoints_name[i][j_name] = file[j][k]['keypoints']
                        # a = a-1
            # else:
            #     j_name = j.split('.')[0]
            #     idx_keypoints_name[i][j_name] = [0]*17

    return idx_keypoints_name

def statistic_4_points(file):
    '''
    :param file:
    :param file_name_list:
    :return:
    '''
    left_wrist_x = []
    left_wrist_y = []
    right_wrist_x = []
    right_wrist_y = []
    left_ankle_x = []
    left_ankle_y = []
    right_ankle_x = []
    right_ankle_y = []
    idx_keypoints_name = fix_each_idx_keypoints(file)
    # index = len(statistic_idx(file_name_list))
    # matrix = np.zeros((index,8))
    # n = len(matrix)
    for i in idx_keypoints_name:
        lwx = []
        lwy = []
        rwx = []
        rwy = []
        lax = []
        lay = []
        rax = []
        ray = []
        for j in idx_keypoints_name[i].keys():

            j = j.split('.')[0]
            lwx.append(idx_keypoints_name[i][j][33])
            lwy.append(idx_keypoints_name[i][j][34])
            rwx.append(idx_keypoints_name[i][j][36])
            rwy.append(idx_keypoints_name[i][j][37])
            lax.append(idx_keypoints_name[i][j][45])
            lay.append(idx_keypoints_name[i][j][46])
            rax.append(idx_keypoints_name[i][j][48])
            ray.append(idx_keypoints_name[i][j][49])
        left_wrist_x.append(lwx)
        left_wrist_y.append(lwy)
        right_wrist_x.append(rwx)
        right_wrist_y.append(rwy)
        left_ankle_x.append(lax)
        left_ankle_y.append(lay)
        right_ankle_x.append(rax)
        right_ankle_y.append(ray)
    return left_wrist_x,left_wrist_y,right_wrist_x,right_wrist_y,left_ankle_x,left_ankle_y,right_ankle_x,right_ankle_y

def select_main_idx(file,file_name_list):
    lwx,lwy,rwx,rwy,lax,lay,rax,ray = statistic_4_points(file)
    # idx_keypoints_name = fix_each_idx_keypoints(file)

    idx = [i for i in range(len(lwx)) if len(lwx[i])>200]
    Idx = []
    for i in idx:
       Idx.append(max(lwx[i])-min(lwx[i]))
        # # rwx_m = max(rwx[i]) - min(rwx[i])
        # # lax_m = max(lax[i]) - min(lax[i])
        # # rax_m = max(rax[i]) - min(rax[i])

    Idx = np.array(Idx).argsort()[-1]
    return idx[Idx]

def save_main_keypoints(idx_keypoints_name,txt_dir,file):
    idx = select_main_idx(file)
    ### save main idx's keypoints
    with open(txt_dir,'w') as tf:
        name = 'idx{}'.format(idx+1)
        for i in idx_keypoints_name[name].keys():
            tf.write(str(i)+'.jpg'+': '+str(idx_keypoints_name[name][i])+'\n')



def plot_image(path):
    file,file_name_list = get_file(path)
    idx = select_main_idx(file,file_name_list)
    lwx, lwy, rwx, rwy, lax, lay, rax, ray = statistic_4_points(file)
    # plt.title('vault')
    plt.figure(figsize=(24,13.5),dpi=80)

    ax = plt.gca()
    ax.invert_yaxis()
    plt.scatter(lwx[idx], lwy[idx], color='red',label='left_wrist')

    plt.scatter(rwx[idx], rwy[idx], color='blue',label='right_wrist')
    plt.scatter(lax[idx], lay[idx], color='green',label='left_ankle')
    plt.scatter(rax[idx], ray[idx], color='yellow',label='right_ankle')

    plt.legend(loc='upper center')
    plt.show()


if __name__ == '__main__':
    path = '/home/dev/zy/Alphapose/PoseFlow/video_data/jump-results-forvis-tracked.json'
    # path = '/home/dev/zy/Alphapose/three/jump_result/jump-results-forvis-tracked.json'
    txt_dir = '/home/dev/zy/Alphapose/PoseFlow/video_data/jump.txt'
    plot_image(path)
 #frame_name_list == list(file.keys())

