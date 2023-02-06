
from doctest import testfile
import os
import glob
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
import scipy.ndimage
from datetime import datetime
import random
import pdb
from sklearn.model_selection import GroupKFold

seed = 10
np.random.seed(seed)


# preprocess subject label and data
xl_path = '/media/andjela/SeagatePor1/CP/Calgary_Preschool_Dataset_Updated_20200213_copy.xlsx'      
path_dict = '/media/andjela/SeagatePor1/CP/PatientDict.txt'  
data_path = '/media/andjela/SeagatePor1/CP/RigidReg_1.0/images/'
df = pd.read_excel(xl_path)
df_subid = pd.read_fwf(path_dict)
# df_raw = pd.read_csv(path_dict, usecols=['PTID', 'DX_bl', 'DX', 'EXAMDATE', 'AGE'])


# load label, age, image paths
'''
struct subj_data

age: baseline age,
label: label for the subject, 0 - NC, 2 - AD, 3 - sMCI, 4 - pMCI
label_all: list of labels for each timestep, 0 - NC, 1 - MCI, 2 - AD
date_start: baseline date, in datetime format
date: list of dates, in datetime format
date_interval: list of intervals, in year
img_paths: list of image paths
'''

img_paths = glob.glob(f'{data_path}*/*.nii.gz') 
img_paths = sorted(img_paths)
# print(img_paths)
subj_data = {}
label_dict = {'Normal': 0, 'NC': 0, 'CN': 0, 'MCI': 1, 'LMCI': 1, 'EMCI': 1, 'AD': 2, 'Dementia': 2, 'sMCI':3, 'pMCI':4}
nan_label_count = 0
nan_idx_list = []
for img_path in img_paths:
    subj_id = os.path.basename(img_path).replace('.nii.gz', '')
    # print(subj_id)
    # date = os.path.basename(img_path).split('-')[1] + '-' + os.path.basename(img_path).split('-')[2] + '-' + os.path.basename(img_path).split('-')[3].split('_')[0]
    # date_struct = datetime.strptime(date, '%Y-%m-%d')
    rows = df.loc[(df['ScanID'] == subj_id)]
    if rows.shape[0] == 0:
        print('Missing label for', subj_id)
    else:
        # matching date
        # date_diff = []
        # for i in range(rows.shape[0]):
        #     date_struct_now = datetime.strptime(rows.iloc[i]['EXAMDATE'], '%Y-%m-%d')
        #     date_diff.append(abs((date_struct_now - date_struct).days))
        # i = np.argmin(date_diff)
        # if date_diff[i] > 120:
        #     print('Missing label for', subj_id, date_diff[i], date_struct)
        #     continue

        # build dict
        # Retrieves only value in series when using iloc[0]
        if subj_id not in subj_data:
            subj_data[subj_id] = {'age': rows.loc[(df['ScanID'] == subj_id)]['Age (Years)'].iloc[0], 'sex': rows.loc[(df['ScanID'] == subj_id)]['Biological Sex (Female = 0; Male = 1)'].iloc[0], 'handedness': rows.loc[(df['ScanID'] == subj_id)]['Handedness'].iloc[0], 'img_paths': []}

        # Accessing specific elements
        # if 'Both' in subj_data[subj_id]['handedness']:
        #     print(f'{subj_id} is both-handed')

     
        # if rows.iloc[i]['EXAMDATE'] in subj_data[subj_id]['date']:
        #     print('Multiple image at same date', subj_id, rows.iloc[i]['EXAMDATE'])
        #     continue

        # subj_data[subj_id]['date'].append(rows.iloc[i]['EXAMDATE'])
        # subj_data[subj_id]['date_interval'].append((date_struct - subj_data[subj_id]['date_start']).days / 365.)
        subj_data[subj_id]['img_paths'].append(os.path.basename(img_path))
        # if pd.isnull(rows.iloc[i]['DX']) == False:
        #     subj_data[subj_id]['label_all'].append(label_dict[rows.iloc[i]['DX']])
        # else:
        #     nan_label_count += 1
        #     nan_idx_list.append([subj_id, len(subj_data[subj_id]['label_all'])])
        #     subj_data[subj_id]['label_all'].append(-1)

# fill nan
# print('Number of nan label:', nan_label_count)
# for subj in nan_idx_list:
#     subj_data[subj[0]]['label_all'][subj[1]] = subj_data[subj[0]]['label_all'][subj[1]-1]
#     if subj_data[subj[0]]['label_all'][subj[1]] == -1:
#         print(subj)

# # get sMCI, pMCI labels
# num_ts_nc = 0
# num_ts_ad = 0
# num_ts_mci = 0
# num_nc = 0
# num_ad = 0
# num_smci = 0
# num_pmci = 0
# subj_list_dict = {'NC':[], 'sMCI':[], 'pMCI': [], 'AD': []}
# for subj_id in subj_data.keys():
#     if len(list(set(subj_data[subj_id]['label_all']))) != 1:    # have NC/MCI/AD mix in timesteps
#         print(subj_id, subj_data[subj_id]['label_all'])
#         if list(set(subj_data[subj_id]['label_all'])) == [1,2] or list(set(subj_data[subj_id]['label_all'])) == [2,1] or list(set(subj_data[subj_id]['label_all'])) == [0,1,2]:
#             subj_data[subj_id]['label'] = 4
#             num_pmci += 1
#             subj_list_dict['pMCI'].append(subj_id)
#         elif list(set(subj_data[subj_id]['label_all'])) == [0,1] or list(set(subj_data[subj_id]['label_all'])) == [1,0]:
#             subj_data[subj_id]['label'] = 3
#             num_smci += 1
#             subj_list_dict['sMCI'].append(subj_id)
#         elif list(set(subj_data[subj_id]['label_all'])) == [0,2] or list(set(subj_data[subj_id]['label_all'])) == [2,0]:
#             subj_data[subj_id]['label'] = 2
#             num_ad += 1
#             subj_list_dict['AD'].append(subj_id)
#     elif subj_data[subj_id]['label'] == 1:  # sMCI
#         subj_data[subj_id]['label'] = 3
#         num_smci += 1
#         subj_list_dict['sMCI'].append(subj_id)
#     elif subj_data[subj_id]['label'] == 0:  # NC
#         num_nc += 1
#         subj_list_dict['NC'].append(subj_id)
#     else:
#         num_ad += 1
#         subj_list_dict['AD'].append(subj_id)
#     label_all = np.array(subj_data[subj_id]['label_all'])
#     num_ts_nc += (label_all==0).sum()
#     num_ts_mci += (label_all==1).sum()
#     num_ts_ad += (label_all==2).sum()
# print('Number of timesteps, NC/MCI/AD:', num_ts_nc, num_ts_mci, num_ts_ad)
# print('Number of subject, NC/sMCI/pMCI/AD:', num_nc, num_smci, num_pmci, num_ad)

# # save subj_list_dict to npy
# np.save('/data/jiahong/data/ADNI/ADNI_longitudinal_subj.npy', subj_list_dict)

# statistics about timesteps
# max_timestep = 0
# num_cls = [0,0,0,0,0]
# num_ts = [0,0,0,0,0,0,0,0,0]
# counts = np.zeros((5, 8))
# for subj_id, info in subj_data.items():
#     num_timestep = len(info['img_paths'])
#     if len(info['label_all']) != num_timestep or len(info['date_interval']) != num_timestep:
#         print('Different number of timepoint', subj_id)
#     max_timestep = max(max_timestep, num_timestep)
#     num_cls[info['label']] += 1
#     num_ts[num_timestep] += 1
#     counts[info['label'], num_timestep] += 1
print('Number of scans: ', len(subj_data))
# print('Max number of timesteps: ', max_timestep)
# print('Number of each timestep', num_ts)
# print('Number of each class', num_cls)
# print('NC', counts[0])
# print('sMCI', counts[3])
# print('pMCI', counts[4])
# print('AD', counts[2])

# counts_cum = counts.copy()
# for i in range(counts.shape[1]-2, 0, -1):
#     counts_cum[:, i] += counts_cum[:, i+1]
# print(counts_cum)

# save subj_data to h5
h5_noimg_path = '/media/andjela/SeagatePor1/CP/CP_longitudinal_noimg.h5'
if not os.path.exists(h5_noimg_path):
    f_noimg = h5py.File(h5_noimg_path, 'a')
    for i, subj_id in enumerate(subj_data.keys()):
        subj_noimg = f_noimg.create_group(subj_id)
        # subj_noimg.create_dataset('label', data=subj_data[subj_id]['label'])
        # subj_noimg.create_dataset('label_all', data=subj_data[subj_id]['label_all'])
        # subj_noimg.create_dataset('date_start', data=subj_data[subj_id]['date_start'])
        # subj_noimg.create_dataset('date_interval', data=subj_data[subj_id]['date_interval'])
        subj_noimg.create_dataset('age', data=subj_data[subj_id]['age'])
        subj_noimg.create_dataset('sex', data=subj_data[subj_id]['sex'])
        subj_noimg.create_dataset('handedness', data=subj_data[subj_id]['handedness'])
        # subj_noimg.create_dataset('img_paths', data=subj_data[subj_id]['img_paths'])

# save images to h5
h5_img_path = '/media/andjela/SeagatePor1/CP/CP_longitudinal_img.h5'
if not os.path.exists(h5_img_path):
    f_img = h5py.File(h5_img_path, 'a')
    for i, subj_id in enumerate(subj_data.keys()):
        subj_img = f_img.create_group(subj_id)
        img_paths = subj_data[subj_id]['img_paths']
        for img_path in img_paths:
            img_nib = nib.load(os.path.join(data_path,img_path))
            img = img_nib.get_fdata()
            img = (img - np.mean(img)) / np.std(img) # Need this normalization???
            subj_img.create_dataset(os.path.basename(img_path), data=img)
        print(i, subj_id)

# def augment_image(img, rotate, shift, flip):
#     # pdb.set_trace()
#     img = scipy.ndimage.interpolation.rotate(img, rotate[0], axes=(1,0), reshape=False)
#     img = scipy.ndimage.interpolation.rotate(img, rotate[1], axes=(0,2), reshape=False)
#     img = scipy.ndimage.interpolation.rotate(img, rotate[2], axes=(1,2), reshape=False)
#     img = scipy.ndimage.shift(img, shift[0])
#     if flip[0] == 1:
#         img = np.flip(img, 0) - np.zeros_like(img)
#     return img

# h5_img_path = '/data/jiahong/data/ADNI/ADNI_longitudinal_img_aug.h5'
# aug_size = 10
# if not os.path.exists(h5_img_path):
#     f_img = h5py.File(h5_img_path, 'a')
#     for i, subj_id in enumerate(subj_data.keys()):
#         subj_img = f_img.create_group(subj_id)
#         img_paths = subj_data[subj_id]['img_paths']
#         rotate_list = np.random.uniform(-2, 2, (aug_size-1, 3))
#         shift_list =  np.random.uniform(-2, 2, (aug_size-1, 1))
#         flip_list =  np.random.randint(0, 2, (aug_size-1, 1))
#         for img_path in img_paths:
#             img_nib = nib.load(os.path.join(data_path,img_path))
#             img = img_nib.get_fdata()
#             img = (img - np.mean(img)) / np.std(img)
#             imgs = [img]
#             for j in range(aug_size-1):
#                 imgs.append(augment_image(img, rotate_list[j], shift_list[j], flip_list[j]))
#             imgs = np.stack(imgs, 0)
#             subj_img.create_dataset(os.path.basename(img_path), data=imgs)
#         print(i, subj_id)

def FindPair(name):
    #Outputs patient number, moving and fixed image scanID as strings for further analysis
    #Possible folder name pairs are below with each string differing in length
    # name = '10006_CL_Dev_004_CL_Dev_008'
    # name1 = 'CL_Dev_004_PS15_048'
    # name2 = 'PS15_048_CL_Dev_004'
    # name3 = 'PS15_048_PS17_017'

    sub_number = name[:5]
    
    #idx contains a list of strings of a given name
    
    idx = [s for s in name[6:].split("_")]


    if len(idx) == 6:
        
        mov = f'{idx[0]}_{idx[1]}_{idx[2]}'
        fix = f'{idx[3]}_{idx[4]}_{idx[5]}'
        return(sub_number, mov, fix)

    elif len(idx) == 5:
        if 'CL' in idx[0]:
            mov = f'{idx[0]}_{idx[1]}_{idx[2]}'
            fix = f'{idx[3]}_{idx[4]}'
            
            return(sub_number, mov, fix)
        elif 'PS' in idx[0]:
            mov = f'{idx[0]}_{idx[1]}'
            fix = f'{idx[2]}_{idx[3]}_{idx[4]}'
            
            return(sub_number, mov, fix)

    elif len(idx) == 4:
        mov = f'{idx[0]}_{idx[1]}'
        fix = f'{idx[2]}_{idx[3]}'
        return(sub_number, mov, fix)

    elif len(idx) == 3:
        mov = f'{idx[0]}'
        fix = f'{idx[1]}_{idx[2]}'
        return(sub_number, mov, fix)


    else:
        print('Not a corresponding folder name', name)

def append_mov_fix(X_list):
    subj_list = []
    # print(X_list)
    for pair_info in X_list:
        sub_number, mov, fix = FindPair(pair_info)
        subj_list.append(mov)
        subj_list.append(fix)

    return subj_list

def GroupedCrossValidationDataPair(PROJECT_DIR, subj_list_postfix, subj_data):

    os.chdir(PROJECT_DIR)
    DATA_PATH = "images"

    images_path = os.path.join(PROJECT_DIR, DATA_PATH)

    # if label == True:
    #     LABEL_PATH = "labels"
    #     labels_path = os.path.join(PROJECT_DIR, LABEL_PATH)

    all_pairs = np.array(sorted(os.listdir(images_path)))
    
    patient_id = [name[:5] for name in all_pairs]
   
    all_patients = sorted(list(set(patient_id)))

    # Modify patient_id to be 0 to 63 instead of 10006 to 10163
    dict = {}
    for i, value in enumerate(all_patients):
        dict[value]=i
    modif_patient_id = []
    for elem in patient_id:
        for key, value in dict.items():
            if elem == key:
                elem = value
                modif_patient_id.append(elem)
    
    group_kfold = GroupKFold(n_splits=5)
    n_splits = group_kfold.get_n_splits(X=all_pairs, groups=modif_patient_id)

    # Iterate through folds
    for fold, (train_index, test_index) in enumerate(group_kfold.split(X=all_pairs, groups=modif_patient_id)):
        X_train, X_test = all_pairs[train_index], all_pairs[test_index]
        # print(len(X_train), len(X_test))
        # Take 10% of test for validation
        X_val = X_test[:int(0.5*len(X_test))]
        # print(len(X_val))
        X_test_rest = sorted(set(X_test) - set(X_val))


        subj_train_list = append_mov_fix(X_train)
        subj_val_list = append_mov_fix(X_val)
        subj_test_list = append_mov_fix(X_test_rest)


        # print(subj_train_list)
        subj_id_list_train, case_id_list_train = get_subj_pair_case_id_list(subj_data, subj_train_list)
        
        subj_id_list_val, case_id_list_val = get_subj_pair_case_id_list(subj_data, subj_val_list)
        subj_id_list_test, case_id_list_test = get_subj_pair_case_id_list(subj_data, subj_test_list)

        save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/fold'+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
        save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/'+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
        save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/'+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)

        

def save_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id+'\n')

# save txt, subj_id, case_id, case_number, case_id, case_number
def save_pair_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+case_id[1]+' '+str(case_id[2])+' '+str(case_id[3])+'\n')

def save_single_data_txt(path, subj_id_list, case_id_list):
    with open(path, 'w') as ft:
        for subj_id, case_id in zip(subj_id_list, case_id_list):
            ft.write(subj_id+' '+case_id[0]+' '+str(case_id[1])+'\n')

def get_subj_pair_case_id_list(subj_data, subj_id_list):
    
    case_id_list_full = []
    
    for subj_id in subj_id_list:
        case_id_list_full.append(subj_data[subj_id]['img_paths'])
        print('HEY')
        # print(len(case_id_list))
        # print('case_id', case_id_list)
    for i in range(len(case_id_list_full)):
        for j in range(i+1, len(case_id_list_full)):
            subj_id_list.append(subj_id)
            case_id_list_full.append([case_id_list_full[i],case_id_list_full[j],i,j])

                # pdb.set_trace()
                # filter out pairs that are too close
                # if subj_data[subj_id]['date_interval'][j] - subj_data[subj_id]['date_interval'][i] >= 2:
                #     subj_id_list_full.append(subj_id)
                #     case_id_list_full.append([case_id_list[i],case_id_list[j],i,j])
    return subj_id_list, case_id_list_full

def get_subj_single_case_id_list(subj_data, subj_id_list):
    subj_id_list_full = []
    case_id_list_full = []
    for subj_id in subj_id_list:
        case_id_list = subj_data[subj_id]['img_paths']
        for i in range(len(case_id_list)):
            subj_id_list_full.append(subj_id)
            case_id_list_full.append([case_id_list[i], i])
    return subj_id_list_full, case_id_list_full

# pdb.set_trace()


# subj_list_postfix = 'NC_AD_pMCI_sMCI'
# subj_list_postfix = 'NC_AD_pMCI_sMCI_far'
# subj_list_postfix = 'NC_AD'
# subj_list_postfix = 'pMCI_sMCI'
# subj_list_postfix = 'NC'
subj_list_postfix = 'CP'


# subj_list_postfix = 'AD_pMCI_sMCI'
# subj_id_all = np.load('/data/jiahong/data/ADNI/ADNI_longitudinal_subj.npy', allow_pickle=True).item()
# subj_id_all = 

res = '1.0'
PROJECT_DIR = f"/media/andjela/SeagatePor1/CP/RigidReg_{res}"


GroupedCrossValidationDataPair(PROJECT_DIR, subj_list_postfix, subj_data)

# for fold in range(5):
#     # for class_name in ['NC', 'AD', 'pMCI', 'sMCI']:
#     # for class_name in ['NC', 'AD']:
#     # for class_name in ['pMCI', 'sMCI']:
#     # for class_name in ['NC']:
#     subj_list = []
#     subj_test_list = []
#     subj_val_list = []
#     subj_train_list = []
#     # for class_name in ['AD', 'pMCI', 'sMCI']:
#     #     class_list = subj_id_all[class_name]
#     #     np.random.shuffle(class_list)
#     #     num_class = len(class_list)

#     #     class_test = class_list[fold*int(0.2*num_class):(fold+1)*int(0.2*num_class)]
#     #     class_train_val = class_list[:fold*int(0.2*num_class)] + class_list[(fold+1)*int(0.2*num_class):]
#     #     class_val = class_train_val[:int(0.1*len(class_train_val))]
#     #     class_train = class_train_val[int(0.1*len(class_train_val)):]
#     #     subj_test_list.extend(class_test)
#     #     subj_train_list.extend(class_train)
#     #     subj_val_list.extend(class_val)

#     if 'single' in subj_list_postfix:
#         subj_id_list_train, case_id_list_train = get_subj_single_case_id_list(subj_data, subj_train_list)
#         subj_id_list_val, case_id_list_val = get_subj_single_case_id_list(subj_data, subj_val_list)
#         subj_id_list_test, case_id_list_test = get_subj_single_case_id_list(subj_data, subj_test_list)

#         save_single_data_txt('../data/ADNI/fold'+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
#         save_single_data_txt('../data/ADNI/fold'+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
#         save_single_data_txt('../data/ADNI/fold'+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)
#     else:
#         subj_id_list_train, case_id_list_train = get_subj_pair_case_id_list(subj_data, subj_train_list)
#         print('id_train', subj_id_list_train)
#         subj_id_list_val, case_id_list_val = get_subj_pair_case_id_list(subj_data, subj_val_list)
#         subj_id_list_test, case_id_list_test = get_subj_pair_case_id_list(subj_data, subj_test_list)

#         save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/fold'+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
#         save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/'+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
#         save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/'+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)
