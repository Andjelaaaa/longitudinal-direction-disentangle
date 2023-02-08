
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


# preprocess subject label and data
def preprocess():
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

    # print(img_paths)
    img_paths = sorted(img_paths)
    # print(img_paths)
    subj_data = {}
    label_dict = {'Normal': 0, 'NC': 0, 'CN': 0, 'MCI': 1, 'LMCI': 1, 'EMCI': 1, 'AD': 2, 'Dementia': 2, 'sMCI':3, 'pMCI':4}
    nan_label_count = 0
    nan_idx_list = []
    for img_path in img_paths:
        # subj_id = os.path.basename(img_path).replace('.nii.gz', '')

        path_elements = [s for s in img_path.split("/")]
        pair = path_elements[-2]
        sub_number, mov, fix = FindPair(pair)


        # print(os.path.basename(img_path))
        # print(subj_id)
        # date = os.path.basename(img_path).split('-')[1] + '-' + os.path.basename(img_path).split('-')[2] + '-' + os.path.basename(img_path).split('-')[3].split('_')[0]
        # date_struct = datetime.strptime(date, '%Y-%m-%d')
        # rows = df.loc[(df['PreschoolID'] == sub_number)]
        rows = df.query(f'PreschoolID=={sub_number}')
        # print(rows.to_string())
        if rows.shape[0] == 0:
            print('Missing label for', sub_number)
        else:
            
            # Retrieves only value in series when using iloc[0]
            if sub_number not in subj_data:
                # subj_data[subj_id] = {'sub': rows.loc[(df['ScanID'] == subj_id)]['PreschoolID'].iloc[0], 'age': rows.loc[(df['ScanID'] == subj_id)]['Age (Years)'].iloc[0], 'sex': rows.loc[(df['ScanID'] == subj_id)]['Biological Sex (Female = 0; Male = 1)'].iloc[0], 'handedness': rows.loc[(df['ScanID'] == subj_id)]['Handedness'].iloc[0], 'img_paths': []}
                subj_data[sub_number] = { 'age': [], 'sex': rows.iloc[0]['Biological Sex (Female = 0; Male = 1)'], 'handedness': rows.iloc[0]['Handedness'], 'img_paths': []}


            # Accessing specific elements
            # if 'Both' in subj_data[subj_id]['handedness']:
            #     print(f'{subj_id} is both-handed')

            
            # Consider only scanID once, not the pairs
            if f'{mov}.nii.gz' not in '\t'.join(subj_data[sub_number]['img_paths']):
                subj_data[sub_number]['img_paths'].append(f'{data_path}{pair}/{mov}.nii.gz')
            elif f'{fix}.nii.gz' not in '\t'.join(subj_data[sub_number]['img_paths']):
                subj_data[sub_number]['img_paths'].append(f'{data_path}{pair}/{fix}.nii.gz')
                
            # Consider only age once, not the pairs (where the age repeats)
            if rows.loc[(df['ScanID'] == mov)]['Age (Years)'].iloc[0] not in subj_data[sub_number]['age']:
                subj_data[sub_number]['age'].append(rows.loc[(df['ScanID'] == mov)]['Age (Years)'].iloc[0])
            elif rows.loc[(df['ScanID'] == fix)]['Age (Years)'].iloc[0] not in subj_data[sub_number]['age']:
                subj_data[sub_number]['age'].append(rows.loc[(df['ScanID'] == fix)]['Age (Years)'].iloc[0])
                
            
    print(subj_data['10006']['age'])
    

    
    print('Number of subjects: ', len(subj_data))
    

    # save subj_data to h5
    h5_noimg_path = '/media/andjela/SeagatePor1/CP/data/CP/CP_longitudinal_noimg.h5'
    if not os.path.exists(h5_noimg_path):
        f_noimg = h5py.File(h5_noimg_path, 'a')
        for i, sub_number in enumerate(subj_data.keys()):
            subj_noimg = f_noimg.create_group(sub_number)
            # subj_noimg.create_dataset('label', data=subj_data[subj_id]['label'])
            # subj_noimg.create_dataset('label_all', data=subj_data[subj_id]['label_all'])
            # subj_noimg.create_dataset('date_start', data=subj_data[subj_id]['date_start'])
            # subj_noimg.create_dataset('date_interval', data=subj_data[subj_id]['date_interval'])
            subj_noimg.create_dataset('age', data=subj_data[sub_number]['age'])
            subj_noimg.create_dataset('sex', data=subj_data[sub_number]['sex'])
            subj_noimg.create_dataset('handedness', data=subj_data[sub_number]['handedness'])
            # subj_noimg.create_dataset('img_paths', data=subj_data[sub_number]['img_paths'])

    # save images to h5
    h5_img_path = '/media/andjela/SeagatePor1/CP/data/CP/CP_longitudinal_img.h5'
    if not os.path.exists(h5_img_path):
        f_img = h5py.File(h5_img_path, 'a')
        for i, sub_number in enumerate(subj_data.keys()):
            subj_img = f_img.create_group(sub_number)
            img_paths = subj_data[sub_number]['img_paths']
            for img_path in img_paths:
                # Changed path to images
                img_nib = nib.load(os.path.join(img_path))
                img = img_nib.get_fdata()
                img = (img - np.mean(img)) / np.std(img) # Need this normalization???
                subj_img.create_dataset(os.path.basename(img_path), data=img)
            # print(i, sub_number)

    return subj_data


def append_mov_fix(X_list):
    subj_list = []
    for pair_info in X_list:
        sub_number, mov, fix = FindPair(pair_info)
        subj_list.append(sub_number)

    return list(np.unique(np.array(subj_list)))

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
        # Take 10% of test for validation
        X_val = X_test[:int(0.5*len(X_test))]
        X_test_rest = sorted(set(X_test) - set(X_val))


        subj_train_list = append_mov_fix(X_train)
        # print('subj_train_list', len(subj_train_list))
        subj_val_list = append_mov_fix(X_val)
        # print('subj_val_list', len(subj_train_list))
        subj_test_list = append_mov_fix(X_test_rest)
        # print('subj_test_list', len(subj_train_list))

        subj_id_list_train, case_id_list_train = get_subj_pair_case_id_list(subj_data, subj_train_list)
        subj_id_list_val, case_id_list_val = get_subj_pair_case_id_list(subj_data, subj_val_list)
        subj_id_list_test, case_id_list_test = get_subj_pair_case_id_list(subj_data, subj_test_list)

        save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/fold'+str(fold)+'_train_' + subj_list_postfix + '.txt', subj_id_list_train, case_id_list_train)
        save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/fold'+str(fold)+'_val_' + subj_list_postfix + '.txt', subj_id_list_val, case_id_list_val)
        save_pair_data_txt('/media/andjela/SeagatePor1/CP/data/CP/fold'+str(fold)+'_test_' + subj_list_postfix + '.txt', subj_id_list_test, case_id_list_test)

        

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
    subj_id_list_full = []

    for sub_number in subj_id_list:
        case_id_list = subj_data[sub_number]['img_paths']
        # print(sub_number, len(case_id_list))
        for i in range(len(case_id_list)):
            for j in range(i+1, len(case_id_list)):
                subj_id_list_full.append(sub_number)
                case_id_list_full.append([case_id_list[i],case_id_list[j],i,j])
    
    return subj_id_list_full, case_id_list_full

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


subj_list_postfix = 'CP'

res = '1.0'
PROJECT_DIR = f"/media/andjela/SeagatePor1/CP/RigidReg_{res}"

subj_data = preprocess()
GroupedCrossValidationDataPair(PROJECT_DIR, subj_list_postfix, subj_data)