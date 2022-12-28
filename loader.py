from cProfile import label
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import csv
import re
import scipy.io as scio
import torch.nn.functional as F
from icecream import  ic



class fMRI_Hidden_Dataset(Dataset):

    def __init__(self, feature_path, exp_path, mode='AD_NC'):
        file_list = []
        feature_list = os.listdir(feature_path)
        with open(exp_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0]:
                    filename = 'ROISignals_Sub_' + row[0]
                    if filename in feature_list:
                        filepath = os.path.join(feature_path, filename)
                        file_list.append(filepath)
        print("Classification mode: {}, total {} samples.".format(mode, len(file_list)))
        self.file_list = file_list
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = torch.load(self.file_list[idx])
        feature = data['feature']
        adj = data['adj']
        adj_discrete = False
        if adj_discrete:
            adj = adj[adj > np.mean(adj)]
        if self.mode == 'MCI_NC':
            label = int(data['label'].item())
        else:
            label = round(int(data['label'].item()) / 2)
        return {'feature': feature.detach().numpy(), 'label': np.array(label), 'adj': adj.detach().numpy()}


class mean_bold_Dataset(Dataset):
    def __init__(self, feature_path, exp_path, mode='AD_NC'):
        file_list = []
        label_list = []
        feature_list = os.listdir(feature_path)
        with open(exp_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0]:
                    filename = 'ROISignals_Sub_' + row[0] + '.mat'
                    if filename in feature_list:
                        filepath = os.path.join(feature_path, filename)
                        file_list.append(filepath)
                        if row[2] == 'AD':
                            label_list.append(0)
                        elif row[2] == 'NC':
                            label_list.append(1)
        """file_list = []
        label_list = []
        feature_list = os.listdir(feature_path)
        with open(exp_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0]:
                    filename = 'Sub_' + row[0] + '.pt'
                    if filename in feature_list:
                        filepath = os.path.join(feature_path, filename)
                        file_list.append(filepath)
                        if row[2] == 'AD':
                            label_list.append(0)
                        elif row[2] == 'NC':
                            label_list.append(1)"""

        print("Classification mode: {}, total {} samples.".format(mode, len(file_list)))
        self.file_list = file_list
        self.label_list = label_list
        self.mode = mode

        """with open(exp_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0]:
                    if ('fMRI' in os.path.basename(feature_path)) \
                            or ('feature_AGNN' in os.path.basename(feature_path)) \
                            or ('AGNN_feature' in os.path.basename(feature_path)):
                        filename = re.sub('Sub', 'Sub_', row[1])
                        filename = re.sub('nii', 'npy', filename)
                    else:
                        filename = 'Sub_' + row[0] + '.npy'
                    if filename in feature_list:
                        filepath = os.path.join(feature_path, filename)
                        file_list.append(filepath)
                        if row[2] == 'AD':
                            label_list.append(2)
                        elif row[2] == 'MCI':
                            label_list.append(1)
                        elif row[2] == 'NC':
                            label_list.append(0)"""

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        sample_data = scio.loadmat(self.file_list[idx])
        ROI_signals = sample_data['ROISignals']
        ROI_signals = ROI_signals.T
        mean_sig = np.mean(ROI_signals)
        std_sig = np.std(ROI_signals)
        normal_ROI_signals = (ROI_signals - mean_sig) / std_sig
        ROI_signals = normal_ROI_signals

        adj = np.corrcoef(ROI_signals)
        adj = torch.from_numpy(adj)

        labels = self.label_list[idx]

        return {'feature': torch.Tensor(ROI_signals), 'label': labels, 'adj': torch.Tensor(adj)}





# class fMRI_REHO_ALFF_Dataset(Dataset):

#     def __init__(self, feature_dir,exp_path, cishu, phase = 'ALFF', mode='AD_NC',state='train'):
#         self.mode = mode
#         feature_list = []
#         if (phase == 'ALFF_REHO') or (phase == 'REHO_ALFF'):
#             for i in range(1,91):
#                 feature_name_AD_NC = 'roi-'+str(i)+'AD+NC'+'cishu'+str(cishu)+'.npz'
#                 feature_path_AD_NC_alff = os.path.join(feature_dir,'ALFF',feature_name_AD_NC)
#                 feature_path_AD_NC_reho = os.path.join(feature_dir,'REHO',feature_name_AD_NC)
#                 data_AD_NC_alff = np.load(feature_path_AD_NC_alff, allow_pickle=True)['data'][slice_AD_NC,:]
#                 data_AD_NC_reho = np.load(feature_path_AD_NC_reho, allow_pickle=True)['data'][slice_AD_NC,:]
#                 data_AD_NC = np.hstack((data_AD_NC_reho,data_AD_NC_alff))
#                 feature_name_AD_MCI = 'roi-'+str(i)+'AD+MCI'+'cishu'+str(cishu)+'.npz'
#                 feature_path_AD_MCI_alff = os.path.join(feature_dir,'ALFF',feature_name_AD_MCI)
#                 feature_path_AD_MCI_reho = os.path.join(feature_dir,'REHO',feature_name_AD_MCI)
#                 data_AD_MCI_alff = np.load(feature_path_AD_MCI_alff, allow_pickle=True)['data'][191:286,:]
#                 data_AD_MCI_reho = np.load(feature_path_AD_MCI_reho, allow_pickle=True)['data'][191:286,:]
#                 data_AD_MCI = np.hstack((data_AD_MCI_reho,data_AD_MCI_alff))
#                 data = np.vstack((data_AD_NC,data_AD_MCI))
#                 feature_list.append(data)
#         else:
#             for i in range(1,91):
#                 feature_name_AD_NC = 'roi-'+str(i)+'AD+NC'+'cishu'+str(cishu)+'.npz'
#                 feature_path_AD_NC = os.path.join(feature_dir,phase,feature_name_AD_NC)
#                 data_AD_NC = np.load(feature_path_AD_NC, allow_pickle=True)['data'][slice_AD_NC,:]
#                 feature_name_AD_MCI = 'roi-'+str(i)+'AD+MCI'+'cishu'+str(cishu)+'.npz'
#                 feature_path_AD_MCI = os.path.join(feature_dir,phase,feature_name_AD_MCI)
#                 data_AD_MCI = np.load(feature_path_AD_MCI, allow_pickle=True)['data'][191:286,:]
#                 data = np.vstack((data_AD_NC,data_AD_MCI))
#                 feature_list.append(data)
#         feature_list = np.array(feature_list)
#         label_list = np.hstack((np.ones(149)*2,np.zeros(102),np.ones(95)))
#         # self.label_list = np.hstack((np.ones(191)*2,np.ones(293)))

#         exp_sub_list = []
#         with open(exp_path,'r') as csvfile:
#             reader = csv.reader(csvfile)
#             for row in reader:
#                 if row[0]:
#                     exp_sub_list.append(int(row[0]))
#         if state == 'test':
#             exp_sub_list = exp_sub_list[26:34]
#         self.label_list = label_list[exp_sub_list]
#         self.feature_list = feature_list[:,exp_sub_list,:]
#         # exp_path = os.path.join(feature_dir,'AD+NC'+state+'_mask_list.npy')
#         # sub_list = np.load(exp_path,allow_pickle=True)[cishu]
#         # self.sub_split_list = [idx for idx in sub_list if idx in slice_AD_NC]
#         # self.feature_list = self.feature_list[:,sub_list,:]
#         # self.label_list = self.label_list[sub_list]s
#         print("Classification mode: {}, total {} samples.".format(mode,len(self.label_list)))


#         # print("Classification mode: {}, total {} samples.".format(mode,len(self.label_list)))

#     def __len__(self):
#         return len(self.label_list)

#     def __getitem__(self, idx):
#         feature = self.feature_list[:,idx,:].squeeze()
#         adj = np.corrcoef(feature)
#         adj_discrete=False
#         if adj_discrete:
#             adj = adj[adj>np.mean(adj)]
#         if self.mode == 'MCI_NC' or self.mode == 'AD_MCI_NC':
#             label = round(self.label_list[idx])
#         else:
#             label = round(self.label_list[idx]/2)
#         return {'feature':feature, 'label':np.array(label), 'adj': adj}

class fMRI_REHO_ALFF_Dataset(Dataset):

    def __init__(self, feature_dir, exp_path, cishu, phase='ALFF', mode='AD_NC', state='train'):
        self.mode = mode
        path_mode = re.sub('_', '+', mode)
        feature_name = 'roi-1' + path_mode + 'cishu' + str(cishu) + '.npz'
        sub_num = np.load(os.path.join(feature_dir, phase, feature_name), allow_pickle=True)['data'].shape[0]
        feature_list = np.zeros((sub_num, 90, 64))
        if (phase == 'ALFF_REHO') or (phase == 'REHO_ALFF'):
            for i in range(1, 91):
                feature_name = 'roi-' + str(i) + path_mode + 'cishu' + str(cishu) + '.npz'
                feature_path_alff = os.path.join(feature_dir, 'ALFF', feature_name)
                feature_path_reho = os.path.join(feature_dir, 'REHO', feature_name)
                data_alff = np.load(feature_path_alff, allow_pickle=True)['data']
                data_reho = np.load(feature_path_reho, allow_pickle=True)['data']
                data = np.hstack((data_reho, data_alff))
                feature_list[:, i - 1, :] = data
        else:
            for i in range(1, 91):
                feature_name = 'roi-' + str(i) + path_mode + 'cishu' + str(cishu) + '.npz'
                feature_path = os.path.join(feature_dir, phase, feature_name)
                data = np.load(feature_path, allow_pickle=True)['data']
                feature_list[:, i - 1, :] = data
        if mode == 'AD_NC':
            label_list = np.hstack((np.ones(191) * 2, np.zeros(167)))
            slice_list = np.hstack((np.arange(149), np.arange(191, 293)))
        elif mode == 'AD_MCI':
            label_list = np.hstack((np.ones(191) * 2, np.ones(160)))
            slice_list = np.hstack((np.arange(149), np.arange(191, 286)))
        else:
            label_list = np.hstack((np.zeros(167), np.ones(160)))
            slice_list = np.hstack((np.arange(102), np.arange(167, 262)))

        exp_path = os.path.join(feature_dir, path_mode + state + '_mask_list.npy')
        sub_list = np.load(exp_path, allow_pickle=True)[cishu]
        sub_list = [idx for idx in sub_list if idx in slice_list]
        self.feature_list = feature_list[sub_list, :]
        self.label_list = label_list[sub_list]
        print(feature_list.shape, label_list.shape)
        print("Classification mode: {}, total {} samples.".format(mode, len(self.label_list)))

        # print("Classification mode: {}, total {} samples.".format(mode,len(self.label_list)))

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        feature = self.feature_list[idx, :, :].squeeze()
        adj = np.corrcoef(feature)
        adj_discrete = False
        if adj_discrete:
            adj = adj[adj > np.mean(adj)]
        if (self.mode == 'NC_MCI') or (self.mode == 'AD_MCI_NC') or (self.mode == 'MCI_NC'):
            label = round(self.label_list[idx])
        else:
            label = round(self.label_list[idx] / 2)
        return {'feature': feature, 'label': np.array(label), 'adj': adj}


if __name__ == "__main__":
    feature_path = '/home/ding/exp_4/data/AGNN_feature_set/2022-07-12_03-58-46/ch50_chll50_nu66/Sub_Dataset'
    # feature_path = '/home/ding/exp_4/data/AGNN_MoCo'
    # feature_path = '/home/ding/exp_4/data/fMRI_Sub_avg_norm'
    # feature_path = '/home/ding/exp_4/data/fMRI_feature/feature_AGNN'
    # feature_path = '/home/ding/exp_4/data/MoCo_KMeans/MoCo_Sub_dataset'
    # feature_path = '/home/ding/exp_4/data/fMRI_norm_DeepDPM/fMRI_DeepDPM_aal_output/AGNN_feature'
    # feature_path = '/home/ding/exp_4/data/脑区图(AGNN)_DTI+fMRI/hiddenfMRI'

    # feature_path = '/home/ding/exp_4/data/MoCo_KMeans/cluster2_Sub_dataset'
    # feature_path='/home/ding/exp_4/data/MoCo_KMeans/GMM_cluster10_Sub_dataset'
    feature_path = '/home/ding/exp_4/data/MoCo_KMeans/Birch_cluster10_Sub_dataset'
    # exp_path = '/home/ding/exp_4/data/fold_split_MCI_NC/exp_0.csv'
    exp_train_path = '/home/ding/exp_4/data/fold_split_AD_NC/trainset_0.csv'
    exp_test_path = '/home/ding/exp_4/data/fold_split_AD_NC/testset_0.csv'
    # exp_train_path='/home/ding/exp_4/data/fold_split_AD_MCI/trainset_1.csv'
    # exp_test_path='/home/ding/exp_4/data/fold_split_AD_MCI/testset_1.csv'

    # testset = fMRI_Avg_Dataset(feature_path, exp_test_path, mode='AD_NC')
    # trainset = fMRI_Avg_Dataset(feature_path, exp_train_path, mode='AD_NC')
    # testset = fMRI_Hidden_Dataset(feature_path,exp_test_path,mode='MCI_NC')
    # trainset = fMRI_Hidden_Dataset(feature_path,exp_train_path,mode='MCI_NC')
    cishu = 0
    mode = 'NC_MCI'
    phase = 'REHO'
    state = 'train'
    # trainset = fMRI_REHO_ALFF_Dataset(feature_path,exp_train_path,cishu,phase,mode,'train')
    # testset = fMRI_REHO_ALFF_Dataset(feature_path,exp_test_path,cishu,phase,mode,'test')

    # print(len(trainset))
    # print(len(testset))
    # for i in range(40):
    #     print(testset[i]['feature'].shape,testset[i]['adj'].shape,testset[i]['label'])
    # for i in range(43):
    #     print(trainset[i]['feature'].shape,trainset[i]['adj'].shape,trainset[i]['label'])

    # print(testset[0]['feature'].shape, testset[0]['adj'].shape, testset[0]['label'])
    # print(testset[20]['feature'].shape, testset[20]['adj'].shape, testset[20]['label'])
    # print(testset[0]['adj'])
    # print(testset[47]['adj'])
    # print(testset[48]['adj'])

    # import numpy as np
    # from sklearn import svm
    # from sklearn.model_selection import train_test_split as ts
    # slice_AD_NC = np.hstack((np.arange(149),np.arange(191,293)))
    # feature_path = '/home/ding/exp_4/data/脑区图(AGNN)_DTI+fMRI/hiddenfMRI'
    # feature_dir = feature_path
    # feature_list = np.zeros((251,90,64))
    # cishu = 20
    # for i in range(1,91):
    #     feature_name_AD_NC = 'roi-'+str(i)+'AD+NC'+'cishu'+str(cishu)+'.npz'
    #     feature_path_AD_NC = os.path.join(feature_dir,phase,feature_name_AD_NC)
    #     data_AD_NC = np.load(feature_path_AD_NC, allow_pickle=True)['data'][slice_AD_NC,:]
    #     # feature_name_AD_MCI = 'roi-'+str(i)+'AD+MCI'+'cishu'+str(cishu)+'.npz'
    #     # feature_path_AD_MCI = os.path.join(feature_dir,phase,feature_name_AD_MCI)
    #     # data_AD_MCI = np.load(feature_path_AD_MCI, allow_pickle=True)['data'][191:286,:]
    #     # data = np.vstack((data_AD_NC,data_AD_MCI))
    #     # feature_list.append(data)
    #     feature_list[:,i-1,:] = data_AD_NC
    # feature_list = feature_list.reshape((251,90*64))
    # label_list = np.hstack((np.ones(149)*2,np.zeros(102)))
    # print('Total feature shape {} label shape {}'
    #       .format(feature_list.shape,label_list.shape))

    # exp_path_train = os.path.join(feature_dir,'AD+NCtrain_mask_list.npy')
    # sub_list_train = np.load(exp_path_train,allow_pickle=True)[cishu]
    # sub_split_list_train = [idx for idx in sub_list_train if idx in slice_AD_NC]
    # feature_list = feature_list[sub_split_list_train,:]
    # label_list = label_list[sub_split_list_train]
    # print('feature shape {} label shape {}'.format(feature_list.shape,label_list.shape))

    # ratio = 0.9
    # X_train, X_test, Y_train, Y_test = ts(feature_list,label_list,test_size=ratio)
    # print('Train set shape of {} train label shape of {} \nTest set shape of {} Test label shape of {}'
    #       .format(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape))

    # classifier = svm.SVC()
    # classifier.fit(X_train,Y_train)
    # for i in range(17):
    #     predict = classifier.predict(X_test[i,np.newaxis])
    #     print(i,X_test[i],predict,Y_test[i])
    # print('cishu {}\ttest size of {}\tclassification avg score {}'.format(cishu,ratio,classifier.score(X_test,Y_test)))

    # AD_list = []
    # MCI_list = []
    # for i in range(len(trainset)):
    #     if trainset[i]['label'] == 1:
    #         AD_list.append(trainset[i]['adj'])
    #     else:
    #         MCI_list.append(trainset[i]['adj'])
    # for j in range(len(testset)):
    #     if testset[j]['label'] == 1:
    #         AD_list.append(testset[j]['adj'])
    #     else:
    #         MCI_list.append(testset[j]['adj'])
    # AD_list = np.array(AD_list)
    # MCI_list = np.array(MCI_list)
    # print(AD_list.shape,MCI_list.shape)
    # AD_avg = np.mean(AD_list,axis=0)
    # MCI_avg = np.mean(MCI_list,axis=0)
    # print(AD_avg.shape,MCI_avg.shape)
    # # np.save('/home/ding/exp_4/test/debug/AD_adj_mean.npy',AD_avg)
    # np.save('/home/ding/exp_4/test/debug/NC_adj_mean.npy',MCI_avg)
