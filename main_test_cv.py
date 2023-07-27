import os
import numpy as np
import torch
import networks
import transforms
import h5py
import argparse
import pyriemann
from torch import nn, optim
from bin.LoadData import *
from FBCSP import FBCSP_whole
from dataset import eegDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import cross_val_score
from dataset_pick_and_loading import load_BCIIV2a_data_epoch, generate_balanced_index, slice_the_time,\
    load_Korea54_DataByParticipoantPath, load_high_gamma_datasetByParticipoantPath, load_EEG52_DataByParticipantPath,\
    load_fatigue_detection_by_participoant_path, load_full_high_gamma_datasetByParticipoantPath
from pre_processing_methods import automated_ICA, surface_laplacian, bandpass_filtering, baseline_correction
from sklearn.metrics import confusion_matrix, accuracy_score
from FBSCNet import FBSCNet
from Bio_FSTNet import Bio_FSTNet
from SE_FBCNet import SE_FBCNet
from SE_FBCNet_spatial import SE_FBCNet_spatial
import mne
import sys

data_path = r'D:\Pytorch_learning\ravikiran-mane-FBCNet-5dffb8b\data\bci42a\originalData'
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--order','-o',help='The order of the pre-processing methods, required parameters, write like ''2130'' to speciafy the order', required=True, type=str)
parser.add_argument('--model_name','-m',help='The model name that will bes used. For example ''FBCSP''', default= 'FBCNet', type=str)
parser.add_argument('--participant_num','-p',help='The participant number you will use', default= '1', type=str)
parser.add_argument('--data_set','-d',help='The dataset neme will be used in training', default= 'BCIIV2a', type=str)
parser.add_argument('--gpu_num','-g',help='Please sprcify gpu number you want to use', default= '0', type=str, required=True)
args = parser.parse_args()

participant_num = args.participant_num
preprocessing_order = args.order
data_set_id = args.data_set
model_name= args.model_name

device = str("cuda:"+str(args.gpu_num)) if torch.cuda.is_available() else "cpu"
if(data_set_id=='BCIIV2a'):
    data_path = r'F:\EEG_datasets\BCIIV2a'
    epoched_data, y= load_BCIIV2a_data_epoch(data_path, participant_num)
    classes = [0,1,2,3]
    num_channels = 22
    num_of_classes = len(classes)
    freq = 250

if(data_set_id=='high_gamma'):
    data_path = r'F:\EEG_datasets\high_gamma'
    epoched_data, y= load_high_gamma_datasetByParticipoantPath(data_path, participant_num)
    classes = [0,1,2,3]
    num_channels = 8
    num_of_classes = len(classes)
    freq = 250

if(data_set_id=='high_gamma_full'):
    data_path =  r'F:\EEG_datasets\high-gamma-dataset\raw\master\data'
    epoched_data, y= load_full_high_gamma_datasetByParticipoantPath(data_path, participant_num)
    classes = [0,1,2,3]
    num_channels = 128
    num_of_classes = len(classes)
    freq = 250

if(data_set_id=='Koera_54'):
    data_path = r'F:\EEG_datasets\dataset_0550'
    epoched_data, y= load_Korea54_DataByParticipoantPath(data_path, participant_num)
    classes = [0,1]
    num_channels = 62
    num_of_classes = len(classes)
    freq = 250

if(data_set_id=='EEG_52'):
    data_path = r'F:\EEG_datasets\dataset_52'
    epoched_data, y= load_EEG52_DataByParticipantPath(data_path, participant_num)
    classes = [0,1]
    num_channels = 64
    num_of_classes = len(classes)
    freq = 512

if(data_set_id=='Fatigue'):
    data_path = r'C:\Users\Administrator\Desktop\fatigue_detection\data'
    epoched_data, y= load_fatigue_detection_by_participoant_path(data_path, participant_num)
    classes = [0,1]
    num_channels = 36
    num_of_classes = len(classes)
    freq = 1000

for i in preprocessing_order:
    if(i == '0'):
        epoched_data = automated_ICA(epoched_data)
    if(i == '1'):
        epoched_data = bandpass_filtering(epoched_data, Fs = freq)
    if(i == '2'):
        epoched_data = baseline_correction(epoched_data)
    if(i == '3'):
        epoched_data = surface_laplacian(epoched_data)

if(model_name == 'LGG'):
    ch_adj, _ = mne.channels.find_ch_adjacency(epoched_data.info, 'eeg')
    ch_adj = ch_adj.todense()
    ch_adj = torch.tensor(ch_adj, dtype=torch.float).to(device)

print(epoched_data.get_data().shape)
training_rate = 0.75
validation_rate = 0.125
testing_rate = 0.125

all_labels = y.reshape(y.shape[0],1)
total_length = all_labels.shape[0]
all_index = np.arange(0, total_length, 1)
np.random.shuffle(all_index)

time_window_slice = 1000
batch_size = 64
data_all = epoched_data.get_data()
my_loc_channels = np.array(list(epoched_data.info.get_montage().get_positions()['ch_pos'].values()))
step_len = 200

if( model_name == 'FBCSP' or model_name == 'Riemannian'):
    flod_amount = 10
    slice_len = int(total_length / flod_amount)
    path_training_results = data_set_id+'//'+model_name

    if(not os.path.exists(path_training_results)):
        os.makedirs(path_training_results)

    path_training_results_data = path_training_results+'//training_results//participant_'+participant_num
    if(not os.path.exists(path_training_results_data)):
        os.makedirs(path_training_results_data)

    f = open(os.path.join(path_training_results_data, "Pre_processing_order_"+str(preprocessing_order)+".txt"),'a')
    accuries = []
    
    if(model_name == 'FBCSP'):
        for i in range(flod_amount):
            temp_index = all_index
            testing_index = temp_index[i * slice_len: (i+1) * slice_len]
            training_index = np.delete(temp_index, testing_index)
            training_data, training_labels = slice_the_time(data_all, all_labels, training_index, time_window_slice, step_len)
            testing_data, testing_labels = slice_the_time(data_all, all_labels, testing_index, time_window_slice, step_len)
            print(training_labels.shape, training_data.shape, data_all.shape, all_labels.shape)
            training_labels = training_labels.reshape(training_labels.shape[0],)
            testing_labels = testing_labels.reshape(testing_labels.shape[0],)
            accuracy = round(FBCSP_whole(training_data, training_labels, testing_data, testing_labels, freq), 4)
            accuries.append(accuracy)
            f.write('Participant ' + participant_num +' flod '+ str(i+1) + ' result:' + str(accuracy)+'\n')
        accuries = np.asarray(accuries)
        f.write('The pre-processing order is: ' + preprocessing_order + '\n')
        f.write('Participant ' + participant_num + ' average accuracy is: ' + str(round(np.average(accuries),4)) + '\n')

    if(model_name == 'Riemannian'):
        used_data, used_labels = slice_the_time(data_all, all_labels, all_index, time_window_slice,step_len)
        used_labels = used_labels.reshape((used_labels.shape[0]))
        cov = pyriemann.estimation.Covariances().fit_transform(used_data)
        mdm = pyriemann.classification.MDM()
        accuracy = cross_val_score(mdm, cov, used_labels)
        f.write('The pre-processing order is: ' + preprocessing_order + '\n')
        f.write('Participant ' + participant_num + ' average accuracy is: ' + str(round(accuracy.mean(),4)) + '\n')

    f.close()
else:
    all_data_index_shuffeled = np.arange(data_all.shape[0])
    np.random.shuffle(all_data_index_shuffeled)
    flods = 10
    cms = []
    accs = []
    for i in range(flods):
        validation_data_index = np.arange(i*int(data_all.shape[0]/10),(i+1)*int(data_all.shape[0]/10), dtype = int)
        training_data_index = np.delete(np.arange(data_all.shape[0]),np.arange(i*(data_all.shape[0]/10),(i+1)*(data_all.shape[0]/10), dtype= int))
        
        training_data, training_labels = slice_the_time(data_all[all_data_index_shuffeled], all_labels[all_data_index_shuffeled], training_data_index, time_window_slice,step_len)
        validation_data, validation_labels = slice_the_time(data_all[all_data_index_shuffeled], all_labels[all_data_index_shuffeled], validation_data_index, time_window_slice,step_len)

        training_dataset = eegDataset(training_data, training_labels)
        validation_dataset = eegDataset(validation_data, validation_labels)

        train_dl = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
        val_dl = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True)

        def setRandom(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        transform_arguemnts = {'filterBank':{'filtBank':[[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]],'fs':250, 'filtType':'filter'}}
        transform = transforms.__dict__[list(transform_arguemnts.keys())[0]](**transform_arguemnts[list(transform_arguemnts.keys())[0]])

        if(model_name == 'LGG' or model_name == 'SE_FBCNet' or model_name == 'SE_FBCNet_spatial'):
            
            if(data_set_id == 'BCIIV2a'):
                original_order = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P3','Pz','P4','POz']
                graph_gen_DEAP = [[original_order[0]], original_order[2:5], original_order[8:11],\
                            original_order[14:17], original_order[18:22], [original_order[1], original_order[6], original_order[7], original_order[13]],\
                            [original_order[5], original_order[11], original_order[12], original_order[17]]]
            
            if(data_set_id == 'Koera_54'):
                original_order = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', \
                                'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', \
                                'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', \
                                'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', \
                                'POz', 'FT9', 'T9', 'FT7', 'TP7', 'P9', 'FT10', 'T10', 'P6',\
                                'TP8', 'P10', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']

                graph_gen_DEAP = [[original_order[56],original_order[57], original_order[0], original_order[1], original_order[59], original_order[58]],\
                            original_order[2:7], [original_order[7], original_order[32], original_order[8],original_order[9],original_order[33],original_order[10]],\
                            [original_order[11],original_order[34],original_order[46],original_order[12],original_order[35],original_order[13],original_order[36],original_order[14],original_order[37],original_order[51],original_order[15]],\
                            [original_order[47],original_order[17],original_order[38],original_order[18],original_order[39],original_order[19],original_order[40],original_order[20],original_order[52]],\
                            [original_order[48],original_order[22],original_order[23],original_order[41],original_order[24],original_order[42],original_order[25],original_order[26],original_order[53]],\
                            [original_order[60],original_order[43],original_order[61],original_order[28],original_order[29],original_order[30]], [original_order[27]], [original_order[31]],[original_order[54],original_order[44],original_order[16], original_order[45]],\
                            [original_order[55],original_order[49],original_order[21],original_order[50]]]
            
            if(data_set_id == 'high_gamma'):
                original_order = ["POz", "Pz", "P2", "P1", "CP2", "CP1", "CP4", "CPz"]
                graph_gen_DEAP = [original_order[:4], original_order[4:]]

            if(data_set_id == 'EEG_52'):
                original_order = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',\
                'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3',\
                'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz','Fz','F2', 'F4',\
                'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2','C4','C6', 'T8','TP8', 'CP6','CP4',\
                'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4','O2']

                graph_gen_DEAP = [['Fp1', 'Fpz', 'Fp2'],['AF7', 'AF3', 'AFz', 'AF4', 'AF8'],\
                    ['F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'], ['FT7', 'T7', 'TP7'],\
                    ['FC5', 'FC3', 'FC1', 'FCz','FC2', 'FC4', 'FC6'], ['C5', 'C3', 'C1', 'Cz','C2','C4', 'C6'],\
                    ['FT8', 'T8','TP8'], ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],\
                    ['P9', 'P7','P5', 'P3', 'P1', 'Pz','P2','P4','P6','P8', 'P10'], ['PO7','PO3', 'POz','PO4','PO8'],\
                    ['O1', 'Oz', 'O2', 'Iz']]
                
            if(data_set_id == 'high_gamma_full'):
                original_order = ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', \
                                'M1', 'T7', 'C3', 'Cz', 'C4', 'T8', 'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', \
                                'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2',\
                                'F1', 'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P5', \
                                'P1', 'P2', 'P6', 'PO5', 'PO3', 'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', \
                                'FT10', 'TPP9h', 'TPP10h', 'PO9', 'PO10', 'P9', 'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', \
                                'FFC3h', 'FFC4h', 'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h',\
                                'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h', \
                                'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h', 'FTT9h', 'FTT7h', 'FCC1h',\
                                'FCC2h', 'FTT8h', 'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'CPP1h', 'CPP2h',\
                                'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h']


                graph_gen_DEAP = [['Fp1','Fp2', 'Fpz','AFp3h','AFp4h'], ['F7', 'F3', 'Fz', 'F4', 'F8', 'AFF5h', 'AFF6h'], ['FC5', 'FC1', 'FC2', 'FC6','M1'], ['T7', 'C3',\
                                'Cz', 'C4', 'T8', 'M2'],['CP5', 'CP1', 'CP2', 'CP6'],['P7', 'P3', 'Pz', 'P4', 'P8', 'POz','P9', 'P10'], ['O1', 'Oz', 'O2'],\
                                ['F1', 'F2', 'F6', 'FFT7h', 'FFC1h', 'FFC2h', 'FFT8h'], ['FC3', 'FCz', 'FC4'], ['C5', 'C1', 'C2', 'C6'],['CP3', 'CPz', 'CP4'], ['P5', 'P1', 'P2','P6'], \
                                ['PO5', 'PO3', 'PO4', 'PO6','PO7', 'PO8', 'PO9', 'PO10'], ['FT7', 'FT8', 'FT9', 'FT10'], ['FT7', 'FT8'], ['TPP9h', 'TPP10h'],\
                                ['AFF1', 'AFz', 'AFF2'], [ 'FFC5h','FFC3h', 'FFC4h', 'FFC6h'], ['FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',  'FTT9h', 'FTT7h', 'FCC1h','FCC2h', 'FTT8h', 'FTT10h'], ['CCP5h', 'CCP3h', 'CCP4h','CCP6h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h', 'TPP8h'],\
                                ['CPP5h', 'CPP3h', 'CPP4h', 'CPP6h', 'CPP1h', 'CPP2h'], ['PPO1', 'PPO2','PPO6h', 'PPO10h', 'PPO9h', 'PPO5h', 'PPO6h', 'PPO10h'], ['I1', 'Iz', 'I2', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h', 'OI2h']]

            idx = []
            graph_idx = graph_gen_DEAP
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    print(chan)
                    idx.append(original_order.index(chan))
            print(len(idx))
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(data_set_id), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
            idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(data_set_id), 'r')['data']))
            print(idx_local_graph, idx)
            # import sys
            # sys.exit()
            channels = sum(idx_local_graph)
            input_size = (9, channels, time_window_slice)
            if(model_name == 'LGG'):
                model = networks.__dict__[model_name](num_classes = num_of_classes, input_size = input_size,\
                    sampling_rate = int(128*1), num_T = 64, out_graph = 32, dropout_rate = 0.5, \
                    pool=16, pool_step_rate = 0.25, idx_graph = idx_local_graph, device = device).to(device)
            if(model_name == 'SE_FBCNet'):
                model = SE_FBCNet(nChan = num_channels, nTime = time_window_slice, nClass = num_of_classes, idx = idx, idx_local_graph = idx_local_graph).to(device)
            if(model_name =='SE_FBCNet_spatial'):
                model = SE_FBCNet_spatial(nChan = num_channels, nTime = time_window_slice, nClass = num_of_classes, idx = idx, idx_local_graph = idx_local_graph, chann_locs = my_loc_channels).to(device)
        elif(model_name == 'tempoalNet'):
            model = networks.__dict__[model_name](nChan = num_channels, nTime = time_window_slice, nClass = num_of_classes).to(device)
        elif(model_name == 'TSceptionIJCNN'):
            model = networks.__dict__[model_name](num_classes = 4, input_size = (num_channels, time_window_slice), sampling_rate=freq, num_T=64, num_S=32, hidden=int(time_window_slice/4), dropout_rate=0.2, frequ_bands_amount = 9).to(device)
        elif(model_name == 'FBSCNet'):
            model = FBSCNet(nClass=4, nChan=22,nTime=time_window_slice, nBands=9, m = 32, \
                                sampling_rate = 250, dropout_rate = 0.5, pool=16, pool_step_rate = 0.25).to(device)
        elif(model_name == 'Bio_FSTNet'):
            model = Bio_FSTNet(nChan = num_channels, nTime = time_window_slice, nClass = num_of_classes).to(device)
        else:
            model = networks.__dict__[model_name](nChan = num_channels, nTime = time_window_slice, nClass = num_of_classes).to(device)

        if(model_name == 'TSceptionIJCNN'):
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.NLLLoss()

        if(model_name == 'LGG' or model_name == 'TSceptionIJCNN'):
            learning_rate = 0.0001
        else:
            learning_rate = 0.003

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        path_training_results = data_set_id+'_CV'+'//'+model_name
        if(not os.path.exists(path_training_results)):
            os.makedirs(path_training_results)

        path_training_best_model = path_training_results+'//model_states//participant_'+participant_num
        if(not os.path.exists(path_training_best_model)):
            os.makedirs(path_training_best_model)

        path_training_results_data = path_training_results+'//training_results//participant_'+participant_num
        if(not os.path.exists(path_training_results_data)):
            os.makedirs(path_training_results_data)

        f = open(os.path.join(path_training_results_data, "Pre_processing_order_"+str(preprocessing_order)+".txt"),'a')
        results_dir = os.path.join(path_training_best_model, "Pre_processing_order_"+str(preprocessing_order)+"_best_model.pth")
        last_results_dir = os.path.join(path_training_best_model, "Pre_processing_order_"+str(preprocessing_order)+"_last_model.pth")
        epochs = 250
        highest_acc = 0
        for epoch in range(epochs):
            model.train()
            lossall = 0
            if(epoch != 0 and epoch % 50 == 0 and epoch < 200):
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
            for i,(inputs, targets) in enumerate(train_dl):
                if(model_name == 'FBCNet' or model_name == 'LGG' or model_name == 'tempoalNet' or model_name == 'tempoalNet_' or model_name == 'tempoalNet_me'or model_name == 'TSceptionIJCNN' or model_name == 'FBSCNet' or model_name == 'Bio_FSTNet' or model_name == 'SE_FBCNet' or model_name == 'SE_FBCNet_spatial'):
                    inputs = transform(inputs.numpy())
                    # print(inputs.size())
                
                targets = torch.squeeze(targets.type(torch.int64))
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                output = model(inputs)
                _, preds = torch.max(output, 1)
                acc = round(accuracy_score(targets.tolist(), preds.data.tolist()),3)
                loss = criterion(output, targets)
                loss.backward()
                lossall += loss
                optimizer.step()
            f.write('Epoch No: '+str(epoch+1)+' '+'loss= '+str(lossall)+'\n')
            print('Epoch No: '+str(epoch+1)+' '+'loss= '+str(lossall))
            predicted = []
            actual = []
            loss = 0
            model.eval()
            with torch.no_grad():
                for i,(inputs, targets) in enumerate(val_dl):
                    if(model_name == 'FBCNet' or model_name =='LGG' or model_name == 'tempoalNet' or model_name == 'tempoalNet_' or model_name == 'tempoalNet_me'or model_name == 'TSceptionIJCNN' or model_name == 'FBSCNet' or model_name == 'Bio_FSTNet' or model_name == 'SE_FBCNet' or model_name == 'SE_FBCNet_spatial'):
                        inputs = transform(inputs.numpy())
                    actual.extend(targets.tolist())
                    targets = torch.squeeze(targets.type(torch.int64))
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    preds = model(inputs)

                    loss += criterion(preds, targets)
                    _, preds = torch.max(preds, 1)
                    predicted.extend(preds.data.tolist())

            acc = round(accuracy_score(actual, predicted),3)
            if(acc > highest_acc and epoch >= int(epochs*0.5)):
                highest_acc = acc
                if classes is not None:
                    cm = confusion_matrix(actual, predicted, labels= classes)
                else:
                    cm = confusion_matrix(actual, predicted)
                torch.save(model.state_dict(), results_dir)
                print('Model validation result and confuxion matrix is:\n'+str(highest_acc)+'\n'+str(cm)+'\n')
                f.write('Model validation result and confuxion matrix is:\n'+str(highest_acc)+'\n'+str(cm)+'\n')
        accs.append(highest_acc)
        cms.append(cm)
    accs = np.array(accs)
    cms = np.array(cms)

    print('Model validation mean acc is:\n'+str(np.mean(accs))+'\n')
    print('Model validation confuxion matrix is:\n'+str(np.sum(cms, axis=0))+'\n')
    f.write('Model validation mean acc is:\n'+str(np.mean(accs))+'\n')
    f.write('Model validation confuxion matrix is:\n'+str(np.sum(cms, axis=0))+'\n')
    f.close()