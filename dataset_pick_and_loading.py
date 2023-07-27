import mne
import numpy as np
from scipy.io import loadmat
def load_full_high_gamma_datasetByParticipoantPath(data_path, participant_id):
    T_data_location = data_path+'\\train\\'+participant_id+'.edf'
    rawData = mne.io.read_raw_edf(T_data_location,exclude=['EOGh', 'EOGv', 'EMG_RH', 'EMG_LH', 'EMG_RF'], infer_types = True, preload=True)
    # rawData = mne.io.read_raw_edf(path,exclude=['AF7', 'AF3', 'AF4', 'AF8', 'F5'], infer_types = True, preload=True)
    gdf_events = mne.events_from_annotations(rawData)[0][:,[0,2]].tolist()
    y = np.array([i[1] for i in gdf_events]) -1
    eeg = rawData.get_data()
    fs = 250
    offset = 0.5
    epochWindow = [0.0,4.0]
    epochInterval = np.array(range(int(epochWindow[0]*fs), int(epochWindow[1]*fs)))+int(offset*fs)
    x = np.stack([eeg[:, epochInterval+event[0] ] for event in gdf_events], axis = 2)
    
    T_data_location = data_path+'\\test\\'+participant_id+'.edf'
    raw_fif = mne.io.read_raw_edf(T_data_location,exclude=['EOGh', 'EOGv', 'EMG_RH', 'EMG_LH', 'EMG_RF'], infer_types = True, preload=True)
    gdf_events = mne.events_from_annotations(raw_fif)[0][:,[0,2]].tolist()
    y_new = np.array([i[1] for i in gdf_events]) -1
    eeg = raw_fif.get_data()
    fs = 250
    offset = 0.5
    epochWindow = [0.0,4.0]
    epochInterval = np.array(range(int(epochWindow[0]*fs), int(epochWindow[1]*fs)))+int(offset*fs)
    x_new = np.stack([eeg[:, epochInterval+event[0] ] for event in gdf_events], axis = 2)

    x = np.concatenate((x,x_new), axis = 2)
    y = np.concatenate((y,y_new), axis = 0)

    info = mne.create_info(raw_fif.ch_names, sfreq=250, ch_types='eeg')
    x =np.transpose(x, (2,0,1)) 
    epoched_data = mne.EpochsArray(x, info)
    epoched_data.set_montage('standard_1005')
    print(x.shape, y.shape)
    return epoched_data, y

def load_high_gamma_datasetByParticipoantPath(data_path, participant_id):
    ch_names = ["POz", "Pz", "P2", "P1", "CP2", "CP1", "CP4", "CPz"]
    T_data_location = data_path+'\\Subject '+participant_id+'\\0\\0-raw.fif'
    raw_fif = mne.io.read_raw_fif(T_data_location, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None)
    gdf_events = mne.events_from_annotations(raw_fif)[0][:,[0,2]].tolist()
    y = np.array([i[1] for i in gdf_events]) -1
    eeg = raw_fif.get_data()
    fs = 250
    offset = 0.5
    epochWindow = [0.0,4.0]
    epochInterval = np.array(range(int(epochWindow[0]*fs), int(epochWindow[1]*fs)))+int(offset*fs)
    x = np.stack([eeg[:, epochInterval+event[0] ] for event in gdf_events], axis = 2)

    T_data_location = data_path+'\\Subject '+participant_id+'\\1\\1-raw.fif'
    raw_fif = mne.io.read_raw_fif(T_data_location, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None)
    gdf_events = mne.events_from_annotations(raw_fif)[0][:,[0,2]].tolist()
    y_new = np.array([i[1] for i in gdf_events]) -1
    eeg = raw_fif.get_data()
    fs = 250
    offset = 0.5
    epochWindow = [0.0,4.0]
    epochInterval = np.array(range(int(epochWindow[0]*fs), int(epochWindow[1]*fs)))+int(offset*fs)
    x_new = np.stack([eeg[:, epochInterval+event[0] ] for event in gdf_events], axis = 2)

    x = np.concatenate((x,x_new), axis = 2)
    y = np.concatenate((y,y_new), axis = 0)

    info = mne.create_info(ch_names, sfreq=250, ch_types='eeg')
    print(len(ch_names), x.shape)
    x =np.transpose(x, (2,0,1)) 
    epoched_data = mne.EpochsArray(x, info)
    epoched_data.set_montage('standard_1020')
    print(x.shape, y.shape)
    return epoched_data, y

def load_HaLT_DataByParticipoantPath(data_location, participant_num):
    data_location = data_location +'//HaLT_dataset_'+str(participant_num)+'.npz'
    data_file = np.load(data_location)
    x = data_file['arr_0'][:,:21,:]
    y = data_file['arr_1']
    print(x.shape, y.shape)

    channels_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', \
            'P4', 'O1', 'O2', 'A1', 'A2', 'F7', 'F8', 'T3', 'T4', \
            'T5', 'T6', 'Fz', 'Cz', 'Pz']

    print(len(channels_list))

    info = mne.create_info(channels_list, sfreq=200, ch_types='eeg')
    epoched_data = mne.EpochsArray(x, info)
    epoched_data.set_montage('standard_1020')
    return epoched_data, y

def load_Korea54_DataByParticipoantPath(data_location, participant_num):
    data_location = data_location +'//S'+str(participant_num)+'.mat'
    mat = loadmat(data_location)
    x, y = mat['down_sample'], mat['label'][:,0]

    channels_list = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', \
        'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', \
        'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', \
        'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', \
        'POz', 'FT9', 'T9', 'FT7', 'TP7', 'P9', 'FT10', 'T10', 'P6',\
        'TP8', 'P10', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']
    
    info = mne.create_info(channels_list, sfreq=250, ch_types='eeg')
    epoched_data = mne.EpochsArray(x, info)
    epoched_data.set_montage('standard_1020')
    print(y.shape)
    return epoched_data, y

def load_fatigue_detection_by_participoant_path_old(data_location, participant_num):

    rawData_fs = mne.io.read_raw_cnt(data_location+'/'+str(participant_num)+'/Fatigue state.cnt', eog=['HEOL', 'HEOR', 'VEOU', 'VEOL'])
    rawData = mne.io.read_raw_cnt(data_location+'/'+str(participant_num)+'/Normal state.cnt', eog=['HEOL', 'HEOR', 'VEOU', 'VEOL'])
    data_used = rawData.get_data(picks='eeg')
    data_used_fs = rawData_fs.get_data(picks='eeg')
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', \
                'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'Pz', 'P4', 'T6', 'A2', 'O1', 'Oz', 'O2', 'FT9', 'FT10', 'PO1', 'PO2']
    info = mne.create_info(ch_names, sfreq=1000, ch_types='eeg')
    raw_data_used = mne.io.RawArray(data_used, info)
    raw_data_used_fs = mne.io.RawArray(data_used_fs, info)
    raw_data_used.set_montage('standard_1020')
    raw_data_used_fs.set_montage('standard_1020')

    data_visualization = raw_data_used.get_data()[:,:600000]
    data_visualization_fs = raw_data_used_fs.get_data()[:,:600000]
    print(data_visualization.shape)

    all_data_train = []
    all_data_test = []
    all_data_val = []

    all_labels_train = []
    all_labels_test = []
    all_labels_val = []

    i = 0
    # 0: normal data, 1: fatigue state
    while i <= data_visualization.shape[1]:
        if(i%(data_visualization.shape[1]/4) >= 100000 and i%(data_visualization.shape[1]/4) <= 125000):
            all_data_test.append(data_visualization[:, i:i+1000])
            all_labels_test.append(0)
            all_data_test.append(data_visualization_fs[:, i:i+1000])
            all_labels_test.append(1)
            i = i + 500
            print(i)
            if(data_visualization.shape[1] - i < 1000):
                break
            continue
        
        if(i%(data_visualization.shape[1]/4) >= 125000 and i%(data_visualization.shape[1]/4) <= 150000):
            all_data_val.append(data_visualization[:, i:i+1000])
            all_labels_val.append(0)
            all_data_val.append(data_visualization_fs[:, i:i+1000])
            all_labels_val.append(1)
            i = i + 500
            print(i)
            if(data_visualization.shape[1] - i < 1000):
                break
            continue

        if(i%(data_visualization.shape[1]/4) == 99000):
            i = i + 1000
            if(data_visualization.shape[1] - i < 1000):
                break
            continue
        all_data_train.append(data_visualization[:, i:i+1000])
        all_labels_train.append(0)
        all_data_train.append(data_visualization_fs[:, i:i+1000])
        all_labels_train.append(1)
        i = i + 500
        if(data_visualization.shape[1] - i < 1000):
                break
    all_data_train = np.array(all_data_train)
    all_data_test = np.array(all_data_test)
    all_data_val = np.array(all_data_val)

    all_labels_train = np.array(all_labels_train)
    all_labels_test = np.array(all_labels_test)
    all_labels_val = np.array(all_labels_val)

    return all_data_train, all_data_test, all_data_val, all_labels_train, all_labels_test, all_labels_val

def load_fatigue_detection_by_participoant_path(data_location, participant_num):
    print(data_location)
    rawData_fs = mne.io.read_raw_cnt(data_location+'/'+str(participant_num)+'/Fatigue state.cnt', eog=['HEOL', 'HEOR', 'VEOU', 'VEOL'])
    rawData = mne.io.read_raw_cnt(data_location+'/'+str(participant_num)+'/Normal state.cnt', eog=['HEOL', 'HEOR', 'VEOU', 'VEOL'])
    data_used = rawData.get_data(picks='eeg')
    data_used_fs = rawData_fs.get_data(picks='eeg')
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', \
                'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'A1', 'T5', 'P3', 'Pz', 'P4', 'T6', 'A2', 'O1', 'Oz', 'O2', 'FT9', 'FT10', 'PO1', 'PO2']
    info = mne.create_info(ch_names, sfreq=1000, ch_types='eeg')
    raw_data_used = mne.io.RawArray(data_used, info)
    raw_data_used_fs = mne.io.RawArray(data_used_fs, info)
    raw_data_used.set_montage('standard_1020')
    raw_data_used_fs.set_montage('standard_1020')

    data_visualization = raw_data_used.get_data()[:,:600000]
    data_visualization_fs = raw_data_used_fs.get_data()[:,:600000]
    # print(data_visualization.shape)

    all_data = []
    all_labels = []
    data_visualization_fs[np.isnan(data_visualization_fs)] = 0
    data_visualization[np.isnan(data_visualization)] = 0
    i = 0
    # 0: normal data, 1: fatigue state
    data_interival = 3000
    while i <= data_visualization.shape[1]:
        all_data.append(data_visualization[:, i:i+data_interival])
        all_labels.append(0)
        all_data.append(data_visualization_fs[:, i:i+data_interival])
        all_labels.append(1)
        i = i + data_interival
        if(data_visualization.shape[1] - i < data_interival):
                break

    all_data = np.array(all_data)
    y = np.array(all_labels).reshape(len(all_labels),)
    epoched_data = mne.EpochsArray(all_data, info)
    epoched_data.set_montage('standard_1020')
    return epoched_data, y

def load_EEG52_DataByParticipantPath(data_location, participant_num):
    data_path = data_location +'//s'+str(participant_num)+'.mat'
    mat = loadmat(data_path)
    labels = mat['eeg'][0][0][-6]
    labels_time_point = []
    for i in range(labels.shape[1]):
        if(labels[0,i] == 1):
            labels_time_point.append(i)
    dataset_left =  mat['eeg'][0][0][7]*0.01
    dataset_right =  mat['eeg'][0][0][8]*0.01
    all_eeg_data = []
    all_labels = []
    for i in range(len(labels_time_point)):
        all_eeg_data.append(dataset_left[:,labels_time_point[i]:labels_time_point[i]+1500])
        all_labels.append(0)
        all_eeg_data.append(dataset_right[:,labels_time_point[i]:labels_time_point[i]+1500])
        all_labels.append(1)
    all_eeg_data = np.array(all_eeg_data)
    # all_eeg_data = all_eeg_data * -1e-6
    
    # all_eeg_data = np.around(all_eeg_data,4)
    # print(all_eeg_data)
    print('Is any Nan: ', np.isnan(all_eeg_data).any())
    y = np.array(all_labels)
    channel_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',\
    'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3',\
    'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz','Fz','F2', 'F4',\
    'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2','C4','C6', 'T8','TP8', 'CP6','CP4',\
    'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4','O2']
    info = mne.create_info(channel_names, sfreq=512, ch_types='eeg')
    epoched_data = mne.EpochsArray(all_eeg_data[:,:64,:], info)
    # y = y.reshape
    epoched_data.set_montage('standard_1020')
    return epoched_data, y

def load_BCIIV2a_data_epoch(data_path, participant_id, chans = list(range(22))):

    if(participant_id == '4'):
        eventCode = [4]
    else:
        eventCode = [6]
        
    fs = 250
    offset = 0.5
    epochWindow = [0.5,4.5]

    T_data_location = data_path+'\\A0'+participant_id+'T.gdf'

    raw_gdf = mne.io.read_raw_gdf(T_data_location, stim_channel="auto")
    raw_gdf.load_data()
    gdf_events = mne.events_from_annotations(raw_gdf)[0][:,[0,2]].tolist()
    eeg = raw_gdf.get_data()

    if chans is not None:
        eeg = eeg[chans,:]

    events = [event for event in gdf_events if event[1] in eventCode]
    y = np.array([i[1] for i in events])
    epochInterval = np.array(range(int(epochWindow[0]*fs), int(epochWindow[1]*fs)))+int(offset*fs)
    x = np.stack([eeg[:, epochInterval+event[0] ] for event in events], axis = 2)
    
    x = x*1e6
    labelPath = data_path+'\\A0'+participant_id+'T.mat'

    y = loadmat(labelPath)["classlabel"].squeeze()
    y = y -1
    eventCode = [6] # start of the trial at t=0
    fs = 250
    offset = 1.5

    T_data_location = data_path+'\\A0'+participant_id+'E.gdf'

    # load the gdf file using MNE
    raw_gdf = mne.io.read_raw_gdf(T_data_location, stim_channel="auto")
    raw_gdf.load_data()
    gdf_events = mne.events_from_annotations(raw_gdf)[0][:,[0,2]].tolist()
    eeg = raw_gdf.get_data()

    if chans is not None:
        eeg = eeg[chans,:]
    
    #Epoch the data
    events = [event for event in gdf_events if event[1] in eventCode]
    y_new = np.array([i[1] for i in events])
    epochInterval = np.array(range(int(epochWindow[0]*fs), int(epochWindow[1]*fs)))+int(offset*fs)
    x_new = np.stack([eeg[:, epochInterval+event[0] ] for event in events], axis = 2)
    
    # Multiply the data with 1e6
    x_new = x_new*1e6
    labelPath = data_path+'\\A0'+participant_id+'E.mat'

    # Load the labels
    y_new = loadmat(labelPath)["classlabel"].squeeze()
    y_new = y_new - 1 
    print(x.shape, x_new.shape)
    x = np.concatenate((x,x_new), axis = 2)
    y = np.concatenate((y,y_new), axis = 0)

    channels_list = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P3','Pz','P4','POz']
    info = mne.create_info(channels_list, sfreq=250, ch_types='eeg')
    print(len(channels_list), x.shape)
    x =np.transpose(x, (2,0,1)) 
    epoched_data = mne.EpochsArray(x, info)
    epoched_data.set_montage('standard_1020')
    print(y.shape,x.shape)
    return epoched_data, y

def generate_balanced_index(labels,required_length,required_length_val, required_length_test, classes_amount = 4):
    zero_size = 0
    zero_size_val = 0
    zero_size_test = 0
    
    one_size = 0
    one_size_val = 0
    one_size_test = 0
    
    two_size = 0
    two_size_val = 0
    two_size_test = 0

    three_size = 0
    three_size_val = 0
    three_size_test = 0

    output_index = []
    output_index_val = []
    output_index_test = []

    # print('Label shape:',labels.shape)
    for i in range(labels.shape[0]):
        # print(labels[i])
        if(labels[i] == 0):
            if(zero_size <= int(required_length/classes_amount)):
                output_index.append(i)
                zero_size += 1
                continue
            if(zero_size_val<= int(required_length_val/classes_amount)):
                output_index_val.append(i)
                zero_size_val += 1
                continue
            if(zero_size_test<= int(required_length_test/classes_amount)):
                output_index_test.append(i)
                zero_size_test += 1
                continue
        if(labels[i] == 1):
            if(one_size <= int(required_length/classes_amount)):
                output_index.append(i)
                one_size += 1
                continue
            if(one_size_val<= int(required_length_val/classes_amount)):
                output_index_val.append(i)
                one_size_val += 1
                continue
            if(one_size_test<= int(required_length_test/classes_amount)):
                output_index_test.append(i)
                one_size_test += 1
                continue
        if(labels[i] == 2):
            if(two_size <= int(required_length/classes_amount)):
                output_index.append(i)
                two_size += 1
                continue
            if(two_size_val<= int(required_length_val/classes_amount)):
                output_index_val.append(i)
                two_size_val += 1
                continue
            if(two_size_test<= int(required_length_test/classes_amount)):
                output_index_test.append(i)
                two_size_test += 1
                continue
        if(labels[i] == 3):
            if(three_size <= int(required_length/classes_amount)):
                output_index.append(i)                                        
                three_size += 1
                continue
            if(three_size_val<= int(required_length_val/classes_amount)):
                output_index_val.append(i)
                three_size_val += 1
                continue
            if(three_size_test<= int(required_length_test/classes_amount)):
                output_index_test.append(i)
                three_size_test += 1
                continue
    return np.array(output_index), np.array(output_index_val), np.array(output_index_test)

def slice_the_time(data, labels, index, time_window, step_len):
    used_data = data
    data_real = []
    labels_real = []
    for i in index:
        for j in range(int((data.shape[2]-time_window)/step_len)+1) :
            data_real.append(used_data[i,:,j*step_len:j*step_len+time_window])
            labels_real.append(labels[i])
    return np.array(data_real), np.array(labels_real)
