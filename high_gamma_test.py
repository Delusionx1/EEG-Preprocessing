import mne
import numpy as np
from pre_processing_methods import automated_ICA, surface_laplacian, bandpass_filtering, baseline_correction

path= r'F:\EEG_datasets\high-gamma-dataset\raw\master\data\train\3.edf'
rawData = mne.io.read_raw_edf(path,exclude=['EOGh', 'EOGv', 'EMG_RH', 'EMG_LH', 'EMG_RF', 'AF7', 'AF3', 'AF4', 'AF8', 'F5'], infer_types = True, preload=True)
# rawData = mne.io.read_raw_edf(path,exclude=['AF7', 'AF3', 'AF4', 'AF8', 'F5'], infer_types = True, preload=True)
print(mne.channels.get_builtin_montages())
rawData.set_montage('standard_1005', match_case=False)
gdf_events = mne.events_from_annotations(rawData)[0][:,[0,2]].tolist()
eeg = rawData.get_data()
fs = 250
offset = 0.5
epochWindow = [0.0,4.0]
epochInterval = np.array(range(int(epochWindow[0]*fs), int(epochWindow[1]*fs)))+int(offset*fs)
x = np.stack([eeg[:, epochInterval+event[0] ] for event in gdf_events], axis = 2)
print(rawData.ch_names)
orginal_order = ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', \
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
info = mne.create_info(orginal_order, sfreq=250, ch_types='eeg')
print(len(orginal_order), x.shape)
x =np.transpose(x, (2,0,1)) 
epoched_data = mne.EpochsArray(x, info)
epoched_data = surface_laplacian(epoched_data)
print(rawData.get_data().shape, x.shape)
graph_gen_DEAP = [[],]