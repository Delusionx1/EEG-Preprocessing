from mne.preprocessing import ICA
from mne_icalabel.iclabel import iclabel_label_components
import numpy as np
from scipy import special
import math
import mne
from numpy import inf

def automated_ICA(epoched_data):
    channel_amount = epoched_data.get_data().shape[1]
    print(channel_amount)
    n_components = int(channel_amount * 0.75)
    filt_raw = epoched_data.copy().filter(l_freq=1., h_freq=None)
    ica = ICA(n_components=n_components, max_iter='auto', random_state=97)
    ica.fit(filt_raw)
    ica_array = iclabel_label_components(filt_raw, ica, inplace=True)
    ic_types = np.argmax(ica_array, axis=1)
    exclude_index = []
    for i in range(n_components):
        if(ic_types[i] == 2):
            exclude_index.append(i)
            print('EOG channel dected')
        if(ic_types[i] == 3):
            exclude_index.append(i)
            print('ECG channel dected')
    epoched_data = ica.apply(epoched_data, exclude_index)
    return epoched_data

def bandpass_filtering(epoched_data, Fs = 250, norch_freq = 50):
    montage = epoched_data.get_montage()
    info = epoched_data.info
    data_norched = epoched_data.get_data()
    data_norched = mne.filter.notch_filter(data_norched, Fs = Fs, freqs= norch_freq)
    epoched_data = mne.EpochsArray(data=data_norched, info=info)
    epoched_data.set_montage(montage, match_case=False)
    return epoched_data.filter(l_freq=1., h_freq=50, method='iir')

def baseline_correction(epoched_data):
    montage = epoched_data.get_montage()
    info = epoched_data.info
    data_to_be_baseline_corr = epoched_data.get_data()
    epoched_data = mne.EpochsArray(data=data_to_be_baseline_corr, info=info, baseline=(0,0.5))
    epoched_data.set_montage(montage, match_case=False)
    return epoched_data

def surface_laplacian(epoched_data):
    m = 4
    leg_order = 50
    smoothing = 1e-5
    montage = epoched_data.get_montage()
    
    # get electrodes positions
    locs = epoched_data._get_channel_positions()

    x = locs[:,0]
    y = locs[:,1]
    z = locs[:,2]

    # arrange data
    data = epoched_data.get_data() # data
    data = np.rollaxis(data, 0, 3)
    orig_data_size = np.squeeze(data.shape)

    numelectrodes = len(x)
    print('numelectrods', numelectrodes)
    
    # normalize cartesian coordenates to sphere unit
    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r

    junk1, junk2, spherical_radii = cart2sph(x,y,z)
    maxrad = np.max(spherical_radii)
    x = x/maxrad
    y = y/maxrad
    z = z/maxrad
    
    # compute cousine distance between all pairs of electrodes
    cosdist = np.zeros((numelectrodes, numelectrodes))
    for i in range(numelectrodes):
        for j in range(i+1,numelectrodes):
            cosdist[i,j] = 1 - (((x[i] - x[j])**2 + (y[i] - y[j])**2 + (z[i] - z[j])**2)/2)

    cosdist = cosdist + cosdist.T + np.identity(numelectrodes)

    # get legendre polynomials
    legpoly = np.zeros((leg_order, numelectrodes, numelectrodes))
    for ni in range(leg_order):
        for i in range(numelectrodes):
            for j in range(i+1, numelectrodes):
                #temp = special.lpn(8,cosdist[0,1])[0][8]
                legpoly[ni,i,j] = special.lpn(ni+1,cosdist[i,j])[0][ni+1]

    legpoly = legpoly + np.transpose(legpoly,(0,2,1))

    for i in range(leg_order):
        legpoly[i,:,:] = legpoly[i,:,:] + np.identity(numelectrodes)

    # compute G and H matrixes
    twoN1 = np.multiply(2, range(1, leg_order+1))+1
    gdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m, dtype=float)
    hdenom = np.power(np.multiply(range(1, leg_order+1), range(2, leg_order+2)), m-1, dtype=float)

    G = np.zeros((numelectrodes, numelectrodes))
    H = np.zeros((numelectrodes, numelectrodes))

    for i in range(numelectrodes):
        for j in range(i, numelectrodes):

            g = 0
            h = 0

            for ni in range(leg_order):
                g = g + (twoN1[ni] * legpoly[ni,i,j]) / gdenom[ni]
                h = h - (twoN1[ni] * legpoly[ni,i,j]) / hdenom[ni]

            G[i,j] = g / (4*math.pi)
            H[i,j] = -h / (4*math.pi)

    G = G + G.T
    H = H + H.T

    G = G - np.identity(numelectrodes) * G[1,1] / 2
    H = H - np.identity(numelectrodes) * H[1,1] / 2

    if np.any(orig_data_size==1):
        data = data[:]
    else:
        data = np.reshape(data, (orig_data_size[0], np.prod(orig_data_size[1:3])))

    # compute C matrix
    Gs = G + np.identity(numelectrodes) * smoothing
    GsinvS = np.sum(np.linalg.inv(Gs), 0)
    dataGs = np.dot(data.T, np.linalg.inv(Gs))
    C = dataGs - np.dot(np.atleast_2d(np.sum(dataGs, 1)/np.sum(GsinvS)).T, np.atleast_2d(GsinvS))

    # apply transform
    original = np.reshape(data, orig_data_size)
    surf_lap = np.reshape(np.transpose(np.dot(C,np.transpose(H))), orig_data_size)

    # re-arrange data into mne's epoched_data object
    events = epoched_data.events
    event_id = epoched_data.event_id 
    ch_names = epoched_data.ch_names
    sfreq = epoched_data.info['sfreq']
    tmin = epoched_data.tmin
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    original = np.rollaxis(original, 2, 0)
    surf_lap = np.rollaxis(surf_lap, 2, 0)

    print('Is any Nan: ', np.isnan(surf_lap).any())
    # np.nan_to_num(surf_lap, nan=0.0)
    # surf_lap[surf_lap == -inf] = 0
    before = mne.EpochsArray(data=original, info=info, events=events, event_id=event_id, tmin=tmin, 
                             on_missing='ignore')
    before.set_montage(montage, match_case=False)
    after = mne.EpochsArray(data=surf_lap, info=info, events=events, event_id=event_id, tmin=tmin, 
                            on_missing='ignore')
    after.set_montage(montage, match_case=False)
    
    return after