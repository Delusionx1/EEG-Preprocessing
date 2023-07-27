import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
from bin.FBCSP import FBCSP
from bin.Classifier import Classifier
from sklearn.svm import SVR

def FBCSP_whole(training_data, training_labels, testing_data, testing_labels, freq):
    fbank = FilterBank(freq)
    fbank.get_filter_coeff()
    time_using = training_data.shape[-1] / freq
    window_details = {'tmin':0.0,'tmax':time_using}

    filtered_data_train = fbank.filter_data(training_data, window_details)
    filtered_data_test = fbank.filter_data(testing_data, window_details)

    testing_accuracy = []
    m_filters = 2

    y_classes_unique = np.unique(training_labels)
    n_classes = len(np.unique(training_labels))

    fbcsp = FBCSP(m_filters)
    fbcsp.fit(filtered_data_train,training_labels)
    y_train_predicted = np.zeros((training_labels.shape[0], n_classes), dtype=np.float)
    y_test_predicted = np.zeros((testing_labels.shape[0], n_classes), dtype=np.float)
    # print(filtered_data_train.shape)
    for j in range(n_classes):
        cls_of_interest = y_classes_unique[j]
        select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

        y_train_cls = np.asarray(select_class_labels(cls_of_interest, training_labels))
        y_test_cls = np.asarray(select_class_labels(cls_of_interest, testing_labels))

        x_features_train = fbcsp.transform(filtered_data_train,class_idx=cls_of_interest)
        x_features_test = fbcsp.transform(filtered_data_test,class_idx=cls_of_interest)

        classifier_type = SVR(gamma='auto')
        classifier = Classifier(classifier_type)
        y_train_predicted[:,j] = classifier.fit(x_features_train,np.asarray(y_train_cls,dtype=np.float))
        y_test_predicted[:,j] = classifier.predict(x_features_test)

    y_test_predicted_multi = get_multi_class_regressed(y_test_predicted)
    te_acc =np.sum(y_test_predicted_multi == testing_labels, dtype=np.float) / len(testing_labels)

    testing_accuracy.append(te_acc)
    mean_testing_accuracy = np.mean(np.asarray(testing_accuracy))

    return mean_testing_accuracy

def cross_validate_Ntimes_Kfold(self, y_labels, ifold=0):
        from sklearn.model_selection import StratifiedKFold
        train_indices = {}
        test_indices = {}
        random_seed = ifold
        skf_model = StratifiedKFold(n_splits=self.kfold, shuffle=True, random_state=random_seed)
        i = 0
        for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
            train_indices.update({i: train_idx})
            test_indices.update({i: test_idx})
            i += 1
        return train_indices, test_indices

def cross_validate_sequential_split(y_labels,kfold):
    from sklearn.model_selection import StratifiedKFold
    train_indices = {}
    test_indices = {}
    skf_model = StratifiedKFold(n_splits=kfold, shuffle=False)
    i = 0
    for train_idx, test_idx in skf_model.split(np.zeros(len(y_labels)), y_labels):
        train_indices.update({i: train_idx})
        test_indices.update({i: test_idx})
        i += 1
    return train_indices, test_indices

def cross_validate_half_split(y_labels):
    import math
    unique_classes = np.unique(y_labels)
    all_labels = np.arange(len(y_labels))
    train_idx =np.array([])
    test_idx = np.array([])
    for cls in unique_classes:
        cls_indx = all_labels[np.where(y_labels==cls)]
        if len(train_idx)==0:
            train_idx = cls_indx[:math.ceil(len(cls_indx)/2)]
            test_idx = cls_indx[math.ceil(len(cls_indx)/2):]
        else:
            train_idx=np.append(train_idx,cls_indx[:math.ceil(len(cls_indx)/2)])
            test_idx=np.append(test_idx,cls_indx[math.ceil(len(cls_indx)/2):])

    train_indices = {0:train_idx}
    test_indices = {0:test_idx}

    return train_indices, test_indices

def split_xdata(eeg_data, train_idx, test_idx):
    x_train_fb=np.copy(eeg_data[:,train_idx,:,:])
    x_test_fb=np.copy(eeg_data[:,test_idx,:,:])
    return x_train_fb, x_test_fb

def split_ydata(y_true, train_idx, test_idx):
    y_train = np.copy(y_true[train_idx])
    y_test = np.copy(y_true[test_idx])

    return y_train, y_test

def get_multi_class_label(y_predicted, cls_interest=0):
    y_predict_multi = np.zeros((y_predicted.shape[0]))
    for i in range(y_predicted.shape[0]):
        y_lab = y_predicted[i, :]
        lab_pos = np.where(y_lab == cls_interest)[0]
        if len(lab_pos) == 1:
            y_predict_multi[i] = lab_pos
        elif len(lab_pos > 1):
            y_predict_multi[i] = lab_pos[0]
    return y_predict_multi

def get_multi_class_regressed(y_predicted):
    y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
    return y_predict_multi


class FilterBank:
    def __init__(self,fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data,window_details={}):
        n_trials, n_channels, n_samples = eeg_data.shape
        if window_details:
            n_samples = int(self.fs*(window_details.get('tmax')-window_details.get('tmin')))
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            if window_details:
                eeg_data_filtered = eeg_data_filtered[:,:,:]
            filtered_data[i,:,:,:]=eeg_data_filtered

        return filtered_data

