import tensorflow as tf
import random
import numpy as np
import os
import scipy.io as sio
import time

def loadTF(tfrecords_filename):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        signal_string = example.features.feature['value'].bytes_list.value[0]
        signal = np.fromstring(signal_string, dtype=np.float64)
        return signal

def loadData(path, L, tstep):
    Data = {}
    speaker = os.listdir(path)
    lab_num = []
    for s in speaker[0:5]:
        spath = path+s+'/'
        filename = os.listdir(spath)
        tfrecords_filename = path + s + '/' + filename[0]
        signal = loadTF(tfrecords_filename)
        # mat = sio.loadmat(spath + filename[0])
        # mat = mat['value']
        length = L*tstep
        tmp = np.reshape(signal[0: length*(signal.shape[0] // length)], [-1, length])
        Sind = np.zeros([1, 10], dtype=np.int8)
        Sind[0, speaker.index(s)] = 1

        if 'Label' in locals():
            Label = np.concatenate((Label, np.tile(Sind, [tmp.shape[0], 1])), axis=0)
            Data = np.concatenate((Data, tmp), axis=0)
        else:
            Label = np.tile(Sind, [tmp.shape[0], 1])
            Data = tmp

        for name in filename[1:len(filename)]:
            tfrecords_filename = path + s + '/' + name
            signal = loadTF(tfrecords_filename)
            tmp = np.reshape(signal[0:length * (signal.shape[0] // length)], [-1, length])
            Label = np.concatenate((Label,np.tile(Sind, [tmp.shape[0], 1])), axis=0)
            Data = np.concatenate((Data, tmp), axis=0)
    return Data, Label

def pickTransferInput(path, src, trg, length):
    speaker = os.listdir(path)
    Sind = np.zeros([1, 10], dtype=np.int8)
    Sind[0, speaker.index(trg)] = 1

    srcPath = path+src+'/'
    srcfile = os.listdir(srcPath)
    ind = random.randint(0, len(srcfile) - 1)
    tfrecords_filename = srcPath+srcfile[ind]
    signal = loadTF(tfrecords_filename)
    tmp = np.reshape(signal[0:length * (signal.shape[0] // length)], [-1, length])
    Label = np.tile(Sind, [tmp.shape[0], 1])
    return tmp, Label, srcfile[ind]

def nextbatch(data, Label, batch):
    ind = random.sample(range(data.shape[0]), batch)
    srcData = data[ind, :]
    l = Label[ind, :]
    return srcData, l