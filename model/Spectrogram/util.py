import tensorflow as tf
import random
import numpy as np
import os
import scipy.io as sio
import pyworld as pw

def loadMat(filename):
    data = sio.loadmat(filename)
    return np.transpose(data['magnitude'])



def nextbatch(path, featureSize, batchSize):
    speakerFolder = os.listdir(path)

    label_batch = np.zeros([1, len(speakerFolder)])
    x_batch = np.zeros([1, featureSize])

    while(x_batch.shape[0] <= batchSize + 1):

        speaker = random.choice(speakerFolder)
        speakerLabel = np.zeros([1, len(speakerFolder)])
        speakerLabel[0, speakerFolder.index(speaker)] = 1

        Folderpath = path + speaker + '/'
        speakerFile = os.listdir(Folderpath)
        file = random.choice(speakerFile)
        spectral = loadMat(Folderpath + file)
        length = spectral.shape[0]
        if(length <= batchSize):
            segment = spectral[0: length, :]
        else:
            ind = random.choice(range(max(length-128, 0)))
            segment = spectral[ind: ind+128, :]
        speakerLabel = np.tile(speakerLabel, [length, 1])
        x_batch = np.concatenate((x_batch, segment), axis = 0)
        label_batch = np.concatenate((label_batch, speakerLabel), axis = 0)

    return x_batch[1:batchSize+1, :], label_batch[1:batchSize+1, :]

def pickOne(path, src, trg):

    speakerFolder = os.listdir(path)
    trg_label = np.zeros([1, len(speakerFolder)])
    trg_label[0, speakerFolder.index(trg)] = 1

    Folderpath = path + src + '/'
    speakerFile = os.listdir(Folderpath)
    filename = random.choice(speakerFile)
    spectral = loadMat( Folderpath + filename)
    length = spectral.shape[0]
    trg_label = np.tile(trg_label, [length, 1])
    return spectral, trg_label, filename


def synthesis(path, src, trg, sp, filename):
    spath = path + src + '/'
    tfrecords_filename = spath + filename
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        ap_string = example.features.feature['ap'].bytes_list.value[0]
        ap = np.fromstring(ap_string, dtype=np.float64)
        length = int(example.features.feature['len'].int64_list.value[0])
        ap = ap.reshape([length, 513])
        Nf_string = example.features.feature['Nfactor'].bytes_list.value[0]
        Nfactor = np.fromstring(Nf_string, dtype=np.float64)

    sp = np.multiply(sp, np.transpose(np.tile(Nfactor, (513, 1))))
    fs = 16000
    f0_c = convert_f0(path, src, trg, filename)
    y = pw.synthesize(f0_c, sp, ap, fs)
    y = y.reshape([-1])
    opname = src + '_To_' + trg + '_' + filename + '.mat'
    sio.savemat(opname, mdict={'y': y})
    return sp, ap, f0_c


def convert_f0(path, src, trg, filename):
    srcfile = path + src + '/' + filename
    trgfile = path + trg + '/' + filename

    record_iterator = tf.python_io.tf_record_iterator(path=srcfile)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        f0_string = example.features.feature['f0'].bytes_list.value[0]
        f0_s = np.fromstring(f0_string, dtype=np.float64)

    record_iterator = tf.python_io.tf_record_iterator(path=trgfile)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        f0_string = example.features.feature['f0'].bytes_list.value[0]
        f0_t = np.fromstring(f0_string, dtype=np.float64)

    f0_s = f0_s
    f0_t = f0_t
    mu_s = np.log(np.mean(f0_s[f0_s > 1]))
    mu_t = np.log(np.mean(f0_t[f0_t > 1]))
    std_s = np.log(np.std(f0_s[f0_s > 1]))
    std_t = np.log(np.std(f0_t[f0_t > 1]))
    lf0 = np.where(f0_s > 1., np.log(f0_s), f0_s)
    lf0 = np.where(lf0 > 1., (lf0 - mu_s) / std_s * std_t + mu_t, lf0)
    f0_c = np.where(lf0 > 1., np.exp(lf0), lf0)
    f0_c = f0_c.reshape(-1)
    return f0_c