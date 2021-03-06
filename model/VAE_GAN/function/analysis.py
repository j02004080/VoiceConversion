import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import os
import random
import tensorflow as tf
import pyworld as pw
import math
import sys

def loadData(path):
    Data = {}
    for speaker in os.listdir(path):
        spath = path + speaker + '/'
        filename = os.listdir(spath)
        Data[speaker] = []
        for name in filename:
            tfrecords_filename = path + speaker + '/' + name
            record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                spec_string = example.features.feature['spec'].bytes_list.value[0]
                spec = np.fromstring(spec_string, dtype=np.float64)
                length = int(example.features.feature['len'].int64_list.value[0])
                spec = spec.reshape([length, 513])
                Data[speaker].append(spec)
    return Data

def pickTransferInput(path, src, trg):
    srcPath = path+src+'/'
    srcfile = os.listdir(srcPath)
    speaker = os.listdir(path)
    ind = random.randint(0, len(srcfile)-1)
    tfrecords_filename = path + src + '/' + srcfile[ind]
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        spec_string = example.features.feature['spec'].bytes_list.value[0]
        spec = np.fromstring(spec_string, dtype=np.float64)
        length = int(example.features.feature['len'].int64_list.value[0])
        srcData = spec.reshape([length, 513])
    label = np.zeros([1, len(speaker)])
    label[0, speaker.index(trg)] = 1
    y = np.tile(label, [length, 1])
    return srcData, y, srcfile[ind]

def nextbatch(data, batchSize):
    speaker = list(data.keys())
    s = random.choice(speaker)
    sind = speaker.index(s)
    dat = random.choice(data[s])
    i = random.choice(range(dat.shape[0] - batchSize))
    src = dat[i:i+batchSize, :]
    label = np.zeros([1, len(speaker)])
    label[0, sind] = 1
    y = np.tile(label, [batchSize, 1])
    return src, y
  
def sythesis(path, src, trg, sp, filename):
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
    opname = src + '_To_' + trg + '_' + filename + '.wav'
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
    mu_s = np.log(np.mean(f0_s[f0_s>1]))
    mu_t = np.log(np.mean(f0_t[f0_t>1]))
    std_s = np.log(np.std(f0_s[f0_s>1]))
    std_t = np.log(np.std(f0_t[f0_t>1]))
    lf0 = np.where(f0_s > 1., np.log(f0_s), f0_s)
    lf0 = np.where(lf0 > 1., (lf0 - mu_s)/std_s * std_t + mu_t, lf0)
    f0_c = np.where(lf0 > 1., np.exp(lf0), lf0)
    f0_c = f0_c.reshape(-1)
    return f0_c
    
    
