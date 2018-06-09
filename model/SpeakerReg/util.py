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

def nextbatch(path, segmentLength, batchSize):
    speakerFolder = os.listdir(path)

    label_batch = np.zeros([1, len(speakerFolder)])
    x_batch = np.zeros([1, segmentLength])

    while(x_batch.shape[0] <= batchSize + 1):

        speaker = random.choice(speakerFolder)
        speakerLabel = np.zeros([1, len(speakerFolder)])
        speakerLabel[0, speakerFolder.index(speaker)] = 1

        Folderpath = path + speaker + '/'
        speakerFile = os.listdir(Folderpath)
        file = random.choice(speakerFile)
        raw = loadTF(Folderpath + file)
        length = len(raw)
        segment = raw[0: segmentLength * (length // segmentLength)]
        segment = segment.reshape([length // segmentLength, segmentLength])
        speakerLabel = np.tile(speakerLabel, [(length // segmentLength), 1])
        x_batch = np.concatenate((x_batch, segment), axis = 0)
        label_batch = np.concatenate((label_batch, speakerLabel), axis = 0)

    return x_batch[1:batchSize+1, :], label_batch[1:batchSize+1, :]

def pickOne(path, src, trg, segmentLength):

    speakerFolder = os.listdir(path)
    trg_label = np.zeros([1, len(speakerFolder)])
    trg_label[0, speakerFolder.index(trg)] = 1

    Folderpath = path + src + '/'
    speakerFile = os.listdir(Folderpath)
    file = random.choice(speakerFile)
    raw = loadTF( Folderpath + file)
    length = len(raw)
    segment = raw[0: segmentLength * (length // segmentLength)]
    segment = segment.reshape([length // segmentLength, segmentLength])
    trg_label = np.tile(trg_label, [(length // segmentLength), 1])

    return segment, trg_label, file
