import numpy as np
import scipy.io as sio
import os
import random
import tensorflow as tf
import pyworld as pw
import math
import sys

def loadData(path):
    Data = {}
    for speaker in os.listdir(path):
        spath = path+speaker+'/'
        filename = os.listdir(spath)
        mat = sio.loadmat(spath+filename[0])
        Data[speaker] = mat['sp']
        for name in filename[1:len(filename)]:     
                mat = sio.loadmat(spath+name)
                Data[speaker] = np.concatenate((Data[speaker], mat['sp']), axis = 0)
    return Data

def pickTransferInput(path, src, trg):
    srcPath = path+src+'/'
    srcfile = os.listdir(srcPath)
    speaker = os.listdir(path)
    ind = random.randint(0, len(srcfile)-1)
    smat = sio.loadmat(srcPath+srcfile[ind])
    srcData = np.reshape(smat['sp'], [-1, 513, 1, 1])
    y = [speaker.index(trg)]
    return srcData, y, srcfile[ind]

def nextbatch(dat, mini_batch):
    
    source = random.choice(list(dat.keys()))
    y = [list(dat.keys()).index(source)]
    ind = random.randint(0, len(dat[source])-128)
    src = np.transpose(dat[source][ind:ind+128])


    target = random.choice(list(dat.keys()))
    y = [list(dat.keys()).index(target)]
    ind = random.randint(0, len(dat[target])-128)
    trg = np.transpose(dat[target][ind:ind+128])

    
    return src, y, trg
  
def sythesis(path, src, trg, sp, filename):
    file = path+src+'/' + filename
    mat = sio.loadmat(file)
    ap = mat['ap']
    ap = ap.copy(order='C')
    Nfactor = mat['Nfactor']
    sp = np.multiply(sp, np.transpose(np.tile(Nfactor, (513, 1))))
    fs = 16000
    f0_c = convert_f0(path, src, trg, filename)
    # f0_c = f0_c.copy(order='C')
    y = pw.synthesize(f0_c, sp, ap, fs)
    opname = src + '_To_' + trg + '_' + filename[0:len(filename)-4] + '.wav'
    sio.savemat(opname, mdict={'y': y})
    return sp, ap, f0_c
    

def convert_f0(path, src, trg, filename):
    srcfile = path+src+'/' + filename
    trgfile = path+trg+'/' + filename
    smat = sio.loadmat(srcfile)
    tmat = sio.loadmat(trgfile)
    f0_s = smat['f0']
    f0_t = tmat['f0']
    mu_s = np.log(np.mean(f0_s[f0_s>1]))
    mu_t = np.log(np.mean(f0_t[f0_t>1]))
    std_s = np.log(np.std(f0_s[f0_s>1]))
    std_t = np.log(np.std(f0_t[f0_t>1]))
    lf0 = np.where(f0_s > 1., np.log(f0_s), f0_s)
    lf0 = np.where(lf0 > 1., (lf0 - mu_s)/std_s * std_t + mu_t, lf0)
    f0_c = np.where(lf0 > 1., np.exp(lf0), lf0)
    f0_c = f0_c.reshape(-1)
    return f0_c
    
    
