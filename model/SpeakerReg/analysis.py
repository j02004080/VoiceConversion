import numpy as np
import scipy.io as sio
import os
import random
import pyworld as pw


def loadData(path, L, tstep):
    Data = {}
    for speaker in os.listdir(path):
        spath = path+speaker+'/'
        filename = os.listdir(spath)
        mat = sio.loadmat(spath + filename[0])
        mat = mat['value']
        length = L*tstep
        tmp = np.reshape(mat[0: length* (mat.shape[0] // length)], [-1, length])
        Data[speaker] = tmp
        for name in filename[1:len(filename)]:
                mat = sio.loadmat(spath+name)
                mat = mat['value']
                tmp = np.reshape(mat[0:length * (mat.shape[0] // length)], [-1, length])
                Data[speaker] = np.concatenate((Data[speaker], tmp), axis=0)
    return Data


def pickTransferInput(path, src, L, tstep):
    srcPath = path+src+'/'
    srcfile = os.listdir(srcPath)
    ind = random.randint(0, len(srcfile)-1)
    smat = sio.loadmat(srcPath+srcfile[ind])
    smat = smat['value']
    length = L*tstep
    tmp = np.reshape(smat[0:length * (smat.shape[0] // length)], [-1, length])
    return tmp, srcfile[ind][0:len(srcfile[ind])-4]

def nextbatch(data, batch):
    s = list(data.keys())
    label = np.zeros([batch, len(s)])
    speaker = random.choice(s)
    ind = random.randint(0, len(data[speaker])-1)
    srcData = data[speaker][ind]
    label[0, s.index(speaker)] = 1
    for i in range(1, batch):
        speaker = random.choice(s)
        ind = random.randint(0, len(data[speaker])-1)
        srcData = np.concatenate((srcData, data[speaker][ind]))
        label[i, s.index(speaker)] = 1
    return srcData.reshape([batch, data[speaker].shape[1]]), label

def sythesis(path, src, trg, sp, filename):
    file = path+src+'/' + filename
    mat = sio.loadmat(file)
    ap = mat['ap']
    ap = ap.copy(order='C')
    Nfactor = mat['Nfactor']
    sp = np.multiply(sp, np.transpose(np.tile(Nfactor, (513, 1))))
    fs = 16000
    f0_c = convert_f0(path, src, trg, filename)
    f0_c = f0_c.copy(order='C')
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
    
    
