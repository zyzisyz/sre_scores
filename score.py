#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: score.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue Mar 10 23:05:53 2020
# ************************************************************************/

import os
import numpy as np
import math
from scipy.spatial import distance
from scipy import stats as sts
pi = np.array(np.pi)

def read_trails(path):
    '''
    what's trails?
    Trial file is formatted as:
    <spkid(enroll_data)> <uttid(test_data)> target|nontarget 
    If uttid belong to spkid, it is marked 'target', otherwise is 'nontarget'.
    '''
    assert os.path.exists(path)

    trails = np.loadtxt(path, dtype=str, comments='#', delimiter=" ")
    spkid = []
    uttid = []
    target = []
    for it in trails:
        spkid.append(it[0])
        uttid.append(it[1])
        # target == 1; nontarget == 0
        if it[2] == "target":
            target.append(1)
        else:
            target.append(0)
    spkid = np.array(spkid)
    uttid = np.array(uttid)
    target = np.array(target)
    return spkid, uttid, target


def compute_eer(target_scores, nontarget_scores): 
    '''
    lilt's toolkit
    '''
    if isinstance(target_scores , list) is False:
        target_scores = list(target_scores)
    if isinstance(nontarget_scores , list) is False:
        nontarget_scores = list(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores);
    nontarget_size = len(nontarget_scores)

    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position];
    eer = target_position * 1.0 / target_size;
    return eer, th

def cosine_score(enroll_mats, test_mat):
    '''
    cosine_dist
    an enroll spk may have more than one vector
    a test utt only have one vector
    '''
    enroll_mats = np.array(enroll_mats, dtype=float)
    enroll_mean = np.mean(enroll_mats, axis=0)
    test_mat = np.array(test_mat, dtype=float)

    # dim
    d = len(test_mat)
   
    sumData = enroll_mean * test_mat.T
    sumData = sumData.sum()
    denom = np.linalg.norm(enroll_mean)*np.linalg.norm(test_mat)
    score = (sumData / denom)*d

    return score


def euclidean_score(enroll_mats, test_mat):
    '''
    euclidean_score
    an enroll spk may have more than one vector
    a test utt only have one vector
    '''
    enroll_mats = np.array(enroll_mats, dtype=float)
    enroll_mean = np.mean(enroll_mats, axis=0)
    test_mat = np.array(test_mat, dtype=float)

    score = distance.euclidean(enroll_mean, test_mat[0])
    return score



def Gaussian_log_likelihood(enroll_mats, test_mat):
    enroll_mats = np.array(enroll_mats, dtype=float)
    enroll_mean = np.mean(enroll_mats, axis=0)
    
    enroll_var = np.var(enroll_mats, axis=0)

    test_mat = np.array(test_mat, dtype=float)

    log_det_sigma = np.log(enroll_var+1e-15).sum()
    log_probs = -0.5 * ((pow((test_mat-enroll_mean),2)/(enroll_var+1e-15) + np.log(2 * pi) ).sum() + log_det_sigma)
    return log_probs


def score_Gaussian(data, label, mean_class, var_global):
    logp_index = []
    lp_tensor_list=[]
    for i in range(data.size()[0]):
        log_probs = Gaussian_log_likelihood(data[i], mean_class, var_global)
        lp_tensor_list.append(log_probs)
        max_index = torch.argmax(log_probs,dim=0)
        logp_index.append(max_index)
    logp_index = torch.cat(logp_index,0)
    label_mask = torch.eq(label, logp_index).cpu().detach().numpy()
    logp_accuracy = label_mask.sum() / len(label_mask)
    logp_scores = torch.cat(lp_tensor_list,0)
    return logp_accuracy, logp_scores


def lda_nl_score(enroll_mats, test_mat, epsilon):
    '''
    normlize likehood score
    an enroll spk may have more than one vector
    a test utt only have one vector
    epsilon is the val_global
    '''
    enroll_mats = np.array(enroll_mats, dtype=float)
    enroll_mean = np.mean(enroll_mats, axis=0)
    test_mat = np.array(test_mat, dtype=float)

    nk = len(enroll_mats)
    DATA_DIM = len(enroll_mats[0])

    uk = nk*epsilon / (nk*epsilon+1) * enroll_mean


def main(enroll_npz, test_npz, trails_path):
    '''
    this the main function to compute the score and eer
    '''
    # check
    assert os.path.exists(enroll_npz)
    assert os.path.exists(test_npz)
    assert os.path.exists(trails_path)

    # trails: <spkid(enroll_data)> <uttid(test_data)> target|nontarget 
    spkid, uttid, target = read_trails(trails_path)
    trails_len = len(target)

    # load enroll data
    enroll = np.load(enroll_npz)
    enroll_data = enroll['feats']
    enroll_spk_label = enroll['spker_label']
    enroll_utt_label = enroll['utt_label']

    # load test data
    test = np.load(test_npz)
    test_data = test['feats']
    test_spk_label = test['spker_label']
    test_utt_label = test['utt_label']

    # build hashmap enroll_spk -> vectors (enroll_spk2mats)
    # an enroll_spk may have more than one vector
    enroll_spk2mats = {}
    for idx in range(len(enroll_spk_label)):
        label = enroll_spk_label[idx]
        if label not in enroll_spk2mats:
            enroll_spk2mats[label] = []
        enroll_spk2mats[label].append(enroll_data[idx])

    # build hashmap test_utt -> vector (test_utt2mat)
    # a test utt only have one vector
    test_utt2mat = {}
    for idx in range(len(test_utt_label)):
        label = test_utt_label[idx]
        test_utt2mat[label] =  test_data[idx]

    print("successfully load data and build hashmap")


    epsilon = np.std(enroll_data, axis=0)

    # computer score
    target_scores = []
    nontarget_scores = []
    for i in range(trails_len):
        enroll_mats = enroll_spk2mats[spkid[i]]
        test_mat = test_utt2mat[uttid[i]]
        score = cosine_score(enroll_mats, test_mat)
        if(target[i]):
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    print("scoring is done, now compute the EER and Threshold")
    eer, th = compute_eer(target_scores, nontarget_scores)
    print("python Cosine EER: {:.2f}%".format(eer*100.0))
    print("Threshold: {:.2f}".format(th))


if __name__ == "__main__":
    enroll_npz = "data/enroll-split/xvector.npz"
    test_npz = "data/test/xvector.npz"
    trails_path = "data/test/core-core.lst"
    main(enroll_npz, test_npz, trails_path)

