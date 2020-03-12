#!/bin/bash

#*************************************************************************
#	> File Name: run.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Wed Mar 11 14:43:30 2020
# ************************************************************************/

######################################
# data_prepare
######################################

# ./data_prepare.sh

######################################
# kaldi cosine score
######################################
time=$(date "+%Y-%m-%d %H:%M:%S")
echo kaldi start $time

./kaldi_eer.sh

time=$(date "+%Y-%m-%d %H:%M:%S")
echo kaldi end $time

echo
echo

######################################
# python cosine score
######################################

time=$(date "+%Y-%m-%d %H:%M:%S")
echo python start $time

python -u score.py

time=$(date "+%Y-%m-%d %H:%M:%S")
echo python end $time

