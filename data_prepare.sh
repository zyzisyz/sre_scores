#!/bin/bash

#*************************************************************************
#	> File Name: data_prepare.sh
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com 
#	> Created Time: Wed Mar 11 22:47:14 2020
# ************************************************************************/

for sub in enroll test enroll-split
do
	echo prepare $sub
	python -u local/ark2npz.py \
		--src_file ./data/$sub/xvector.ark \
		--dest_file ./data/$sub/xvector.npz \
		--utt2spk ./data/$sub/utt2spk
done

