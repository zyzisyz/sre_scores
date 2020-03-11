#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
#             2019   Lantian Li
# Apache 2.0.
#
# This is an x-vector-based recipe for Speakers in the Wild (SITW).

. ./local/path.sh

trails=./data/test/test.lst

# Cosine metric.
local/cosine_scoring.sh \
	./data/enroll \
	./data/test \
	$trails \
	./foo

eer=$(paste $trails ./foo/cosine_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "kaldi Cosine EER: $eer%"
