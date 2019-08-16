DIR=./biwei-dialog0
python3 ${DIR}/infer_seq2seq.py \
    -test_data /mnt/mnt8tsdc/limiaojin/23_based_96_pre_seqseq/data/weibo/unique_src.txt \
    -test_out ./out/23_out \
    -vocab ./data/weibo/weibo.vocab.pkl \
    -seq2seq /mnt/mnt8tsdc/limiaojin/23_based_96_pre_seqseq/out/checkpoint_epoch9.pkl \
    -sampler /mnt/mnt8tsdc/limiaojin/96_pre_sampler/96_out/sampler/checkpoint_epoch9.pkl \
    -config ./config.yml \
    -beam_size 3 \
    -gpuid 0 \
    -topk_tag 1000 \
    -decode_max_length 20 \
    -num_cluster 3 \
    -tensorboard ./test/log/test2.json
