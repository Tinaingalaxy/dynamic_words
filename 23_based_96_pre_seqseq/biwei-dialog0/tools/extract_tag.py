import codecs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-src_in', type=str)
parser.add_argument('-tgt_in', type=str)
parser.add_argument('-tag_out', type=str)
args = parser.parse_args()


with codecs.open(args.src_in,'r',encoding='utf8',errors='ignore') as src_f, \
    codecs.open(args.tgt_in,'r',encoding='utf8',errors='ignore') as tgt_f, \
    codecs.open(args.tag_out,'wb',encoding='utf8') as tag_f:
    for src,tgt in zip(src_f,tgt_f):
        tag_list = []
        words_in_src = src.strip().split(' ')
        tag_list.extend(words_in_src)
        words_in_tgt = tgt.strip().split(' ')
        tag_list.extend(words_in_tgt)
        tag_set = set(tag_list)
        tag_list = list(tag_set)
        tag_f.write(' '.join(tag_list)+'\n')

print('done!')

    