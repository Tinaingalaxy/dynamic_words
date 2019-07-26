import argparse
import codecs
parser = argparse.ArgumentParser()
parser.add_argument('-train_src', type=str)
parser.add_argument('-train_tag', type=str)
parser.add_argument('-train_tgt', type=str)
parser.add_argument('-out', type=str)
parser.add_argument('-max_len', type=int)

args = parser.parse_args()


with codecs.open(args.train_src,'r',encoding='utf8') as src_f, \
    codecs.open(args.train_tag,'r',encoding='utf8') as tag_f, \
    codecs.open(args.train_tgt,'r',encoding='utf8') as tgt_f, \
    codecs.open(args.out,'wb',encoding='utf8') as out_f:
    for src_line,tag_line,tgt_line in zip(src_f,tag_f,tgt_f):
        src,tags,tgt = src_line.strip(),tag_line.strip(),tgt_line.strip()
        tags = tags.split(' ')
        if len(tags) > args.max_len:
            tags = tags[:args.max_len]
        
        for t in tags:
            out_f.write(src+'\t'+t+'\t'+tgt+'\n')
