import codecs
import dialog0.IO as IO
import torch
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument("-data", type=str)
parser.add_argument("-tag", type=str)
parser.add_argument("-new_tag", type=str)
parser.add_argument("-len", type=int)
args = parser.parse_args()

def load_fields(train):
    fields = IO.load_fields(
                torch.load(args.data + '.vocab.pkl'))
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train.examples[0].__dict__])
    train.fields = fields

    print(' * vocabulary size. source = %d; target = %d; tag = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab), len(fields['tag'].vocab)))

    return fields
def main():
    train = torch.load(args.data + '.train.pkl')
    fields = load_fields(train)
    with codecs.open(args.tag,'r',encoding='utf8',errors='ignore') as tag_f, \
        codecs.open(args.new_tag,'wb',encoding='utf8') as new_tag_f:
        for tag_line in tag_f:
            tags = tag_line.strip().split(' ')
            tag_indices = [fields['tag'].vocab.stoi[t] for t in tags]
            new_labels = ['1' for i in range(len(tags))]
            while len(tag_indices) < args.len:
                random_int = random.randint(4,49999)
                if random_int in tag_indices:
                    continue
                else:
                    tag_indices  = tag_indices + [random_int]
                    new_labels = new_labels + ['0']
            new_tags = [fields['tag'].vocab.itos[t] for t in tag_indices]
            if len(new_tags) == args.len:
                new_tag_f.write(' '.join(new_tags)+'\t'+' '.join(new_labels)+'\n')
            else:
                raise "not match length "

if __name__ == '__main__':
    main()

