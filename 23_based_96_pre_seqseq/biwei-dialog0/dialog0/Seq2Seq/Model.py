import torch
import torch.nn as nn
from torch.autograd import Variable


class Seq2SeqWithTag(nn.Module):
    def __init__(self, 
                 shared_embeddings, 
                 tag_encoder,
                 encoder,
                 decoder,
                 attention_dynamic,
                 generator, 
                 feat_merge,
                 use_tag=False):
        super(Seq2SeqWithTag, self).__init__()
        self.shared_embeddings = shared_embeddings
        self.tag_encoder = tag_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.use_tag = use_tag
        self.feat_merge = feat_merge
        self.attention_dynamic = attention_dynamic
    def forward(self, input):
        src_inputs = input[0]
        tgt_inputs = input[1]
        tag_inputs = input[2]
        src_lengths = input[3]
        # Run wrods through encoder

        # tag_inputs = tag_inputs.unsqueeze(dim = 0)
        encoder_outputs, encoder_hidden = self.encode(src_inputs, src_lengths, None)

        tag_hidden = self.tag_encode(tag_inputs)

        # tag_inputs = tag_inputs.cpu().data.numpy()[0].tolist()
        selected_tag_input = [tag_input.expand(1, tgt_inputs.size(1)) for tag_input in tag_inputs]
        # print("传入model之中的tag:",selected_tag_input)
        tag_hidden_list = [self.tag_encode(tag_) for tag_ in selected_tag_input]


        decoder_init_hidden = self.init_decoder_state((encoder_hidden,tag_hidden))
        # decoder_outputs, decoder_hiddens, attn_scores = \
        #             self.decode(
        #         tgt_inputs, tag_hidden, encoder_outputs, decoder_init_hidden
        #                         )
        # print(decoder_hiddens.shape)
        # print(attn_scores)
        decoder_outputs = self.word_by_word_decode( tgt_inputs, tag_hidden_list, encoder_outputs, decoder_init_hidden)

        return decoder_outputs

    def init_decoder_state(self, input):
        enc_hidden = input[0]
        tag_hidden = input[1]

        
        if not isinstance(enc_hidden, tuple):  # GRU
            h= enc_hidden
            # if self.feat_merge == 'sum':
            #     h = enc_hidden+tag_hidden.expand_as(enc_hidden)
            # elif self.feat_merge == 'concat':
            #     h = torch.cat([enc_hidden,tag_hidden.expand_as(enc_hidden)],dim=-1)
            return h

        else:  # LSTM
            h,c = enc_hidden
            # tag_hidden = tag_hidden.expand_as(c)
            # if self.feat_merge == 'sum':
            #     tag_hidden = tag_hidden.expand_as(c)
            #     h += tag_hidden
            #     c += tag_hidden
            # elif self.feat_merge == 'concat':
            #     h = torch.cat([h,tag_hidden.expand_as(h)],dim=-1)
            #     c = torch.cat([c,tag_hidden.expand_as(c)],dim=-1)
            return (h,c)

    def tag_encode(self, input):
        tag_embeddings = self.shared_embeddings(input)
        tag_hidden = self.tag_encoder(tag_embeddings)
        return tag_hidden


    def encode(self, input, lengths=None, hidden=None):
        encoder_input = self.shared_embeddings(input)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, lengths, None)

        return encoder_outputs, encoder_hidden

    def decode(self, input, tag_hidden, context, state):
        decoder_input = self.shared_embeddings(input)
        decoder_outputs, decoder_hiddens, attn_scores= self.decoder(
                decoder_input, tag_hidden, context, state
            )
        return decoder_outputs, decoder_hiddens, attn_scores

    def drop_checkpoint(self, epoch, opt, fname):
        torch.save({'shared_embeddings_dict': self.shared_embeddings.state_dict(),
                    'tag_encoder_dict':self.tag_encoder.state_dict(),
                    'encoder_dict': self.encoder.state_dict(),
                    'decoder_dict': self.decoder.state_dict(),
                    'generator_dict': self.generator.state_dict(),
                    'attention_dict':self.attention_dynamic.state_dict(),
                    'epoch': epoch,
                    'opt': opt,
                    },
                   fname)


    def load_checkpoint(self, cpnt):
        cpnt = torch.load(cpnt,map_location=lambda storage, loc: storage)
        self.shared_embeddings.load_state_dict(cpnt['shared_embeddings_dict'])
        self.tag_encoder.load_state_dict(cpnt['tag_encoder_dict'])
        self.encoder.load_state_dict(cpnt['encoder_dict'])
        self.decoder.load_state_dict(cpnt['decoder_dict'])
        self.generator.load_state_dict(cpnt['generator_dict'])
        self.attention_dynamic.load_state_dict(cpnt['attention_dict'])
        epoch = cpnt['epoch']
        return epoch

    def word_by_word_decode(self,tgt_inputs, tag_hidden_list, encoder_outputs, decoder_init_hidden):
        (row,column) = tgt_inputs.shape
        list_decoder_out = []
        decoder_hiddens_list = []
        attn_scores_list = []
        mask = torch.ones([1000])
        # print("正确的tagshape是：",tag_hidden_list[0].shape)
        tag_hidden_all = torch.cat(tag_hidden_list,dim=0)
        # print("tag_hidden_all.shape,",tag_hidden_all.shape)
        # print("在输入attention_by_dynamic之前的init_hidden：",decoder_init_hidden.shape)
        hidden_last = decoder_init_hidden
        for idx in range(row):
            c_t, attn_dist = self.attention_dynamic(hidden_last, tag_hidden_all, mask)
            c_t = c_t.transpose(dim0 = 1,dim1 = 0)
            # print("c_t,attn_dist的shape分别是：", c_t.shape, attn_dist.shape)

            tgt_input_now = tgt_inputs[idx]
            tgt_input_now = tgt_input_now.unsqueeze(0)
            # print("查看decoder_out_put的shape,看和batch是否有关",tgt_input_now.shape)
            decoder_outputs, decoder_hiddens, attn_scores \
                = self.decode(
                tgt_input_now, c_t, encoder_outputs, decoder_init_hidden
            )
            # print("看是不是完成了decode")
            #

            # print("\t\t每一步word_by_word的decoder_hiddens:",decoder_hiddens.shape,)
            list_decoder_out.append(decoder_outputs)
            decoder_hiddens_list.append(decoder_hiddens)
            attn_scores_list.append(attn_scores)


            hidden_last = decoder_hiddens
        decoder_outputs_cat = torch.cat(list_decoder_out,dim = 0)
        decoder_hiddens_out = torch.cat(decoder_hiddens_list,dim = 0)
        # attn_scores_out = torch.cat(attn_scores_list)
        # print("\t最后的word_by_word的decoder_hiddens:", decoder_hiddens_out.shape, )
        # print("*"*90)
        # print("\tword_by_word的decoder_outs:", decoder_outputs_cat.shape)
        return decoder_outputs_cat
        # os._exit(0)
