#!/usr/bin/env python

from __future__ import division, unicode_literals
import os
import argparse
import math
import codecs
import torch
from torch.autograd import Variable
import numpy as np
from itertools import count

import onmt.io
import onmt.translate
import onmt
import onmt.ModelConstructor
import onmt.modules
import opts
import torch.nn as nn

parser = argparse.ArgumentParser(
    description='attack.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
opts.add_md_help_argument(parser)
opts.attack_opts(parser)

opt = parser.parse_args()

def attack(all_word_embedding, label_onehot, translator, src, batch, new_embedding, input_embedding, modifier, const, GROUP_LASSO, TARGETED, GRAD_REG, NN):
    if TARGETED:
        lr_a = [0.5,1]
    else:
        lr_a = [2]
    if NN:
        lr_a= [0.1] 
    cur_best = Variable(torch.zeros(1)).cuda()
    cur_best.data[0] = 999
    cur_best_modi = 999
#    FLAG = False
    m = label_onehot.size()[0]
    for lr in lr_a:
        CFLAG = True
        for k in range(200):
            #loss1=0
            new_word_list=[]
            loss1 = Variable(torch.zeros(1)).cuda()
            loss2 = Variable(torch.zeros(1)).cuda()
            if NN:
                for i in range(input_embedding.size()[0]):
                    new_embedding[i] = modifier[i] + input_embedding[i]
            else:
                for i in range(input_embedding.size()[0]):
                    new_embedding[i] = modifier[i] + input_embedding[i]
                    new_placeholder = new_embedding[i].data
                    temp_place = new_placeholder.expand_as(all_word_embedding)
                    new_dist = torch.norm(temp_place - all_word_embedding.data, 2 ,2)
                    v_dist = Variable(new_dist, requires_grad = True)
                    _ , new_word = torch.min(new_dist,0)
                    min_dist, _ = torch.min(v_dist, 0)
                    new_word_list.append(new_word)
                    new_embedding.data[i]  = all_word_embedding[new_word[0]].data
                    del temp_place
           
            output_a, attn, output_i= translator.getOutput(new_embedding, src, batch)
            if TARGETED:
                n = output_a.size()[0]
                mask = None
                for iter_ind in range(m):
                    if mask:
                        if mask == output_a.size()[0]-1:
                            #print("b")
                            #print(t_loss.data[0],mask,output_a.size()[0], fake_onehot.size()[0])
                            output_a = output_a[0:mask,:]
                            fake_onehot = fake_onehot[0:mask,:]
                        else:
                            output_a = torch.cat((output_a[0:mask,:],output_a[mask+1:,:]))
                            fake_onehot = torch.cat((fake_onehot[0:mask,:], fake_onehot[mask+1:,:]))
                    mask = None
                    placeholder = label_onehot[iter_ind].clone()
                    fake_onehot = placeholder.expand_as(output_a)
                    real, reali = torch.max (torch.mul(output_a, fake_onehot),1)
                    other, otheri = torch.max(torch.mul(output_a, (1-fake_onehot)) - fake_onehot*10000, 1)
                    t_loss, t_pos = torch.min(torch.clamp(other-real, min=0),0)
                    if t_loss.data[0] < 0:
                        mask = t_pos.data[0]
            #            print(mask)
            #            if FLAG:
            #                print(t_loss.data)
                    loss1 = loss1 + t_loss
            else:
                if output_a.size()[0] > label_onehot.size()[0]:
                    output_a = output_a[:label_onehot.size()[0],:]
                else:
                    label_onehot = label_onehot[:output_a.size()[0],:]
                real, reali = torch.max(torch.mul(output_a, label_onehot),1)
                other, otheri = torch.max(torch.mul(output_a, (1-label_onehot)),1)
                loss1 = torch.sum(torch.clamp(real-other,min=0))            
    
            #print(loss1.data[0],"\t", torch.norm(modifier.data))
            if loss1.data[0]<= 0 :
                #print(loss1.data[0],"\t", torch.norm(modifier.data))
                if torch.norm(modifier.data) < cur_best_modi:
                    print(loss1.data[0],"\t", torch.norm(modifier.data))
                    cur_best_modi = torch.norm(modifier.data)
                    best_word = new_word_list
                    best_output_a = output_a.clone()
                    best_attn = attn
                    best_output_i = output_i.clone()
#               
                #break
            if loss1.data[0] < cur_best.data[0]:
                print(cur_best.data[0],"\t", torch.norm(modifier.data))
                cur_best = loss1.clone()
                best_word = new_word_list
                best_output_a = output_a.clone()
                best_attn = attn
                best_output_i = output_i.clone()
#                FLAG = True
#            else:
#                FLAG = False
        #            print(cur_best.data[0],"\t", torch.norm(modifier.data))
            if k == 199:
                new_word_list = best_word
                output_a = best_output_a
                attn = best_attn
                output_i = best_output_i
                if cur_best.data[0] <= 0:
                    break
                CFLAG = False
                print("lr=",lr)
            loss2 = torch.max(modifier)
            #loss2 = torch.sum(modifier * modifier)
            if GRAD_REG:
                loss = const * loss1 + min_dist + loss2
            else:
                loss = const * loss1 + loss2
            loss.backward(retain_graph=True)
            modifier.data -= lr * modifier.grad.data
            if GROUP_LASSO:
                gamma = lr 
                l2dist = torch.norm(modifier, 2, 2)
                for j in range(input_embedding.size()[0]):
                    if l2dist.data[j][0] > gamma * const:
                        modifier.data[j] = modifier.data[j] - gamma*const* modifier.data[j]/l2dist.data[j][0]
                    else:
                        modifier.data[j] = torch.zeros(1,500).cuda()
            modifier.grad.data.zero_()    
        if CFLAG:
            break
    return modifier, output_a, attn, new_word_list, output_i, CFLAG


def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    #print(opt)
    # Load the model.
    fields, model, model_opt = \
        onmt.ModelConstructor.load_test_model(opt, dummy_opt.__dict__)
    #print(model_opt)
    n_src = len(fields['src'].vocab) 
    n_tgt = len(fields['tgt'].vocab)
    # File to write sentences to.
    out_file = codecs.open(opt.output, 'w', 'utf-8')

    # Test data
    data = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tgt,
                                 src_dir=opt.src_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)

    test_data = onmt.io.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=1, train=False, sort=False,
        shuffle=False)

    # Translator
    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha, opt.beta)
    translator = onmt.translate.Translator(model, fields,
                                           beam_size=opt.beam_size,
                                           n_best=opt.n_best,
                                           global_scorer=scorer,
                                           max_length=opt.max_sent_length,
                                           copy_attn=model_opt.copy_attn,
                                           cuda=opt.cuda,
                                           beam_trace=opt.dump_beam != "")
    builder = onmt.translate.TranslationBuilder(
        data, translator.fields,
        opt.n_best, opt.replace_unk, opt.tgt)

    # Statistics
    counter = count(1)
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    
    pdist = nn.PairwiseDistance(p=2)
    if opt.tar_dir:
        TARGETED = True
    else:
        TARGETED = False
    GROUP_LASSO = opt.gl
    GRAD_REG = opt.gr
    NN = opt.nn
    const = 10
    if TARGETED:
        targets_list = []
        tar = onmt.io.build_dataset(fields, opt.data_type,
                                 opt.src, opt.tar_dir,
                                 src_dir=opt.tar_dir,
                                 sample_rate=opt.sample_rate,
                                 window_size=opt.window_size,
                                 window_stride=opt.window_stride,
                                 window=opt.window,
                                 use_filter_pred=False)
        tar_data = onmt.io.OrderedIterator(
            dataset=tar, device=opt.gpu,
            batch_size=1, train=False, sort=False,
            shuffle=False)
    
    all_index = Variable(torch.LongTensor(range(n_src)).view(n_src,1,1).cuda())
    all_word_embedding, _ = translator.getEmbedding(all_index, FLAG=False)
 
    for batch in test_data: 
        batch_data = translator.translate_batch(batch, data)
        predBatch = batch_data["predictions"]
        translations = builder.from_batch(batch_data)
        if TARGETED:
            target_list = []
            for target in tar_data:
                target_inds = onmt.io.make_features(target, 'tgt')
                target_list.append(target_inds.data.cpu().view(-1,1))
            #print(target_list)
            true_label = target_list[0] 
            #print(true_label)
        else:    
            label_data = translator.translate_batch(batch, data)
            pred = label_data["predictions"]
            true_label=torch.LongTensor(pred[0][0]).view(-1,1)
        label_onehot = torch.FloatTensor(true_label.size()[0], n_tgt)
        label_onehot.zero_()
        label_onehot.scatter_(1,true_label,1)
        if TARGETED:
            label_onehot = label_onehot[1:-1,:]
        label_onehot = Variable(label_onehot, requires_grad = False).cuda()
        #print(label_onehot)
        #print(batch)
        input_embedding, src= translator.getEmbedding(batch)
        #print(src)
        hidden_size = input_embedding.size()[2]
        
        if GROUP_LASSO:
            modifier_initial = torch.zeros(input_embedding.size()).cuda()
        else:
            modifier_initial = torch.zeros(input_embedding.size()).cuda()
        modifier = Variable(modifier_initial, requires_grad = True)
        #print(input_embedding)
        new_embedding = input_embedding.clone()
        modifier, output_a, attn, new_word, output_i, CFLAG = attack(all_word_embedding, label_onehot, translator, src, batch, new_embedding, input_embedding, modifier, const, GROUP_LASSO, TARGETED, GRAD_REG, NN)
        words_list = builder.get_word(output_i, attn, batch)
        
        print(words_list)
        new_embedding = input_embedding.clone()
        new_embedding = modifier + input_embedding
        if NN:
            changed_words=[]
            for i in range(input_embedding.size()[0]):
                dis = []
                for dic_embedding_index in range(all_word_embedding.size()[0]):
                    #if dic_embedding_index == src.data[index][0][0]:
                    #    continue
                    new_dist  = pdist(new_embedding[i], all_word_embedding[dic_embedding_index])
                    dis.append(new_dist.data[0][0])
                print(min(dis), np.argmin(dis))
                changed_words.append(np.argmin(dis))
            print(changed_words)
            new_word = changed_words
        #print(new_word)
        newsrc = src.clone()
        for i in range(input_embedding.size()[0]):
            newsrc.data[i][0] = new_word[i]
        print(builder.get_source(newsrc, batch))
        new_pred = translator.translate_batch(batch,data, newsrc=newsrc, FLAG=False)
        predBatch = builder.from_batch(new_pred)
        for trans in predBatch:
            n_best_preds = [" ".join(pred) for pred in trans.pred_sents[:opt.n_best]]
        for trans in translations:
            o_preds = [" ".join(pred) for pred in trans.pred_sents[:opt.n_best]]
        
        print(n_best_preds)
        out_file.write(''.join(builder.get_source(newsrc, batch)))
        out_file.write('\t\t')
        out_file.write(n_best_preds[0])
        out_file.write('\t\t')
        out_file.write(o_preds[0])
        out_file.write('\n')
        out_file.flush()
if __name__ == "__main__":
    main()
