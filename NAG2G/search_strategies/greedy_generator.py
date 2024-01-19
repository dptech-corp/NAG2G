import math
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from torch import Tensor
# from . import move_to_cuda, strip_pad

import logging
import math
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from dpaie import search, utils
# from fairseq.data import data_utils
# from fairseq.models import FairseqIncrementalDecoder
# from dpaie.ngram_repeat_block import NGramRepeatBlock

logger = logging.getLogger(__name__)

'''
Translate algorithms, whitch supprot a batch input.
    algorithms:
     - greedy search (when args.beam_size <= 0)
     - beam search (when args.beam_size > 0. Support to adjust 
                    these parameters: beam_size and length_penalty)
    
    inputs:
     - src_tokens: (batch_size, src_len)
    outputs:
     - gen_seqs: (batch_size, max_seq_len/tgt_len.max()) (related to the stop rules)
    
'''

"""
Referenced from facebookresearch/XLM,
 at https://github.com/facebookresearch/XLM/blob/master/xlm/model/transformer.py
"""
class BeamHypotheses(object):
    def __init__(self, n_hyp, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """

        lp = len(hyp) ** self.length_penalty # deafult length penalty
        # lp = (5 + len(hyp)) ** self.length_penalty / (5 + 1) ** self.length_penalty # Google GNMT's length penalty
        score = sum_logprobs / lp
        # score = sum_logprobs
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]] # delete the worst hyp in beam
                self.worst_score = sorted_scores[1][0] # update worst score with the sencond worst hyp
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.n_hyp:
            return False
        else:
            return self.worst_score >= best_sum_logprobs 
            # / cur_len ** self.length_penalty
            # return self.worst_score >= best_sum_logprobs / cur_len ** self.length_penalty

class GreedyGenerator(nn.Module):
    def __init__(
        self,
        model,
        dict,
        output_num_return_sequences_per_batch  = 10,
        max_seq_len=180,
        beam_size=5,
        temperature=1.0,
        match_source_len=False,
    ):
        super().__init__()
        self.model = model
        self.dict = dict
        self.output_num_return_sequences_per_batch = beam_size
        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.temperature = temperature
        self.match_source_len = match_source_len 
        self.pad_idx = self.dict.pad()
        self.bos_idx = self.dict.bos()
        self.eos_idx = self.dict.eos()

    def _generate(self, sample):

        self.model.eval()

        src_tokens = sample['net_input']['reverse_src_dataset'] # B x T 
        reverse_tgt_tokens = sample['net_input']['src_tokens'] # B x T 
        reaction_type = sample['net_input']['reaction_type']   
        src_lengths = torch.sum(src_tokens.ne(self.pad_idx), dim=1)
        batch_size = src_tokens.size(0)
        src_padding_mask = src_tokens.eq(self.pad_idx)
        encoder_out, padding_mask = self.model.forward_encoder(src_tokens, reaction_type)
        
        # expand to beam size the source latent representations
        encoder_out = encoder_out.repeat_interleave(self.beam_size, dim=0) # (B x beam_size) x T x C
        reverse_tgt_tokens = reverse_tgt_tokens.repeat_interleave(self.beam_size, dim=0) # (B x beam_size) x T x C
        src_padding_mask = src_padding_mask.repeat_interleave(self.beam_size, dim=0) # (B x beam_size) x T x C
        # generated sentences (batch with beam current hypotheses)
        generated = src_tokens.new(batch_size * self.beam_size, self.max_seq_len).fill_(self.pad_idx)  # upcoming output
        generated[:, 0].fill_(self.bos_idx)

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(self.beam_size *2, 1.0)
                for _ in range(batch_size)
        ] # 为每个输入句子定义维护其beam search序列的类实例

        # scores for each sentence in the beam
        beam_scores = encoder_out.new(batch_size, self.beam_size).fill_(0).to(src_tokens.device) # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9 # 需要初始化为-inf

        all_scores, avg_attn_scores = self.model.forward_decoder(reverse_tgt_tokens, encoder_out, self.temperature, padding_mask = src_padding_mask)  # (batch_size * beam_size, n_tgt_words) 
        # current position
        cur_len = 1
        # done sentences
        n_tgt_words = all_scores.size(-1)
        pre_len = all_scores.size(1)
        # - scores: (batch_size * beam_size, n_tgt_words) 

        done = [False] * batch_size # 标记每个输入句子的beam search是否完成


        while cur_len < self.max_seq_len and cur_len < pre_len:
 
            scores = all_scores[:,cur_len,:]
            _scores = scores + beam_scores.view(batch_size * self.beam_size, 1)  # 累加上以前的scores
            _scores = _scores.view(batch_size, self.beam_size * n_tgt_words) # (batch_size, beam_size * vocab_size)   
            next_scores, next_words = torch.topk(_scores, 2 * self.beam_size, dim=-1, largest=True, sorted=True)  
            # - next_scores, next_words: (batch_size, 2 * beam_size)  
            # next batch beam content
            next_batch_beam = []  # list of (batch_size * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            # for each sentence
            for sent_id in range(batch_size):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(next_scores[sent_id].max().item(), cur_len)
                if done[sent_id]:
                    next_batch_beam.extend([(0, self.pad_idx, 0)] * self.beam_size)  # pad the batch
                    continue
                # next sentence beam content
                next_sent_beam = []
                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    # get beam and word IDs

                    beam_id = idx // n_tgt_words
                    word_id = idx % n_tgt_words    
      
                    # end of sentence, or next word
                    effective_beam_id = sent_id * self.beam_size + beam_id
                    if word_id == self.eos_idx or cur_len + 1 == self.max_seq_len:
                        generated_hyps[sent_id].add(generated[effective_beam_id, :cur_len].clone(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, effective_beam_id))
                    # the beam for next step is full
                    if len(next_sent_beam) == self.beam_size:
                        break
                # update next beam content
                if len(next_sent_beam) == 0:
                    next_sent_beam = [(0, self.pad_idx, 0)] * self.beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
            # prepare next batch
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])  
            beam_words = generated.new([x[1] for x in next_batch_beam])     
            beam_idx = src_tokens.new([x[2] for x in next_batch_beam])      
            # re-order batch and internal states
            generated = generated[beam_idx, :]                              
            generated[:, cur_len] = beam_words                              
            # update current length
            cur_len = cur_len + 1   
            if all(done):
                break
        # select the best hypotheses
        tgt_len = src_tokens.new(batch_size*self.output_num_return_sequences_per_batch)
        best = []
        best_score = []
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.hyp, key=lambda x: x[0])

            for j in range(self.output_num_return_sequences_per_batch):
                try:
                    effective_batch_idx = self.output_num_return_sequences_per_batch * i + j
                    best_cand = sorted_hyps.pop()
                    best_hyp = best_cand[1]
                    score = best_cand[0]
                    tgt_len[effective_batch_idx] = len(best_hyp)
                    best.append(best_hyp)
                    best_score.append(score)
                except:
                    tgt_len[effective_batch_idx] = 0
                    best.append(torch.tensor([]))
                    best_score.append(-1000)
        # generate target batch
        gen_seqs = src_tokens.new(batch_size*self.output_num_return_sequences_per_batch, tgt_len.max().item()).fill_(self.pad_idx)
        gen_scores = [-1e5] * len(gen_seqs)
        for i, hypo in enumerate(best):
            gen_seqs[i, :tgt_len[i]] = hypo
            gen_scores[i] = best_score[i]
        tgt_lengths = torch.sum(gen_seqs.ne(self.pad_idx), dim=1)
        return gen_seqs, tgt_lengths, gen_scores