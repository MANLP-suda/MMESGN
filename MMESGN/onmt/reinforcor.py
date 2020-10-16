#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import configargparse
import onmt.opts as opts
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import onmt.transformer as nmt_model
from inputters.dataset import build_dataset, OrderedIterator, make_features
from onmt.beam import Beam
from utils.misc import tile
import onmt.constants as Constants

import time
import numpy as np
from sklearn import metrics

from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim

from utils.logging import logger
from utils.distributed import all_gather_list, all_reduce_and_rescale_tensors

def build_reinforcor(model,fields,opt,model_saver,optim):
	# dummy_parser = configargparse.ArgumentParser(description='reinforce.py')
	# opts.model_opts(dummy_parser)
	# dummy_opt = dummy_parser.parse_known_args([])[0]
	# # build the model and get the checkpoint and field
	# fields, model = nmt_model.load_reinforce_model(opt, dummy_opt.__dict__)

	reinforcor = Reinforcor(model, fields, opt,model_saver,optim)

	return reinforcor

class Reinforcor(object):
	def __init__(self, model, fields, opt, model_saver,optim,out_file=None):
		self.model = model
		self.fields = fields
		self.gpu = opt.gpu
		self.cuda = opt.gpu > -1
		self.device = torch.device('cuda' if self.cuda else 'cpu')
		self.decode_length = opt.decode_length
		self.decode_min_length = opt.decode_min_length
		self.min_length = opt.min_length
		self.minimal_relative_prob = opt.minimal_relative_prob
		self.out_file = out_file
		self.tgt_eos_id = fields["tgt"].vocab.stoi[Constants.EOS_WORD]
		self.tgt_bos_id = fields["tgt"].vocab.stoi[Constants.BOS_WORD]
		self.reward=opt.reward
		self.model_saver=model_saver
		self.grad_accum_count = opt.accum_count
		self.model.train()
		self.n_gpu=opt.world_size
		self.optim=optim

	# self.src_eos_id = fields["src"].vocab.stoi[Constants.EOS_WORD]


	
	def reinforce(self,train_iter_fct,rein_steps):
		# data = build_dataset(self.fields,
		# 					 src_data_iter=src_data_iter,
		# 					 tgt_data_iter=tgt_data_iter,
		# 					 use_filter_pred=False)
		#
		# if self.cuda:
		# 	cur_device = "cuda"
		# else:
		# 	cur_device = "cpu"
		# # sort=True sort_within_batch=True shuffle=True
		# data_iter = OrderedIterator(
		# 	dataset=data, device=cur_device,
		# 	batch_size=batch_size, train=False, sort=False,
		# 	sort_within_batch=False, shuffle=False)
		# start_time = time.time()
		# print("Begin decoding ...")
		# batch_count = 0
		# all_translation = []

		logger.info('Start training...')
		step = 0
		true_batchs = []
		accum = 0
		normalization = 0
		train_iter = train_iter_fct()
		flag=True

		# optim = torch.optim.Adam(self.model.parameters(), lr=0.00001)
		# epoch=0
		# scheduler = ExponentialLR(optim,0.9)

		while step <= rein_steps:
			# print("learning_rate"+str(optim.learning_rate))
			for i, batch in enumerate(train_iter):  # maybe batch_size is 113
				# for batch in data_iter:
				#sample ids is changed

				true_batchs.append(batch)

				normalization += batch.batch_size
				accum += 1
				if accum == self.grad_accum_count:
					self._maybe_save(step)
					self._gradient_accumulation_rein(true_batchs, normalization)
					true_batchs = []
					normalization = 0
					accum=0
					step+=1
					if step > rein_steps:
						break
					
				
			


				# self.model.zero_grad()
				# greedy_pre,sample_ids,probs,tgt = self.reinforce_batch(batch)
				# sample_ids = sample_ids.t().data.tolist()#batch_size*10
				# tgt = tgt.t().data.tolist()#batch_size*10
				# probs = probs.t()#batch_size*10
				# batch_size = probs.size(0)
				# rewards = []
				# # get the acc of every sample
				# for y, y_hat in zip(sample_ids, tgt):  # y,y_hat  len=110
				# 	rewards.append(self.get_acc(y, y_hat))
				# # expand the reward score to every label
				# rewards = torch.Tensor(rewards).unsqueeze(1).expand_as(probs)
				# rewards = Variable(rewards).cuda()

				# #greedy_pre
				# greedy_pre = greedy_pre.t().data.tolist()
				# baselines = []
				# for y, y_hat in zip(greedy_pre, tgt):
				# 	baselines.append(self.get_acc(y, y_hat))
				# baselines = torch.Tensor(baselines).unsqueeze(1).expand_as(probs)
				# baselines = Variable(baselines).cuda()
				# rewards = rewards - baselines

				# loss = -(probs * rewards).sum() / batch_size
				# # elif self.config.reward == 'hamming_loss':
				# # loss = (probs * rewards).sum() / batch_size

				# loss.backward()

				# #update parms
				# self.optim.step()

				# lo=loss.item()
				# print('loss:' + str(lo))

				# batch_transtaltion = []
				# batch_count += 1
				# print("batch: " + str(batch_count) + "...")
				# self._maybe_save(step)
				
				# if step > rein_steps:
				# 	break
			# print('Decoding took %.1f minutes ...' % (float(time.time() - start_time) / 60.))
			train_iter = train_iter_fct()
			# epoch+=1
			# scheduler.step(epoch)
	
	def _gradient_accumulation_rein(self,true_batchs, normalization):
		if self.grad_accum_count > 1:
			self.model.zero_grad()

		for batch in true_batchs:
			target_size = batch.tgt.size(0)

			if self.grad_accum_count == 1:
				self.model.zero_grad()

			greedy_pre,sample_ids,probs,tgt = self.reinforce_batch(batch)
			
			sample_ids = sample_ids.t().data.tolist()#batch_size*10
			tgt = tgt.t().data.tolist()#batch_size*10
			probs = probs.t()#batch_size*10
			batch_size = probs.size(0)
			rewards = []
			# get the acc of every sample
			for y, y_hat in zip(sample_ids, tgt):  # y,y_hat  len=110
				rewards.append(self.get_acc(y, y_hat))
			# expand the reward score to every label
			rewards = torch.Tensor(rewards).unsqueeze(1).expand_as(probs)
			rewards = Variable(rewards).cuda()

			#greedy_pre
			greedy_pre = greedy_pre.t().data.tolist()
			baselines = []
			for y, y_hat in zip(greedy_pre, tgt):
				baselines.append(self.get_acc(y, y_hat))
			baselines = torch.Tensor(baselines).unsqueeze(1).expand_as(probs)
			baselines = Variable(baselines).cuda()
			rewards = rewards - baselines

			loss = -(probs * rewards).sum() / float(normalization)
			# elif self.config.reward == 'hamming_loss':
			# loss = (probs * rewards).sum() / batch_size

			loss.backward()

			lo=loss.item()
			print('loss:' + str(lo))



				# 4. Update the parameters and statistics.
			if self.grad_accum_count == 1:
				# Multi GPU gradient gather
				if self.n_gpu > 1:
					grads = [p.grad.data for p in self.model.parameters()
							if p.requires_grad
							and p.grad is not None]
					all_reduce_and_rescale_tensors(
						grads, float(1))
				self.optim.step()
			
			if self.model.decoder.state is not None:
				self.model.decoder.detach_state()

		# in case of multi step gradient accumulation,
		# update only after accum batches
		if self.grad_accum_count > 1:
			if self.n_gpu > 1:
				grads = [p.grad.data for p in self.model.parameters()
						if p.requires_grad
						and p.grad is not None]
				all_reduce_and_rescale_tensors(
					grads, float(1))
			self.optim.step()


	def get_acc(self, y, y_hat):
		y_true = np.zeros(6)
		y_pre = np.zeros(6)
		for i in y:
			if i == 3:
				break
			else:
				if i > 3:
					y_true[i - 4] = 1
		for i in y_hat:
			if i == 3:
				break
			else:
				if i > 3:
					y_pre[i - 4] = 1
		if self.reward == 'f1':
			r = metrics.f1_score(np.array([y_true]), np.array([y_pre]), average='micro')
		elif self.reward == 'hacc':
			r = 1 - metrics.hamming_loss(np.array([y_true]), np.array([y_pre]))
		elif self.reward == 'linear':
			f1 = metrics.f1_score(np.array([y_true]), np.array([y_pre]), average='micro')
			hacc = 1 - metrics.hamming_loss(np.array([y_true]), np.array([y_pre]))
			r = 0.5 * f1 + 0.5 * hacc
		return r
	def reinforce_batch(self, batch):
		def get_inst_idx_to_tensor_position_map(inst_idx_list):
			''' Indicate the position of an instance in a tensor. '''
			return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

		def reinforce_decode_step( len_dec_seq, inst_idx_to_position_map,dec_seq):
			''' Decode and update beam status, and then return active beam idx '''

			def predict_word(dec_seq, n_active_inst, len_dec_seq):
				"""

				:param dec_seq: 1*150
				:param n_active_inst:30
				:param n_bm: 5
				:param len_dec_seq:
				:return:
				"""
				# dec_seq: (1, batch_size * beam_size)
				dec_output, *_ = self.model.decoder(dec_seq, step=len_dec_seq)
				# dec_output: (1, batch_size * beam_size, hid_size)
				word_prob = self.model.generator(dec_output.squeeze(0))
				# word_prob: (batch_size * beam_size, vocab_size)
				# word_prob = word_prob.view(n_active_inst, -1)
				# word_prob: (batch_size, beam_size, vocab_size)

				return word_prob

			n_active_inst = len(inst_idx_to_position_map)  # 30

			# dec_seq = prepare_beam_dec_seq(inst_dec_beams)
			# dec_seq: (1, batch_size )
			# in here ,we predict the word
			#word_prob batch_size*10
			word_prob = predict_word(dec_seq, n_active_inst, len_dec_seq)

			# Update the beam with predicted word prob information and collect incomplete instances
			# active_inst_idx_list, select_indices = collect_active_inst_idx_list(
			# 	inst_dec_beams, word_prob, inst_idx_to_position_map)

			# if select_indices is not None:
			# 	assert len(active_inst_idx_list) > 0
			# 	self.model.decoder.map_state(
			# 		lambda state, dim: state.index_select(dim, select_indices))

			return word_prob



		# with torch.no_grad():
		# -- Encode
		# src_seq:(batch_size,seq_len,dim)
		src_seq = make_features(batch, 'src')
		tgt=make_features(batch,'tgt')
		src_seq = src_seq.transpose(0, 1).contiguous()
		# src: (seq_len_src, batch_size)
		src_emb, src_enc, _ = self.model.encoder(src_seq)
		# src_emb: (seq_len_src, batch_size, emb_size)
		# src_end: (seq_len_src, batch_size, hid_size)
		self.model.decoder.init_state(src_seq, src_enc)
		src_len = src_seq.size(0)

		# -- Repeat data for beam search
		# n_bm = self.beam_size
		batch_size = src_seq.size(1)
		# change the length of the src and src_enc ,five times batch_size (150)
		# self.model.decoder.map_state(lambda state, dim: tile(state, n_bm, dim=dim))
		# src_enc: (seq_len_src, batch_size * beam_size, hid_size)

		# -- Prepare beams
		decode_length = self.decode_length
		# decode_min_length = 0
		# if self.decode_min_length >= 0:
			# decode_min_length = src_len - self.decode_min_length
		# inst_dec_beams = [Beam(n_bm, decode_length=decode_length, minimal_length=decode_min_length,
		# 					   minimal_relative_prob=self.minimal_relative_prob, bos_id=self.tgt_bos_id,
		# 					   eos_id=self.tgt_eos_id, device=self.device) for _ in range(n_inst)]

		# -- Bookkeeping for active or not
		active_inst_idx_list = list(range(batch_size))  # [0,......batch_size]
		# change into {0:0,...idx:idx}
		inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

		dec_seq_greedy=Variable(torch.LongTensor(1,batch_size).fill_(self.tgt_bos_id)).cuda()
		dec_seq_mul=Variable(torch.LongTensor(1,batch_size).fill_(self.tgt_bos_id)).cuda()
		# -- Decode
		# all instances have finished their path to (<EOS>) no need <EOS>
		#first step use the greed method  baseline
		outputs,sample_ids=[],[]
		for len_dec_seq in range(0, decode_length):
			output_prob= reinforce_decode_step( len_dec_seq, inst_idx_to_position_map,dec_seq_greedy)
			id=output_prob.max(1)[1]
			sample_ids+=[id]
			outputs+=[output_prob]
			dec_seq_greedy=id.unsqueeze(0)
		#second we use mutinol
		sample_ids = torch.stack(sample_ids).squeeze()
		outputs_mul,probs_mul=[],[]
		for len_dec_seq in range(0, decode_length):
			output_prob= reinforce_decode_step( len_dec_seq, inst_idx_to_position_map,dec_seq_mul)
			predicted = F.softmax(output_prob,1).multinomial(1)
			one_hot = Variable(torch.zeros(output_prob.size())).cuda()
			one_hot.scatter_(1, predicted.long(), 1)
			prob = torch.masked_select(F.log_softmax(output_prob,1), one_hot.type(torch.ByteTensor).cuda())
			probs_mul+=[prob]
			outputs_mul+=[predicted]
			dec_seq_mul=predicted.transpose(0,1)

		probs_mul = torch.stack(probs_mul).squeeze()
		outputs_mul = torch.stack(outputs_mul).squeeze()  # [max_tgt_len, batch]

		return sample_ids,outputs_mul,probs_mul,tgt


	def _maybe_save(self, step):
		"""
		Save the model if a model saver is set
		"""
		if self.model_saver is not None:
			self.model_saver.maybe_save(step)
