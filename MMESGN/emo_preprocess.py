# -- coding: utf-8 --
import argparse
import torch


# import data.dict as dict
# from data.dataloader import dataset


def preprocess_data(raw_data):
	raw_data_0 = torch.FloatTensor(raw_data[0])  # 8273*25*300
	raw_data_1 = torch.FloatTensor(raw_data[1])  # 8273*25*35
	raw_data_2 = torch.FloatTensor(raw_data[2])  # 8273*25*74
	raw_data = torch.cat([raw_data_0, raw_data_1, raw_data_2], 2)  # 8273*25*409
	raw_data = raw_data.numpy().tolist()
	return raw_data  # 8273*25*409


def preprocess_emo_2(raw_emo):  # 8273*6
	emo_all = []
	data = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

	for sample in raw_emo:
		temp = []
		if sample[0] > 0:
			temp.append(data[0])
		if sample[1] > 0:
			temp.append(data[1])
		if sample[2] > 0:
			temp.append(data[2])
		if sample[4] > 0:
			temp.append(data[4])
		if sample[3] > 0:
			temp.append(data[3])
		if sample[5] > 0:
			temp.append(data[5])
		emo_all.append(temp)

	return emo_all


def preprocess_emo(y_train_emo, y_test_emo, y_valid_emo):
	dicts = {}
	data1 = ['happiness', 'sadness', 'anger', 'disgust', 'surprise', 'fear']
	dicts['emo'] = makeVocabulary(data1)
	index = dicts['emo'].convertToIdx(['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'], None)

	datas = [y_train_emo, y_test_emo, y_valid_emo]
	das = []
	raw_emos_all = []
	for data in datas:
		da = []
		raw_emos = []
		data = torch.from_numpy(data)
		for len in range(data.size(0)):
			d = []
			raw_emo = []
			d.append(2)
			for emo in range(data.size(1)):
				if data[len][emo] > 0:
					d.append(index[emo].item())
					raw_emo.append(index[emo].item())
			d.append(3)
			da.append(d)
			raw_emos.append(raw_emo)
		das.append(da)
		raw_emos_all.append(raw_emos)
	saveVocabulary('emo', dicts['emo'], 'data/data/save_data' + '.tgt.dict')
	print('Saving data to \'' + 'data/data/save_data ')
	save_data = {'dicts': dicts,
				 'train': das[0],
				 'valid': das[1],
				 'test': das[2]}
	torch.save(save_data, 'data/data/save_data.pt')
	emo_vocab_size = dicts['emo'].size()
	return das[0], das[1], das[2], emo_vocab_size, raw_emos_all[0], raw_emos_all[1], raw_emos_all[2]


def makeVocabulary(data):
	vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD,
					   dict.BOS_WORD, dict.EOS_WORD], lower=False)

	# vocab.setidxToLabel({'Happiness':0,'Sadness':1,'Anger':2,'Disgust':3,'Surprise':4,'Fear':5,'<blank> ':6,'<unk>':7,'<s>':8,'</s>':9})
	# vocab.setlabelToIdx({0:'Happiness',1:'Sadness',2:'Anger',3:'Disgust',4:'Surprise',5:'Fear',6:':<blank> ',7:'<unk>',8:'<s>',9:'</s>'})
	for d in data:
		vocab.add(d)
	return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize, char=False):
	vocab = None
	if vocabFile is not None:
		# If given, load existing word dictionary.
		print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
		vocab = dict.Dict()
		vocab.loadFile(vocabFile)
		print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

	if vocab is None:
		# If a dictionary is still missing, generate it.
		print('Building ' + name + ' vocabulary...')
		genWordVocab = makeVocabulary(dataFile, vocabSize, char=char)
		vocab = genWordVocab

	print()
	return vocab


def saveVocabulary(name, vocab, file):
	print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
	vocab.writeFile(file)


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, sort=False, char=False):
	src, tgt = [], []
	raw_src, raw_tgt = [], []
	sizes = []
	count, ignored = 0, 0

	print('Processing %s & %s ...' % (srcFile, tgtFile))
	srcF = open(srcFile)
	tgtF = open(tgtFile)

	while True:
		sline = srcF.readline()
		tline = tgtF.readline()

		# normal end of file
		if sline == "" and tline == "":
			break

		# source or target does not have same number of lines
		if sline == "" or tline == "":
			print('WARNING: source and target do not have the same number of sentences')
			break

		sline = sline.strip()
		tline = tline.strip()

		# source and/or target are empty
		if sline == "" or tline == "":
			print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
			ignored += 1
			continue

		if opt.lower:
			sline = sline.lower()
			tline = tline.lower()

		srcWords = sline.split()
		tgtWords = tline.split()

		#
		if opt.src_length == 0 or (len(srcWords) <= opt.src_length and len(tgtWords) <= opt.tgt_length):

			if char:
				srcWords = [word + " " for word in srcWords]
				tgtWords = list(" ".join(tgtWords))
			else:
				srcWords = [word + " " for word in srcWords]
				tgtWords = [word + " " for word in tgtWords]

			src += [srcDicts.convertToIdx(srcWords,
										  dict.UNK_WORD)]
			tgt += [tgtDicts.convertToIdx(tgtWords,
										  dict.UNK_WORD,
										  dict.BOS_WORD,
										  dict.EOS_WORD)]
			raw_src += [srcWords]
			raw_tgt += [tgtWords]
			sizes += [len(srcWords)]
		else:
			ignored += 1

		count += 1

		if count % opt.report_every == 0:
			print('... %d sentences prepared' % count)

	srcF.close()
	tgtF.close()

	if opt.shuffle == 1:
		print('... shuffling sentences')
		perm = torch.randperm(len(src))
		src = [src[idx] for idx in perm]
		tgt = [tgt[idx] for idx in perm]
		sizes = [sizes[idx] for idx in perm]
		raw_src = [raw_src[idx] for idx in perm]
		raw_tgt = [raw_tgt[idx] for idx in perm]

	if sort:
		print('... sorting sentences by size')
		_, perm = torch.sort(torch.Tensor(sizes))
		src = [src[idx] for idx in perm]
		tgt = [tgt[idx] for idx in perm]
		raw_src = [raw_src[idx] for idx in perm]
		raw_tgt = [raw_tgt[idx] for idx in perm]

	print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
		  (len(src), ignored, opt.src_length))

	return dataset(src, tgt, raw_src, raw_tgt)

	dicts = {}
	if opt.share:
		assert opt.src_vocab_size == opt.tgt_vocab_size
		print('share the vocabulary between source and target')
		dicts['src'] = initVocabulary('source and target',
									  [opt.train_src, opt.train_tgt],
									  opt.src_vocab,
									  opt.src_vocab_size)
		dicts['tgt'] = dicts['src']
	else:
		dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
									  opt.src_vocab_size)
		dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
									  opt.tgt_vocab_size, char=opt.char)

	print('Preparing training ...')
	train = makeData(opt.train_src, opt.train_tgt, dicts['src'], dicts['tgt'], char=opt.char)

	print('Preparing validation ...')
	valid = makeData(opt.valid_src, opt.valid_tgt, dicts['src'], dicts['tgt'], char=opt.char)

	print('Preparing test ...')
	test = makeData(opt.test_src, opt.test_tgt, dicts['src'], dicts['tgt'], char=opt.char)

	if opt.src_vocab is None:
		saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
	if opt.tgt_vocab is None:
		saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')

	print('Saving data to \'' + opt.save_data + '.train.pt\'...')
	save_data = {'dicts': dicts,
				 'train': train,
				 'valid': valid,
				 'test': test}
	torch.save(save_data, opt.save_data)


