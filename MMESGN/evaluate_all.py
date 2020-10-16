import codecs
from sklearn import metrics
import numpy as np
import os

def read_file_to_list(path,vocab_dict):
	src_file = codecs.open(path, 'r','utf-8')
	path_list = []
	for line in src_file:
		temp = [0, 0, 0, 0, 0, 0]
		line = line.strip().split(' ')
		for key in line:
			if key:
				k = vocab_dict[key]
				temp[k] = 1
		path_list.append(temp)
	src_file.close()
	return path_list
def dense(y):
	label_y = []
	for i in range(len(y)):
		for j in range(len(y[i])):
			label_y.append(y[i][j])

	return label_y
def get_accuracy(y, y_pre):
	print('metric_acc:  ' + str(round(metrics.accuracy_score(y, y_pre),4)))
	sambles = len(y)
	count = 0.0
	for i in range(sambles):
		y_true = 0
		all_y = 0
		for j in range(len(y[i])):
			if y[i][j] > 0 and y_pre[i][j] > 0:
				y_true += 1
			if y[i][j] > 0 or y_pre[i][j] > 0:
				all_y += 1
		if all_y <= 0:
			all_y = 1

		count += float(y_true) / float(all_y)
	acc = float(count) / float(sambles)
	acc=round(acc,4)
	print('accuracy_hand:' + str(acc))


def get_metrics(y, y_pre):
	"""

	:param y:1071*6
	:param y_pre: 1071*6
	:return:
	"""

	test_labels = dense(y)
	test_pred = dense(y_pre)
	print(metrics.classification_report(test_labels, test_pred, digits=4))
	# print(metrics.classification_report(test_labels, test_pred, digits=4))
	# print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
	# print("Micro average Test Precision, Recall and F1-Score...")
	# print(metrics.precision_recall_fscore_support(test_labels,test_pred, average='micro'))
	y=np.array(y)
	y_pre=np.array(y_pre)
	hamming_loss = metrics.hamming_loss(y, y_pre)
	print("hammloss: "+str(round(hamming_loss,4)))
	macro_f1 = metrics.f1_score(y, y_pre, average='macro')
	macro_precision = metrics.precision_score(y, y_pre, average='macro')
	macro_recall = metrics.recall_score(y, y_pre, average='macro')
	get_accuracy(y, y_pre)
	y = np.array(y)
	y_pre = np.array(y_pre)

	print(metrics.classification_report(y, y_pre, digits=4))
	# print("micro_precision, micro_precison,micro_recall")
	micro_f1 = metrics.f1_score(y, y_pre, average='micro')
	micro_precision = metrics.precision_score(y, y_pre, average='micro')
	micro_recall = metrics.recall_score(y, y_pre, average='micro')
	# print(""+str(round(micro_precision,4))+"\t"+str(round(micro_recall,4))+"\t"+str(round(micro_f1,4)))
	return hamming_loss, macro_f1, macro_precision, macro_recall, micro_f1, micro_precision, micro_recall

vocab_dict={'happiness':0,'sadness':1,'anger':2,'fear':4,'disgust':3,'surprise':5}

src_file='test_data/test.src'
src_list=read_file_to_list(src_file,vocab_dict)

tgt_path_prevx='test_data/test.tran'
best_step=0
best_f1=0
for i in range(10, 5000, 10):
	current_path = tgt_path_prevx + '_' + str(i)
	if os.path.exists(current_path):
		tgt_list=read_file_to_list(current_path,vocab_dict)
		if len(tgt_list)!=len(src_list):
			continue
		if i ==1400 or i == 1700:
			get_metrics(src_list,tgt_list)
		src_list=np.array(src_list)
		tgt_list=np.array(tgt_list)
		micro_f1 = metrics.f1_score(src_list, tgt_list, average='micro')
		print(' '+str(i)+'   micro_f1: '+str(round(micro_f1,4))+'\n')
		# get_accuracy(src_list,tgt_list)
		if micro_f1>best_f1:
			best_f1=micro_f1
			best_step=i
	else:
		continue

print('best_step:'+str(best_step)+'\n')
print('best_f1:'+str(round(best_f1,4))+'\n')


