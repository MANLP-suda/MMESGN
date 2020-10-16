import data_loader
import codecs
import emo_preprocess
X_train, X_valid, X_test, y_train, y_valid, y_test = data_loader.read_cmumosei_emotion_pkl()
out_file = codecs.open('test_data/test.src', 'w+', 'utf-8')
for sample in y_test:
	tran=' '.join(sample)
	out_file.write(tran + '\n')
out_file.close()