### IMPORT ###
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import pickle

from util import embed
from util import embed_synset
from util import embed_word
from util import mutable_pos


### DATASET CLASS ###
# Container for Datasets

## Fields:
# 1) X: 3D List of shape (num sentences in dataset, lenght of corresponding sentence, size of word embedding). X[i][j][k] contains the
# k-th component of the embedding of word j in sentence j.
# 2) var: 2D List of dictionaries of shape (num sentences in dataset, lenght of corresponding sentence). var[i][j] is a dictionary d whose 
# keys are the sense embedding which can be predicted for word j in sentence i and d[s] = babelnet id corresponding to s. I know,
# could have done it better.
# 3) loss_w is a 2D List of booleans. If loss_w[i][j] is true then in the training phase the loss will be computed on word j in sentence i,
# otherwise no.
# 4) y: 3D List of same shape as X. y[i][j][k] contains the k-th component of the embedding of label corresponding to word j in sentence j.
# 5) is_train: boolean. If it's false then the dataset y field is empty, otherwise no.
# 6) length: int. Number of sentences in the dataset.

## Methods: 
# 1) __init__: can build from binaries or from xml files
# 2) from_xml: constructor subroutine. Takes as input an xml file, is_bin, a BabelNet object, a Sensembed object and eventually a txt file
# for labels.
# 3) from_bin: subroutines of the constructor. Takes as input a list of binary files.
# 4) save: saves X,var, loss_w and y in 4 binariy files
class Dataset:
	X = []
	var = []
	loss_w = []
	y = []
	is_train = False
	length = 0
	
	def __init__(self, files, is_bin, babelnet = None, sensembed = None, labels_txt_file = None):
		if is_bin:
			self.from_bin(files, is_bin)
		else:
			self.from_xml(files, is_bin, babelnet, sensembed, labels_txt_file)
			
	def from_xml(self, file, is_bin, babelnet = None, sensembed = None, labels_txt_file = None):
		# Takes variables from babelnet
		babelnet_xml_tree = babelnet.get_babelnet_xml_tree()
		
		# Opens xml file
		with open(file, 'r', encoding='utf-8', errors='ignore') as f:
			xml_tree = ET.parse(f)
		xml_root = xml_tree.getroot()
		
		n_sent = len(xml_root.findall('.//sentence'))
		
		# Opens labels file
		if labels_txt_file != None:
			self.is_train = True
			with open(labels_txt_file) as f:
				file_raw_labels = f.read()
			lab_dict = {label[:label.find('bn:') - 1] : label[label.find('bn:'):] for label in file_raw_labels.split('\n')}
			del file_raw_labels
		
		# Heart of the code: for every sentence in the document
		for sentence in tqdm(xml_root.findall('./text/sentence'), total=n_sent, unit="sent"):
			X_sentence = []
			y_sentence = []
			var_sentence = []
			loss_sentence = []
			# For every word in the sentence
			for word in sentence.findall('./*'):
				# Embed the word
				is_instance = word.tag == "instance"
				loss_flag = False if word.tag == "wf" else True
				pos = ('[@pos="' + word.get("pos") + '"]') if word.get("pos") in mutable_pos else ''
				query = './*[@lemma="' + word.get("lemma") + '"]/meaning' + pos
				try:
					synsets = babelnet_xml_tree.findall(query)
				except:
					print('Error')
					continue
				word_embedding, loss_flag = embed_word(word, synsets[:3], loss_flag, sensembed)
				X_sentence.append(word_embedding)
				
				# Embed the synsets if the word is instance
				d = {'id' : word.get('id') if word.tag == "instance" else None}
				if is_instance:
					for synset in synsets:
						is_label = self.is_train and synset.get("id") == lab_dict[word.get("id")]
						synset_embedding, loss_flag = embed_synset(synset, loss_flag, is_label, sensembed)
						synset_embedding_str = ' '.join([str(s) for s in synset_embedding])
						if synset_embedding_str not in d:
							d[synset_embedding_str] = synset.get("id")
						if is_label:
							y_sentence.append(synset_embedding)
				# If it's not instance embed unk
				elif self.is_train:
					y_sentence.append(embed('unk', sensembed))
				loss_sentence.append(loss_flag)
				var_sentence.append(d)
			self.X.append(X_sentence)
			self.var.append(var_sentence)
			self.loss_w.append(loss_sentence)
			self.y.append(y_sentence)
		self.length = len(self.X)	
		
	def from_bin(self, files, is_bin):
		self.is_train = True if len(files) == 4 else False
		with open(files[0], 'rb') as f:
			self.X = pickle.load(f)
		with open(files[1], 'rb') as f:
			self.var = pickle.load(f)
		with open(files[2], 'rb') as f:
			self.loss_w = pickle.load(f)
		if self.is_train:
			with open(files[3], 'rb') as f:
				self.y = pickle.load(f)
		self.length = len(self.X)
				
			
	def save(self, bin_files):
		variables = [self.X] + [self.var] + [self.loss_w] + [self.y]*self.is_train
		for file, variable in zip(bin_files, variables):
			with open(file, 'wb') as f:
				 pickle.dump(variable, f)
		
	
	def get_X(self):
		return self.X
	
	def get_y(self):
		return self.y
	
	def get_loss_w(self):
		return self.loss_w
	
	def get_is_train(self):
		return self.is_train
	
	def get_var(self):
		return self.var
		
	def get_length(self):
		return self.length
		
### SENSEMBED CLASS ###
# Container for Sensembed embeddings

## Fields:
# 1) d_lemmas: dictionary whose keys are words and d_lemmas[word] = sensembed(word)
# 2) d_synsets: dictionary whose keys are BabelNet synsets and d_synsets[synset] = sensembed(synset)

## Methods: 
# 1) __init__: takes as input the full Sensembed file, the set of useful lemmas, the set of useful synsets and builds, just for them,
# the two dictionaries.
class Sensembed:
	d_lemmas = {}
	d_synsets = {}
	
	def __init__(self, file, lemmas_set, babelnet_synsets_set):
		with open(file, encoding='utf-8') as f:
			for line in f:
				# If it's reading a synset line, add synset to d_synsets
				if 'bn:' in line:
					bn = line.find('bn:')
					if line[bn : line.find(' ')] in babelnet_synsets_set:
						key, _, value = line.partition(' ')
						self.d_synsets[key[bn : ]] = [float(x) for x in value.split(' ')[:-1]]
				# If it's reading a word line, add word to d_lemmas
				elif line[:line.find(' ')] in lemmas_set or line[:line.find(' ')] == 'unk':
					key, _, value = line.partition(' ')
					self.d_lemmas[key] = [float(x) for x in value.split(' ')[:-1]]
					
	def get_d_lemmas(self):
		return self.d_lemmas
		
	def get_d_synsets(self):
		return self.d_synsets
	
	def get_lemmas_set(self):
		return set(self.d_lemmas.keys())
	
	def get_synsets_set(self):
		return set(self.d_synsets.keys())
		
### BABELNET_REDUCED CLASS ###
# Container for BabelNet data		

## Fields: 
#1) bab_xml_tree: 4 levels tree. First level: words, Second level: synsets, Third level: hypernyms of father and words 
# members of the same synset as father, Fourth level: words which are lemmatization of father
# 2) babelnet_synsets_set: self explicative
# 3) babelnet_lemmas_set: self explicative

## Methods:
# __init__: builds fields from xml file. The xml file has been build with the Java API. I omitted the Java code. 
class BabelNetReduced:
	def __init__(self, bab_xml_file):
		with open(bab_xml_file, 'r', encoding='utf-8', errors='ignore') as f:
			self.babelnet_xml_tree = ET.parse(f)
		bab_root = self.babelnet_xml_tree.getroot()
		self.babelnet_synsets_set = set(sy.get('id') for sy in bab_root.findall('.//meaning'))
		self.babelnet_lemmas_set = set(
			sense.text for sense in bab_root.findall('.//sense')) | set(
			word.get('lemma') for word in bab_root.findall('.//word'))
			
	def get_babelnet_xml_tree(self):
		return self.babelnet_xml_tree
		
	def get_babelnet_synsets_set(self):
		return self.babelnet_synsets_set
		
	def get_babelnet_lemmas_set(self):
		return self.babelnet_lemmas_set
		

	
### GENERATE BATCH FUNCTION ###
# Takes in input a Dataset object, two parameters window_size and batch_size, outputs a slice of size batch_size
# of the dataset in a nice format for the neural network to input.

def generate_batch(dataset, window_size, batch_size=None):
	l_dataset = dataset.get_length()
	X = dataset.get_X()
	y = dataset.get_y()
	var = dataset.get_var()
	loss = dataset.get_loss_w()
	
	# Select batch_size random sentences
	if batch_size != None:
		id_sent = np.random.randint(0, l_dataset + 1, size=batch_size)
		X = [sent for i, sent in enumerate(X) if i in id_sent]
		y = [lab for i, lab in enumerate(y) if i in id_sent]
		var = [v for i, v in enumerate(var) if i in id_sent]
		loss = [l for i, l in enumerate(loss) if i in id_sent]
	X_new = []
	y_new = []
	var_new = []
	loss_new = []
	
	# splits sentences longer than window_size 
	for i in range(len(X)):
		if len(X[i]) > window_size:
			for j in range(len(X[i]) // 20 + 1):
				a = 20*j
				b = min(len(X[i]), 20*(j+1))
				X_new.append(X[i][a : b])
				y_new.append(y[i][a : b])
				var_new.append(var[i][a : b])
				loss_new.append(loss[i][a : b])
		else:
			X_new.append(X[i])
			y_new.append(y[i])
			var_new.append(var[i])
			loss_new.append(loss[i])
	if batch_size != None:
		X = X_new[:batch_size]
		y = y_new[:batch_size]
		var = var_new[:batch_size]
		loss = loss_new[:batch_size]
	
	# Pads with zeroes the sentences
	seq_len = [len(sent) for sent in X]
	l_max_sent = max(seq_len)
	X = [sent + [[float(0)]*412]*(l_max_sent - len(sent)) for sent in X]
	y = [lab + [[float(0)]*400]*(l_max_sent - len(lab)) for i, lab in enumerate(y)]
	var = [v + [float(0)]*(l_max_sent - len(v)) for i, v in enumerate(var)]
	loss_w = [l + [False]*(l_max_sent - len(l)) for i, l in enumerate(loss)]
	
	return np.array(X), np.array(y), var, np.expand_dims(np.array(loss_w), axis=2), np.array(seq_len)