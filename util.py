import numpy as np

### USEFUL (self explicative) VARIABLES ###
mutable_pos = set(['ADJ', 'ADV', 'NOUN', 'VERB'])
resolve_non_mutable = {'.':';', 'ADP':'of', 'CONJ':'and', 'DET':'the', 'NUM':'number', 'PRON':'it', 'PRT':'on', 'X':'etc'}
pos_list = ['ADV', 'ADJ', '.', 'PRON', 'NUM', 'NOUN', 'ADP', 'DET', 'VERB', 'CONJ', 'PRT', 'X']
pos_to_one_hot = {pos_list[i] : [0]*i + [1] + [0]*(len(pos_list) - i - 1) for i in range(len(pos_list))}

### SOFTMAX FUNCTON ###
# Takes as input a vector and outputs its softmax numpy array
def softmax(v):
	return np.exp(v) / np.sum(np.exp(v), axis=0)
	
### LIN_COMB FUNCTION ###
# Takes as input a nxm Matrix and computes the linear combination of matrix 
# rows with weights given by the softmax of the vector of
# the rows indices
def lin_comb(M):
	L = []
	weights = np.array(softmax(np.array(range(1, len(M) + 1))[::-1]))
	M = np.array(M)
	for i in range(len(M)):
		L.append(M[i]*weights[i])
	return np.sum(L, axis=0)
	
### EMBED FUNCTION ###
# takes as input a synset or a word as strings, optionally a pos, and embeds them, 
#concatenating the one hot encoding of the pos.
def embed(word, sensembed, pos = None, is_synset = False):
	d_lemmas = sensembed.get_d_lemmas()
	d_synsets = sensembed.get_d_synsets()

	if pos != None:
		embedding = np.concatenate((d_lemmas[word], pos_to_one_hot[pos]))
	else:
		embedding = d_synsets[word] if is_synset else d_lemmas[word]
	return embedding
	

### EMBED_WORD FUNCTION ###
# Embeds a word using the algorithm in Figure 1 of the report	
def embed_word(word, synsets, loss_flag, sensembed):
	sensembed_lemmas_set = sensembed.get_lemmas_set()
	
	if word.get("lemma") in sensembed_lemmas_set:
		return embed(word.get("lemma"), sensembed, word.get("pos")), loss_flag
	elif word.get("pos") in mutable_pos:
		senses_reprs = []
		for synset in synsets:
			synonim_senses = synset.findall('./sense')
			is_a_senses = synset.findall('./meaning/sense')
			try:
				senses_reprs.append(next(
					syn.text for syn in synonim_senses if syn.text in sensembed_lemmas_set))
			except StopIteration:
				try:
					senses_reprs.append(next(
						sense.text for sense in is_a_senses if sense.text in sensembed_lemmas_set))
				except StopIteration:
					continue
		if len(senses_reprs) != 0:
			senses_reprs = [embed(sense, sensembed, word.get("pos")) for sense in senses_reprs]
			return lin_comb(senses_reprs), loss_flag
		else:
			return embed('unk', sensembed, word.get("pos")), False
	else:
		return embed(resolve_non_mutable[word.get("pos")], sensembed, word.get("pos")), loss_flag
			

### EMBED_WORD FUNCTION ###
# Embeds a synset using the algorithm in Figure 2 of the report			
def embed_synset(synset, loss_flag, is_label, sensembed):
	sensembed_synsets_set = sensembed.get_synsets_set()
	
	if synset.get("id") in sensembed_synsets_set:
		return embed(synset.get("id"), sensembed, is_synset=True), loss_flag
	else:
		is_a_synsets = synset.findall("./meaning")
		try:
			is_a_synsets_repr = next(
				sy.get("id") for sy in is_a_synsets if sy.get("id") in sensembed_synsets_set)
		except StopIteration:
			if is_label:
				return embed('unk', sensembed), False
			else:
				return embed('unk', sensembed), loss_flag
		return embed(is_a_synsets_repr, sensembed, is_synset=True), loss_flag
		
