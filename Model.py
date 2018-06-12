### IMPORT ###
import numpy as np
import os
from tqdm import tqdm
import Preprocessing
import tensorflow as tf
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

### PARAMETERS ###
BATCH_SIZE = 20
WINDOW_SIZE = 20
HIDDEN_SIZE_LSTM = 300

### PREPROCESSING ###
### THE TRAINING HAS BEEN DONE ON SEMCOR BUT, SINCE THE DATA REQUIRED TO TRAIN ON SEMCOR TAKES TOO MUCH
### TO BE LOADED ON GITLAB, I SHOW HERE THE CODE AS IF I TRAINED ON SENSEVAL2

# Build label dictionary (id word : label)
with open('./data/senseval2.txt') as f:
    file_raw_labels = f.read()
lab_dict = {label[:label.find('bn:') - 1] : label[label.find('bn:'):] for label in file_raw_labels.split('\n')}

## Build Dataset semcor (for training)
# From binaries: 
files = ['./bin/senseval2_X', './bin/senseval2_var', './bin/senseval2_loss_w', './bin/senseval2_y']
data = Preprocessing.Dataset(files, is_bin=True)

# From xml (VERY SLOW). I show this code (line 29 to 47) just for clarity, since it requires files which take
# days to load on git (sensembed vectors, babelnet_reduced)

# Build BabelNetReduced babelnet from xml (file created with the Java 
# API whose creation code I omit)
#babelnet = Preprocessing.BabelNetReduced('/data/babelnet_reduced.xml')
#babelnet_xml_tree = babelnet.get_babelnet_xml_tree()
#babelnet_lemmas_set = babelnet.get_babelnet_lemmas_set()
#babelnet_synsets_set = babelnet.get_babelnet_synsets_set()

# Build Sensembed sensembed from xml
#sensembed = Preprocessing.Sensembed(os.path.join('sensembed_vectors','babelfy_vectors'),
#								babelnet_lemmas_set, babelnet_synsets_set)
#d_lemmas = sensembed.get_d_lemmas()
#d_synsets = sensembed.get_d_synsets()
#sensembed_lemmas_set = sensembed.get_lemmas_set()
#sensembed_synsets_set = sensembed.get_synsets_set()

#semcor = Preprocessing.Dataset('/data/semcor.data.xml', is_bin=False, babelnet, sensembed, '/data/semcor.gold.key.bnids.txt')


### MODEL DEFINITION ###
# run on CPU
# comment this part if you want to run it on GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

emb_size = 412 # Size of Sensembed + pos in one hot encoding
label_size = 400 # Size of Sensembed

with tf.name_scope('inputs'):
    train_input = tf.placeholder(tf.float32, shape=[None, None, emb_size])
    sl = tf.placeholder(tf.int32, shape = [None,])
    train_label = tf.placeholder(tf.float32, shape=[None, None, label_size])
    loss_weights = tf.placeholder(tf.int32, shape = [None, None, 1])

with tf.name_scope('lstm_layer'):
    cell_fw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE_LSTM)
    cell_bw = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE_LSTM)
    (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, train_input,
                                                            sequence_length=sl, dtype=tf.float32)
    lstm_output = tf.concat([output_fw, output_bw], axis=-1)

with tf.name_scope('output_layer'):
    output_matrix = tf.Variable(tf.random_normal([HIDDEN_SIZE_LSTM*2, label_size])) #SERVE MASKING??
    output = tf.tensordot(lstm_output, output_matrix, axes=[[2], [0]])

with tf.name_scope('loss'):
    train_label_norm = tf.nn.l2_normalize(train_label, axis=2)
    output_norm = tf.nn.l2_normalize(output, axis=2)
    loss = tf.losses.cosine_distance(train_label_norm, output_norm, axis=2, weights=loss_weights)
        
with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
saver = tf.train.Saver()

init = tf.global_variables_initializer()

### MODEL TRAINING ###
num_steps = 10
with tf.Session() as session:
    init.run()
    sum_loss = 0
    sum_acc = 0
    loss_list = []
    acc_list = []
    stepz = []
    for step in tqdm(range(num_steps), total=num_steps, unit="batch"):
        x,y,var,l_w,l = Preprocessing.generate_batch(data, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE)
        out, lo, _ = session.run(
            [output, loss, optimizer],
            feed_dict={train_input: x, sl : l, train_label : y, loss_weights : l_w}
        )
        
        sum_loss += lo
        right = 0
        allz = 0
        for i_s, sentence in enumerate(out):
            for i_w, prediction in enumerate(sentence):
                if type(var[i_s][i_w]) != float and len(var[i_s][i_w].keys()) != 1:
                    keys = [k for k in list(var[i_s][i_w].keys()) if k != 'id']
                    candidates_sy = [np.array(list(map(lambda x : float(x), k.split(' ')))) 
                                     for k in keys]
                    min_dist_index = np.argmin([cosine(s,prediction) for s in candidates_sy])
                    min_dist_sy = keys[min_dist_index]
                    pred_synset = var[i_s][i_w][min_dist_sy]
                    ground_truth = lab_dict[var[i_s][i_w]['id']]
                    if pred_synset == ground_truth:
                        right += 1
                    allz += 1
        acc = right / allz
        sum_acc += acc
        
        print_step = 2
        write_step = 2
        if step % print_step == 0:
            print('avg_loss:', sum_loss/print_step)
            print('avg_acc:', sum_acc/print_step)
            sum_acc = 0
            sum_loss = 0
        if step % write_step == 0:
            stepz.append(step)
            loss_list.append(lo)
            acc_list.append(acc)
    
    save_path = saver.save(session, "./tmp/final_model.ckpt")
    print("Model saved in path: %s" % save_path)
	
### MODEL EVALUATIION ###
datasets = ["senseval2","senseval3","semeval2007","semeval2013","semeval2015"]
f1_list = []
emb_perc = []
for dataset in datasets:
    with open('./data/' + dataset + '.txt') as f:
        file_raw_labels = f.read()
    lab_dict = {label[:label.find('bn:') - 1] : label[label.find('bn:'):] for label in file_raw_labels.split('\n')}
    files = ['./bin/' + dataset + '_X', './bin/' + dataset + '_var', './bin/' + dataset + '_loss_w', './bin/' + dataset + '_y']
    banana = Preprocessing.Dataset(files, is_bin=True)
    
    with tf.Session() as session: 
        saver.restore(session, "./checkpoint/final_model.ckpt")
        x,y,var,l_w,l = Preprocessing.generate_batch(banana, WINDOW_SIZE, None)
        out = session.run(
            output,
            feed_dict={train_input: x, sl : l, loss_weights : l_w}
        )

        right = 0
        allz_prec = 0
        allz_rec = 0
        for i_s, sentence in enumerate(out):
            for i_w, prediction in enumerate(sentence):
                if type(var[i_s][i_w]) != float and len(var[i_s][i_w].keys()) != 1:
                    keys = [k for k in list(var[i_s][i_w].keys()) if k != 'id']
                    candidates_sy = [np.array(list(map(lambda x : float(x), k.split(' ')))) 
                                     for k in keys]
                    min_dist_index = np.argmin([cosine(s,prediction) for s in candidates_sy])
                    min_dist_sy = keys[min_dist_index]
                    pred_synset = var[i_s][i_w][min_dist_sy]
                    ground_truth = lab_dict[var[i_s][i_w]['id']]
                    if pred_synset == ground_truth:
                        right += 1
                    if l_w[i_s][i_w]:
                        allz_prec += 1
                    allz_rec += 1
        prec = right / allz_prec
        rec = right / allz_rec
        f1 = 2 / (1/rec + 1/prec)
        f1_list.append(f1)
        emb_perc.append(allz_prec / allz_rec)
        print('f1:', f1 )
        print('embedded words percentage:', allz_prec / allz_rec)
		

### PLOTTING ###
n_groups = 5
baseline = [0.656, 0.66, 0.545, 0.638, 0.671]
good_scores = [x + 0.04 for x in baseline]
 
fig, ax = plt.subplots(figsize=(10,6))
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8
 
rects1 = plt.bar(index, f1, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Model')
 
rects2 = plt.bar(index + bar_width, baseline, bar_width,
                 alpha=opacity,
                 color='y',
                 label='Baseline')

rects2 = plt.bar(index + 2*bar_width, good_scores, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Baseline + 4')

 
plt.xlabel('Datasets')
plt.ylabel('F1 score')
plt.title('Evaluation graph')
plt.xticks(index + bar_width, datasets)
plt.legend()
 
plt.tight_layout()
plt.show()


