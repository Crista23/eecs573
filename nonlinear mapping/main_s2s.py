####
# part of the codes are re-developed based on the tutorial and codes at
# https://github.com/ematvey/tensorflow-seq2seq-tutorials
####
import csv
import numpy as np
import tensorflow as tf
import helpers
import string
import random
import sys
from random import randint
import matplotlib.pyplot as plt
import pprint
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import pprint
from nltk.corpus import stopwords
from gensim.models import word2vec


# load the dictionary and the src and target sequences
def build_dict_src_tgt_from_tsv(src = 'problem', tgt = 'implication', filename = 'newdata.tsv'):
    dict = {}
    i_dict={}
    src_list = []
    tgt_list = []

    stop_char = [chr(c) for c in range(256)]
    stop_char = [x for x in stop_char if not x.isalnum()]
    stop_char.remove(' ')
    stop_char.remove('_')
    stop_char.remove(':')
    stop_char = ''.join(stop_char)

    stop_wds = set(stopwords.words('english'))
    # add_stop_wds = {u'will',u'can',u'could',u'have',u'has',u'been', u'a', u'b', u'c', u'd',
    #                 u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    #                 u'z', u'may', u'might', u'details', u'none', u'copyright', u'must', u'due',u'intel', u'bit', u'bits', u'error', u'erratum',
    #                 u'possible', u'set', u'workaround', u'errors', u'possible', u'data', u'code'}
    #
    # stop_wds.update(add_stop_wds)


    with open(filename) as tsvfile:
        st = 2
        content = csv.reader(tsvfile, delimiter='\t')
        for line in content:
            if line[0].startswith(src):
                tokens = filter(None, line[1].lower().translate(None, stop_char).split(' '))
                src_str = []
                for i in tokens:
                    try:
                        if not unicode(i) in stop_wds:
                            src_str.append(i)
                    except:
                        continue

                for each in src_str:
                    if dict.has_key(each):
                        continue
                    else:
                        dict[each] = st
                        i_dict[st] = each
                        st+=1
                src_list.append([dict[x] for x in src_str])

            if line[0].startswith(tgt):
                tokens = filter(None, line[1].lower().translate(None, stop_char).split(' '))
                tgt_str = []
                for i in tokens:
                    try:
                        if not unicode(i) in stop_wds:
                            tgt_str.append(i)
                    except:
                        continue

                for each in tgt_str:
                    if dict.has_key(each):
                        continue
                    else:
                        dict[each] = st
                        i_dict[st] = each
                        st+=1
                tgt_list.append([dict[x] for x in tgt_str])

    dict['<stop>'] = 1
    dict['<pad>'] = 0
    i_dict[1] = '<stop>'
    i_dict[0] = '<pad>'

    return (dict, i_dict, src_list, tgt_list)


def build_dict_src_tgt_from_txt():
    dict = {}
    i_dict={}
    src_list = []
    tgt_list = []

    stop_char = [chr(c) for c in range(256)]
    stop_char = [x for x in stop_char if not x.isalnum()]
    stop_char.remove(' ')
    stop_char.remove('_')
    stop_char.remove(':')
    stop_char = ''.join(stop_char)

    stop_wds = set(stopwords.words('english'))
    # add_stop_wds = {u'will',u'can',u'could',u'have',u'has',u'been', u'a', u'b', u'c', u'd',
    #                 u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    #                 u'z', u'may', u'might', u'details', u'none', u'copyright', u'must', u'due',u'intel', u'bit', u'bits', u'error', u'erratum',
    #                 u'possible', u'set', u'workaround', u'errors', u'possible', u'data', u'code'}

    # stop_wds.update(add_stop_wds)

    st = 2
    tr_wf_t = open('train_problem.txt')
    lines = tr_wf_t.readlines()
    for eachline in lines:
        tokens = filter(None, eachline.lower().translate(None, stop_char).split(' '))
        src_str = []
        for i in tokens:
            try:
                if not unicode(i) in stop_wds:
                    src_str.append(i)
            except:
                continue

        for each in src_str:
            if dict.has_key(each):
                continue
            else:
                dict[each] = st
                i_dict[st] = each
                st += 1
        src_list.append([dict[x] for x in src_str])
    tr_wf_t.close()

    tr_wf_i = open('train_implication.txt')
    lines = tr_wf_i.readlines()
    for eachline in lines:
        tokens = filter(None, eachline.lower().translate(None, stop_char).split(' '))
        tgt_str = []
        for i in tokens:
            try:
                if not unicode(i) in stop_wds:
                    tgt_str.append(i)
            except:
                continue

        for each in tgt_str:
            if dict.has_key(each):
                continue
            else:
                dict[each] = st
                i_dict[st] = each
                st += 1
        tgt_list.append([dict[x] for x in tgt_str])
    tr_wf_i.close()

    dict['<stop>'] = 1
    dict['<pad>'] = 0
    i_dict[1] = '<stop>'
    i_dict[0] = '<pad>'

    return (dict, i_dict, src_list, tgt_list)



def load_test_from_txt(dict):
    test_src_list = []
    test_tgt_list = []

    stop_char = [chr(c) for c in range(256)]
    stop_char = [x for x in stop_char if not x.isalnum()]
    stop_char.remove(' ')
    stop_char.remove('_')
    stop_char.remove(':')
    stop_char = ''.join(stop_char)

    stop_wds = set(stopwords.words('english'))
    #add_stop_wds = {u'will',u'can',u'could',u'have',u'has',u'been', u'a', u'b', u'c', u'd',
    #                u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm', u'n', u'o', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    #                u'z', u'may', u'might', u'details', u'none', u'copyright', u'must', u'due',u'intel', u'bit', u'bits', u'error', u'erratum',
    #                u'possible', u'set', u'workaround', u'errors', u'possible', u'data', u'code'}

    # stop_wds.update(add_stop_wds)

    te_wf_t = open('test_problem.txt')
    lines = te_wf_t.readlines()
    for eachline in lines:
        tokens = filter(None, eachline.lower().translate(None, stop_char).split(' '))
        src_str = []
        for i in tokens:
            try:
                if not unicode(i) in stop_wds:
                    src_str.append(i)
            except:
                continue
        tmp = []
        for x in src_str:
            if dict.has_key(x):
                tmp.append(dict[x])
        test_src_list.append(tmp)
    te_wf_t.close()

    te_wf_i = open('test_implication.txt')
    lines = te_wf_i.readlines()
    for eachline in lines:
        tokens = filter(None, eachline.lower().translate(None, stop_char).split(' '))
        tgt_str = []
        for i in tokens:
            try:
                if not unicode(i) in stop_wds:
                    tgt_str.append(i)
            except:
                continue
        tmp = []
        for x in tgt_str:
            if dict.has_key(x):
                tmp.append(dict[x])
        test_tgt_list.append(tmp)
    te_wf_i.close()

    return (test_src_list, test_tgt_list)

# functions for the neural network structures
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_targets_length)  # all False at the initial step
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input

    elements_finished = (time >= decoder_targets_length)  # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


# start of the program

# load data
#(dict, i_dict, src_list, tgt_list) = build_dict_src_tgt_from_tsv()
(dict, i_dict, src_list, tgt_list) = build_dict_src_tgt_from_txt()
(test_src_list, test_tgt_list) = load_test_from_txt(dict=dict)

# alignment
#len_src_list = [len(each) for each in src_list]
#max_len_src = max(len_src_list)

#len_src_list = [len(each) for each in src_list]
#max_len_src = max(len_src_list)

tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = len(dict)
input_embedding_size = 50

encoder_hidden_units = 50
decoder_hidden_units = encoder_hidden_units * 2

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

# Embedding
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)


# Encoder
encoder_cell = LSTMCell(encoder_hidden_units)


((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )


encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

# decoder
decoder_cell = LSTMCell(decoder_hidden_units)
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)



assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)


decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

decoder_prediction = tf.argmax(decoder_logits, 2)

# Optimizer

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

batch_size = 100

batches = helpers.generate_word_seq(batch_size=batch_size, src_list = src_list, tgt_list = tgt_list)
def next_feed():
    (src_batch, tgt_batch) = next(batches)
    # pprint.pprint(batch)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(src_batch)
    decoder_targets_, decoder_targets_lengths_ = helpers.batch(
        [sequence + [EOS] + [PAD] * 2 for sequence in tgt_batch]
    )
    # pprint.pprint(decoder_targets_lengths_)

    # sys.exit()
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_targets_length: decoder_targets_lengths_,
    }

def next_test_feed():
    (src_batch, tgt_batch) = (test_src_list, test_tgt_list)
    # pprint.pprint(batch)
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(src_batch)
    decoder_targets_, decoder_targets_lengths_ = helpers.batch(
        [sequence + [EOS] + [PAD] * 2 for sequence in tgt_batch]
    )
    # pprint.pprint(decoder_targets_lengths_)

    # sys.exit()
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
        decoder_targets_length: decoder_targets_lengths_,
    }


loss_track = []


max_batches = 5000
batches_in_epoch = 1000



try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, gdt,pred) in enumerate(zip(fd[encoder_inputs].T,fd[decoder_targets].T, predict_.T)):
                print('Training sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('  groundtruth > {}'.format(gdt))
                print('   gdt string  > {}'.format([i_dict[x] for x in gdt]))
                print('    predicted > {}'.format(pred))
                print('   pred string > {}'.format([i_dict[x] for x in pred]))
                if i >= 0:
                    break
            print()

            print('Test data'.format(batch))
            tfd = next_test_feed()
            predict_ = sess.run(decoder_prediction, tfd)
            for i, (inp, gdt, pred) in enumerate(zip(tfd[encoder_inputs].T, tfd[decoder_targets].T, predict_.T)):
                print('  Testing sample {}'.format(i + 1))
                print('   input       > {}'.format(inp))
                print('   groundtruth > {}'.format(gdt))
                print('   gdt string  > {}'.format([i_dict[x] for x in gdt]))
                print('   predicted   > {}'.format(pred))
                print('   pred string > {}'.format([i_dict[x] for x in pred]))
                if i >= 4:
                    break
            print()
            print()
except KeyboardInterrupt:
    print('training interrupted')


# matplotlib inline
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))











