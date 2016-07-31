##
## RNN LSTM model implementation using tensorflow
##
from __future__ import print_function

import sys
import os
import pdb
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
import unittest
import argparse
import time



class Params:
    """
    parameters of the model
    """
    rnn_size = 50
    num_layers = 1
    batch_size = 10
    seq_length = 10
    num_epochs = 200
    grad_clip = 5.0
    learning_rate = 0.02
    decay_rate = 0.97
    input_dict = "dataset/cmudict_clean.dict"
    model = 'lstm'
    savedir = "./savedir"
    savefreq = 1000



class Model():
    """
    Definition of the RNN model
    """
    def __init__(self, args):
        self.args = args

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size)
        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        ##
        ## input data will be of dimension
        ## shape = (batch_size, seq_length, invocab_size)
        ##
        self.input_data = tf.placeholder(tf.float32, 
                                         [args.batch_size, 
                                          args.seq_length, 
                                          args.char_size])

        ##
        ## target data will be of dimension
        ## shape = (batch_size, seq_length)
        ## NOTE : out dim not specified here
        ##
        self.targets = tf.placeholder(tf.int32, 
                                      [args.batch_size, 
                                       args.seq_length])

        ##
        ## initial state is of size batch_size * state_size
        ## this is equivalent to tf.zeros([batch_size, state_size])
        ##
        self.initial_state = cell.zero_state(args.batch_size, 
                                             tf.float32)

        ##
        ## input and final softmax layer outputs
        ## here we specify the out dimention
        ##
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", 
                                        [args.rnn_size, 
                                         args.phvocab_size])
            softmax_b = tf.get_variable("softmax_b", 
                                        [args.phvocab_size])

            ##
            ## unrolling of the input to sequence length
            ## and removing the 1 dim
            ##
            inputs = tf.split(1, args.seq_length, self.input_data)
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        ##
        ## simple rnn decoder. Simple meaning without attention
        ## last_state is the final state from rnn after specified
        ## sequence length. 
        ## last_state is the thought vector
        ##
        outputs, last_state = seq2seq.rnn_decoder(inputs, 
                                                  self.initial_state, 
                                                  cell, 
                                                  scope='rnnlm')

        ##
        ## outputs is a list of size sequence length.
        ## Each list element is of dimention batch_size * rnn_size
        ## i.e for each unrolled input, there will be one output state
        ## (last state) each will be of dimension rnn_size.
        ##
        outconcat = tf.concat(1, outputs)
        output = tf.reshape(outconcat, [-1, args.rnn_size])

        ##
        ## final logit layer
        ## NOTE : x * W (where x is batch * rnn_size)
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        ##
        ## cost function
        ## 
        reshaped_target = tf.reshape(self.targets, [-1]),
        seq_weight = tf.ones([args.batch_size * args.seq_length])
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [reshaped_target], [seq_weight], args.phvocab_size)

        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state

        ##
        ## Optimizer
        ## Adam optimizer and gradient clipping
        ##
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, 
                                                       tvars),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(grads, tvars))



class Dictionary:
    """
    Process the dictionary
    """
    def __init__(self, indict, stress=False):
        """
        """
        if not os.path.isfile(indict):
            print("Error : File not found", indict)
            raise IOError

        self.fdin = open(indict)


    def next(self):
        """
        """
        for line in self.fdin:
            sline = line.strip().upper().split()

            word = sline[0]

            if "(" in word:
                word = word[word.index("(")]

            phone = []

            if not stress:
                for p in sline[1:]:
                    if p[0].isdigit():
                        phone.append(p[:-1])
                    else:
                        phone.append(p)
            else:
                phone = sline[1:]

            return (word, phone)

        raise StopIteration


    def __iter__(self):
        return self



class DataLoader(Exception):
    """
    Prepare and load the data for training
    """
    def __init__(self, indict, batch_size=1, delay=4, data_limit=200):
        """
        input dictionary
        """
        self.delay = delay
        self.batch_size = batch_size

        if not os.path.isfile(indict):
            print("Error : File not found", indict)
            raise IOError

        self.dict_data = {}

        char_vocab = set()
        ph_vocab = set()

        self.max_vocab_len = 0

        with open(indict) as fdin:
            for line in fdin:
                sline = line.strip().upper().split()

                if len(sline) == 0:
                    continue

                if sline[0] == "#":
                    continue

                word = sline[0]
                ph = sline[1:]

                wlen = len(word)
                plen = len(ph)
                
                if plen > wlen:
                    wlen = plen

                if wlen > self.max_vocab_len:
                    self.max_vocab_len = wlen

                char_vocab = char_vocab.union(set(list(word)))
                ph_vocab = ph_vocab.union(set(ph))

                if word in self.dict_data:
                    self.dict_data[word].append(ph)
                else:
                    self.dict_data[word] = [ph]

        ##
        ## set the sequence length based on max word length and delay
        ##
        self.seq_length = self.max_vocab_len + self.delay

        ##
        ## add null to both char and phone vocab
        ## and create char to num mapping
        ## phone to num mapping and vice-versa
        ##
        self.null = '_UNK'
        self.char_vocab = list(char_vocab)
        self.char_vocab.append(self.null)
        self.char_vocab.sort()

        self.ph_vocab = list(ph_vocab)
        self.ph_vocab.append(self.null)
        self.ph_vocab.sort()

        self.char_vocab_dict = {w : i for i, w in enumerate(self.char_vocab)}
        self.ph_vocab_dict = {w : i for i, w in enumerate(self.ph_vocab)}

        self.char_vocab_invdict = {i : w for i, w in enumerate(self.char_vocab)}
        self.ph_vocab_invdict = {i : w for i, w in enumerate(self.ph_vocab)}

        ##
        ## generate the one hot vector representation for the training data
        ##
        self.input_data = []
        self.target_data = []

        count = 0

        for word, phones in self.dict_data.iteritems():
            word_onehot = self.get_onehot_vector(word)

            for ph in phones:
                phlen = len(ph)
                diff = abs(phlen - self.max_vocab_len)

                newph = [self.null] * self.delay
                newph.extend(ph)

                if diff:
                    newph.extend([self.null] * diff)

                phseq = map(self.ph_vocab_dict.get, newph)

                self.input_data.append(word_onehot)
                self.target_data.append(np.reshape(phseq, ((1, self.seq_length))))

            if count >= data_limit:
                break
        ##
        ## prepare the batches
        ##
        if len(self.input_data) != len(self.target_data):
            print("Error : input and target length doesn't match")
            sys.exit(1)


        self.num_batches = int(len(self.input_data) / self.batch_size)
        last_index = self.num_batches * self.batch_size

        ##
        ## fix the batches limit
        ##
        self.input_batch_data = self.input_data[:last_index]
        self.target_batch_data = self.target_data[:last_index]
        ## self.batch_data = np.split(self.train_data[:last_index], self.num_batches, 0)


    def get_onehot_vector(self, word):
        """
        for a given word get its one hot vector representation
        """
        wordlen = len(word)
        diff = abs(wordlen - self.seq_length)
        newword = list(word)
        newword.extend([self.null] * diff)
        
        word_seq = map(self.char_vocab_dict.get, newword)
        onehot = tf.one_hot(word_seq, len(self.char_vocab_dict))
        
        return onehot


    def next_batch(self):
        """
        return next batch
        """
        if self.end_index > len(self.target_batch_data):
            raise StopIteration

        xdata = self.input_batch_data[self.start_index:self.end_index]
        ydata = self.target_batch_data[self.start_index:self.end_index]

        self.start_index += self.end_index
        self.end_index += self.batch_size

        return xdata, ydata


    def reset_batch_pointer(self):
        """
        """
        self.start_index = 0
        self.end_index = self.batch_size


    def __iter__(self):
        """
        return an iterator over the batch of data
        """
        self.index = 0
        return self


    def __del__(self):
        """
        release memory/close files
        """
        try:
            self.fdin.close()
        except:
            pass



def train(args):
    """
    """
    dataloader = DataLoader(args.input_dict, args.batch_size)

    args.seq_length = dataloader.seq_length
    args.char_size = len(dataloader.char_vocab_dict)
    args.phvocab_size = len(dataloader.ph_vocab_dict)
    
    try:
        os.makedirs(args.savedir)
    except:
        pass

    model = Model(args)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        ##
        ## Add ops to save and restore all the variables.
        ##
        saver = tf.train.Saver()

        ##
        ## initial state for the model
        ##
        state = sess.run(model.initial_state)

        for epoch in range(args.num_epochs):
            ##
            ## change the learning rate based on previous epoch
            ##
            sess.run(tf.assign(model.lr, 
                               args.learning_rate * (args.decay_rate ** epoch)))

            dataloader.reset_batch_pointer()

            for n in xrange(dataloader.num_batches):
                b = dataloader.next_batch()
                x, y = b

                inx = np.array(map(sess.run, x))
                iny = np.concatenate(y, 0)

                feed = {model.input_data: inx, model.targets: iny, 
                        model.initial_state: state}

                train_loss, state, _,logits = sess.run([model.cost, 
                                                 model.final_state, 
                                                 model.train_op,
                                                 model.logits], 
                                                feed_dict = feed)

        ##
        ## Save the variables to disk.
        ##
        if (epoch % args.savefreq == 0):
            outfile = os.path.join(args.savedir, "model.chpt.%d" % epoch)
            save_path = saver.save(sess, outfile)

            print("Model saved in file: %s" % save_path)


        dataloader.reset_batch_pointer()

        for n in xrange(10):
            b = dataloader.next_batch()
            x, y = b
            
            inx = np.array(map(sess.run, x))

            feed = {model.input_data: inx}
            logits = sess.run(model.logits, feed_dict = feed)
            logits = tf.split(0, args.batch_size, logits)

            for res in logits:
                result = sess.run(tf.arg_max(res, 1))
                print(result, [dataloader.ph_vocab_invdict[i] for i in result])


def infer(args):
    """
    """
    dataloader = DataLoader(args.input_dict)

    args.seq_length = dataloader.seq_length
    args.char_size = len(dataloader.char_vocab_dict)
    args.phvocab_size = len(dataloader.ph_vocab_dict)

    model = Model(args)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        ##
        ## initial state for the model
        ##
        state = sess.run(model.initial_state)

        dataloader.reset_batch_pointer()

        for n in xrange(dataloader.num_batches):
            b = dataloader.next_batch()
            x, y = b
            inx = np.array([sess.run(x), sess.run(x)])
           
            feed = {model.input_data: inx}
            logits = sess.run(model.logits, feed_dict = feed)
            logits = tf.split(0, args.batch_size, logits)
            
            for res in logits:
                result = sess.run(tf.arg_max(res, 1))
                print(result, [dataloader.ph_vocab_invdict[i] for i in result])


if __name__ == '__main__':
    args = Params()
    train(args)




