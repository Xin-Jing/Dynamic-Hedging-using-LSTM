import pandas as pd
import numpy as np
import tensorflow as tf
import time
from statistics import variance

# use past price changes to predict

# To reload:
# import importlib
# a = importlib.reload(RNN_for_dynamic_hedge)
# x = a.function()

# Generating data
def import_data(address):
    # for logged data
    # S,F,s,f,basis = a.import_data("/Users/xinjing/Desktop/dynamic hedge/SP500_for_dynamic_hedge_2.xlsx")
    table = pd.read_excel(address)
    F = table['log_futures'].as_matrix()
    S = table['log_spot'].as_matrix()
    s = table['diff_spot'].as_matrix() # the first difference of the spot prices
    s = s[1:]
    f = table['diff_futures'].as_matrix()# the first difference of the futures prices
    f = f[1:]
    basis = S - F
    basis = basis[:len(basis)-1]
    return S,F,s,f,basis


def generating_placeholders(k):
    n_steps = k
    X = tf.placeholder(tf.float32, [None, n_steps, 2])
    return X


def window_size_k(data, k):
    # data are s or f (returns of spots or futures)
    # shift day by day the data with window size k
    # the data of the last day is omitted.
    l = len(data) - 1
    result = np.zeros([l - k + 1,k])
    for i in range(l - k + 1):
        #print(i)
        result[i] = data[i:i+k]
    return result


def get_target(s,f,k):
    # _s = s[k-1:len(s)]
    # _f = f[k-1:len(f)]
    _s = s[k :len(s)]
    _f = f[k :len(f)]
    # the returned result is of shape (len(s) - k , 2)
    return np.column_stack((_s,_f))


def combine_s_f(s,f,k):
    # combine s and f of the same time together, s and f are unprocessed log difference
    # s1 and f1 are data processed by window_size_k
    s1 = window_size_k(s, k)
    f1 = window_size_k(f, k)
    result = np.zeros([len(s1),k,2])
    for i in range(len(s1)):
        #print(i)
        result[i] = np.column_stack((s1[i], f1[i]))
    return result
# take the absolute values of the data
# np.abs(array)


def inference(current_batch, hidden_units):
    # current_batch is a place holder with shape (none, hidden_size, input_dimension)
    basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units= hidden_units, activation=tf.sigmoid)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, current_batch, dtype=tf.float32)  # shape = (batch_size, num_steps,hidden_units)
    tf.Print(states,[states],message = "This is the states :")
    outputs = tf.transpose(outputs, [1, 0, 2])  # Now shape = (num_steps, batch_size ,hidden_units)
    last = tf.gather(outputs, int(outputs.get_shape()[0]) - 1) # shape of last = (batch_size, hidden_units)
    weight = tf.Variable(tf.truncated_normal([hidden_units, 1]))
    bias = tf.Variable(tf.constant(0.1, shape=[1]))
    b = tf.matmul(last, weight) + bias # shape of b = (batch_size, 1)
    return b, weight,bias

def do_eval(sess, batch_size, current_batch, current_targets, input_set, target_set):
    """Runs one evaluation against the full epoch of data.
      Args:
        sess: The session in which the model has been trained.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from input_data.read_data_sets().
    """

    position_changes = np.array([])
    num_of_batches = len(input_set)//batch_size
    num_examples = num_of_batches * batch_size
    for i in range(num_of_batches):
        batch = input_set[i * batch_size: (i + 1) * batch_size]
        targets = target_set[i * batch_size: (i + 1) * batch_size]
        changes = sess.run( loss_matrix, { current_batch : batch , current_targets: targets})
        position_changes = np.concatenate([position_changes, changes])
    var = variance(position_changes)
    print('Num examples :{}, Variance :{}'.format(num_examples, var))
    return var


k = 4 # number of time steps
print('time steps: {}'.format(k))
batch_size = 1
print('batch size: {}'.format(batch_size))
hidden_units = 15
print('hidden units: {}'.format(hidden_units))
S,F,s,f,basis = import_data("/Users/xinjing/Desktop/dynamic hedge/BP.xlsx")
print("BP")


all_inputs = combine_s_f(s,f,k)
all_inputs = np.abs(all_inputs)
training_inputs = all_inputs[:-80]
testing_inputs = all_inputs[len(all_inputs)-80:]


all_targets = get_target(s,f,k)
training_targets = all_targets[:-80]
testing_targets = all_targets[len(all_targets)-80:]


current_batch = generating_placeholders(k) # a placeholder of shape [batch_size, n_steps, 2]
current_targets = tf.placeholder(tf.float32, [None, 2]) # a placeholder of shape [batch_size, 2]


b, weight , bias = inference(current_batch, hidden_units) #  shape of b = batch_size * 1
# b = 2 * tf.sigmoid(b) # now b range from 0 to 2
b_flatten = tf.reshape(b,[-1]) # shape of b_flatten: [batch_size,0]


loss_matrix = current_targets[:,0] - b_flatten * current_targets[:,1] # s - b*f, shape of b_flatten, s, f: [batch_size, 0]
loss = tf.reduce_sum(tf.square(loss_matrix)) # square the loss of each sample and sum them over.
# gradient = tf.gradients(loss, [weight, bias])


optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)


loss_track = []
num_of_batches = int(round(len(training_inputs)/batch_size))
training_vars = []
testing_vars = []
ratios = []


for i in range(3):
    for j in range(num_of_batches):
        #print(j)
        start_time = time.time()

        batch = training_inputs[j*batch_size : (j+1)*batch_size]
        targets = training_targets[j*batch_size : (j+1)*batch_size]
        _, l, ratio, weight_ ,bias_= sess.run([train, loss,b, weight, bias], { current_batch : batch ,current_targets: targets})
        loss_track.append(l)
        ratios.append(ratio)
        duration = time.time() - start_time

        #if (i + 1) % 10 == 0 or (i + 1) == 300:

        if (i + 1) % 1 == 0:
            if j+1 == num_of_batches:
        # if 1 == 1:
        #     if j == int(num_of_batches/3) or j == 2 * int(num_of_batches/3) or j + 1 == num_of_batches:
                print('Epoch: {}, Batch: {}, loss: {} ({})'.format(i, j, l, duration))
                print('ratio: {}'.format(ratio))
                print('weight: {}'.format(weight_))
                print('bias: {}'.format(bias_))

                #save_path = saver.save(sess, "/Users/xinjing/Desktop/dynamic hedge/Model1/tmp/model.ckpt")
                #print("Model saved in file: %s" % save_path)

                print('Training Data Eval:')
                # var_train, changes_train = do_eval(sess, batch_size, current_batch, current_targets, training_inputs, training_targets)
                # training_vars.append(var_train)
                training_vars.append(do_eval(sess, batch_size, current_batch, current_targets, training_inputs, training_targets))

                print('Test Data Eval:')
                # var_test, changes_test = do_eval(sess, batch_size, current_batch, current_targets, testing_inputs,
                #                            testing_targets)
                # training_vars.append(var_test)

                testing_vars.append(do_eval(sess, batch_size, current_batch, current_targets, testing_inputs, testing_targets))

                print('_________')

import matplotlib.pyplot as plt


#plt.plot(loss_track)
#plt.show()
#plt.plot(training_vars)
#plt.show()
plt.plot(testing_vars)
plt.show()
