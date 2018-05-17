
# coding: utf-8

# In[1]:


from keras import layers
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import copy
import os
import sys
import time
from six.moves import range


# # Parameters Config

# In[2]:


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


# In[3]:


DATA_SIZE = 100000
DIGITS = int(sys.argv[1])
REVERSE = False
MAXLEN = DIGITS + 1 + DIGITS
chars = '0123456789+ '
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1
ITERATIONS=10
TRAINING_SIZE = int(DATA_SIZE * 0.8)
TESTING_SIZE = DATA_SIZE - TRAINING_SIZE
HISTORY = []


# In[4]:


class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)


# In[5]:


ctable = CharacterTable(chars)


# In[6]:


ctable.indices_char


# # Data Generation

# In[7]:


questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < DATA_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))


# In[8]:


print(questions[:5], expected[:5])


# # Processing

# In[9]:


print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(expected), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)


# In[10]:


indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# train_test_split
train_x = x[:TRAINING_SIZE]
train_y = y[:TRAINING_SIZE]
test_x = x[TRAINING_SIZE:]
test_y = y[TRAINING_SIZE:]

split_at = len(train_x) - len(train_x) // 10
(x_train, x_val) = train_x[:split_at], train_x[split_at:]
(y_train, y_val) = train_y[:split_at], train_y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

print('Testing Data:')
print(test_x.shape)
print(test_y.shape)


# In[11]:


print("input: ", x_train[:3], '\n\n', "label: ", y_train[:3])


# # Build Model

# In[12]:


print('Build model...')
model = Sequential()
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
model.add(layers.RepeatVector(DIGITS + 1))
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

model.add(layers.TimeDistributed(layers.Dense(len(chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


# In[13]:


if DIGITS == 4:
    transfer_name = "{:d}d_adder.h5".format(DIGITS-1)
else:
    transfer_name = "{:d}d_adder_transfer.h5".format(DIGITS-1)

transfer = load_model(os.path.join("model",transfer_name))
#print(transfer.layers[0].get_weights())
def layer_shape(model):
    for y in range(len(model.layers)):
        print("Layer:"+str(y))
        for x in range(len(model.layers[y].get_weights())):
            print(model.layers[y].get_weights()[x].shape)

#Transfer
for n in range(len(model.layers)):
    weights = copy.deepcopy(transfer.layers[n].get_weights())
    model.layers[n].set_weights(weights)


# # Training

# In[14]:


val_acc = 0
iteration = 1
start = time.time()
while val_acc < 0.9:
    print()
    print('-' * 50)
    print('Iteration', iteration)
    results = model.fit(x_train, y_train,
                              batch_size=BATCH_SIZE,
                              epochs=1,
                              validation_data=(x_val, y_val))
    
    HISTORY.append([str(results.history[i][0]) for i in sorted(results.history)])
    val_acc = results.history["val_acc"][0]
    iteration += 1
end = time.time() - start


# In[15]:


model_name = "{:d}d_adder_transfer.h5".format(DIGITS)
model.save(os.path.join("model",model_name))
log_name = "{:d}d_adder_transfer.csv".format(DIGITS)
with open(os.path.join("log",log_name), 'w') as wf:
    wf.write('acc,loss,val_acc,val_loss\n')
    for line in HISTORY:
        wf.write(",".join(line)+"\n")
    wf.write("Time,{}\n".format(str(end)))


# # Validation

# In[16]:


right = 0
preds = model.predict_classes(test_x, verbose=0)
for i in range(len(preds)):
    q = ctable.decode(test_x[i])
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
    print('Q', q[::-1] if REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print('True', end=' ')
        right += 1
    else:
        print('False', end=' ')
    print(guess)
print("MSG : Accuracy is {}".format(right / len(preds)))

