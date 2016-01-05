#!/usr/bin/env python
from random import random as rand
from random import randrange as randi
import math

data = []
labels = []
data.append([1.2, 0.7]);labels.append(1)
data.append([-0.3, -0.5]); labels.append(-1)
data.append([3.0, 0.1]) ;labels.append(1)
data.append([-0.1, -1.0]); labels.append(-1)
data.append([-1.0, 1.1]) ;labels.append(-1)
data.append([2.1, -3]); labels.append(1)

## initial parameters
a2 = rand()-.5
b2 = rand()-.5
c2 = rand()-.5
a1 = rand()-.5
b1 = rand()-.5
c1 = rand()-.5
a3 = rand()-.5
b3= -rand()-.5
c3 = rand()-.5
a4 = rand()-.5
b4= -rand()-.5
c4 = rand()-.5
d4=2
num_correct=0
for j in range(0,200): # optimal net# random initial parameters
# a1 = Math.random() - 0.5; # a random number between -0.5 and 0.5
# ... similarly initialize all other parameters to randoms
  # pick a random data point
  i = randi(len(data))

  x = data[i][0];
  y = data[i][1];
  label = labels[i];

  # compute forward pass
  n1 = max(0, a1*x + b1*y + c1); # activation of 1st hidden neuron
  n2 = max(0, a2*x + b2*y + c2); # 2nd neuron
  n3 = max(0, a3*x + b3*y + c3); # 3rd neuron
  score = a4*n1 + b4*n2 + c4*n3 + d4; # the score

  error=abs(score-label)
  error2=error*error

  # // compute pull
  pull = 0.0;
  # squared hinge loss SVM
  if(label ==  1 and score <  0): pull = error2
  if(label == -1 and score >  0) : pull = -error2
  # compute the pull on top
  # pull = 0.0;
  # if(label == 1 and score < 1): pull = 1; # we want higher output! Pull up.
  # if(label == -1 and score > -1): pull = -1; # we want lower output! Pull down.

  if(pull==0.0) :
      num_correct=num_correct+1
  if(j % 20 == 0) : # every n iterations...
    accuracy = num_correct*1. / 20 #(j+1)# len(data)
    num_correct=0
    print('training accuracy at iter ' + str(j) + ': ' + str(accuracy))
    # if(accuracy==1.0):exit()

  # now compute backward pass to all parameters of the model

  # backprop through the last "score" neuron
  dscore = pull;
  da4 = n1 * dscore;
  dn1 = a4 * dscore;
  db4 = n2 * dscore;
  dn2 = b4 * dscore;
  dc4 = n3 * dscore;
  dn3 = c4 * dscore;
  dd4 = 1.0 * dscore; # phew

  # backprop the ReLU non-linearities, in place
  # i.e. just set gradients to zero if the neurons did not "fire"
  dn3 = 0 if  n3 == 0 else dn3;
  dn2 = 0 if  n2 == 0 else dn2;
  dn1 = 0 if  n1 == 0 else dn1;

  # backprop to parameters of neuron 1
  da1 = x * dn1;
  db1 = y * dn1;
  dc1 = 1.0 * dn1;

  # backprop to parameters of neuron 2
  da2 = x * dn2;
  db2 = y * dn2;
  dc2 = 1.0 * dn2;

  # backprop to parameters of neuron 3
  da3 = x * dn3;
  db3 = y * dn3;
  dc3 = 1.0 * dn3;

  # phew! End of backprop!
  # note we could have also backpropped into x,y
  # but we do not need these gradients. We only use the gradients
  # on our parameters in the parameter update, and we discard x,y

  # add the pulls from the regularization, tugging all multiplicative
  # parameters (i.e. not the biases) downward, proportional to their value
  # TODO : DOESN'T WORK
  # da1 += -a1; da2 += -a2; da3 += -a3;
  # db1 += -b1; db2 += -b2; db3 += -b3;
  # da4 += -a4; db4 += -b4; dc4 += -c4;

  # finally, do the parameter update
  step_size = 0.01;
  a1 += step_size * da1; 
  b1 += step_size * db1; 
  c1 += step_size * dc1;
  a2 += step_size * da2; 
  b2 += step_size * db2;
  c2 += step_size * dc2;
  a3 += step_size * da3; 
  b3 += step_size * db3; 
  c3 += step_size * dc3;
  a4 += step_size * da4; 
  b4 += step_size * db4; 
  c4 += step_size * dc4; 
  d4 += step_size * dd4;
  # wow this is tedious, please use for loops in prod.
  # we're done!
