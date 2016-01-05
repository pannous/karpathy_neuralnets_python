#!/usr/bin/env python
from random import randrange as rand

data = []
labels = []
data.append([1.2, 0.7]);labels.append(1)
data.append([-0.3, -0.5]); labels.append(-1)
data.append([3.0, 0.1]) ;labels.append(1)
data.append([-0.1, -1.0]); labels.append(-1)
data.append([-1.0, 1.1]) ;labels.append(-1)
data.append([2.1, -3]); labels.append(1)

#// initial parameters
a = 1
b = -2
c = -1
num_correct=0
# for j in range(0,400):
for j in range(0,200): # optimal net
  # // pick a random data point
  i = rand(len(data))
  x = data[i][0];
  y = data[i][1];
  label = labels[i];

  # // compute pull
  score = a*x + b*y + c;
  error=abs(score-label)
  error2=error*error

  pull = 0.0;
  # squared hinge loss SVM
  if(label ==  1 and score <  0): pull = error2
  if(label == -1 and score >  0) : pull = -error2
  # if(label ==  1 and score <  1): pull = error2
  # if(label == -1 and score > -1) : pull = -error2
  # if(label ==  1 and score <  .02): pull =  1;
  # if(label == -1 and score > -.02): pull = -1;

  # if score < -1.5 : pull =  1;
  # if score > 1.5 : pull =  -1;
  if(pull==0.0) :
      num_correct=num_correct+1

  # // compute gradient and update parameters
  step_size = 0.005;
  a += step_size * (x * pull + a/1000.); #// -a is from the regularization
  b += step_size * (y * pull + b/1000.); #// -b is from the regularization
  c += step_size * (1 * pull);


  # if(j % 5 == 0) : # every 10 iterations...
  if(j % 25 == 0) : # every 10 iterations...
    accuracy = num_correct*1. / 25 #(j+1)# len(data)
    num_correct=0
    print('training accuracy at iter ' + str(j) + ': ' + str(accuracy))
    # if(accuracy==1.0):exit()
