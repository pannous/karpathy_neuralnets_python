#!/usr/bin/env python
import math


#every Unit corresponds to a wire in the diagrams
class Unit:
  def __init__(self,value, grad) :
    #value computed in the forward pass
    self.value = value;
    #the derivative of circuit output w.r.t this unit, computed in backward pass
    self.grad = grad;


class multiplyGate:
  # __init__(self):
  def forward(self,u0, u1) :
    #store pointers to input Units u0 and u1 and output unit utop
    self.u0 = u0;
    self.u1 = u1;
    self.utop = Unit(u0.value * u1.value, 0.0);
    return self.utop;

  def backward(self) :
    #take the gradient in output unit and chain it with the
    #local gradients, which we derived for multiply gate before
    #then write those gradients to those Units.
    self.u0.grad += self.u1.value * self.utop.grad;
    self.u1.grad += self.u0.value * self.utop.grad;
  

class addGate:
# addGate(self):
  def forward(self,u0, u1) :
    self.u0 = u0;
    self.u1 = u1; #store pointers to input units
    self.utop = Unit(u0.value + u1.value, 0.0);
    return self.utop;

  def backward(self) :
    #add gate. derivative wrt both inputs is 1
    self.u0.grad += 1 * self.utop.grad;
    self.u1.grad += 1 * self.utop.grad;
  


def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def sig(x):
  return 1 / (1 + math.exp(-x))

class sigmoidGate:
  def forward(self,u0) :
    self.u0 = u0;
    self.utop = Unit(sig(self.u0.value), 0.0);
    return self.utop;

  def backward(self) :
    s = sig(self.u0.value);
    self.u0.grad += (s * (1 - s)) * self.utop.grad;
  



#create input units
a = Unit(1.0, 0.0);
b = Unit(2.0, 0.0);
c = Unit(-3.0, 0.0);
x = Unit(-1.0, 0.0);
y = Unit(3.0, 0.0);

#create the gates
mulg0 = multiplyGate();
mulg1 = multiplyGate();
addg0 = addGate();
addg1 = addGate();
sg0 = sigmoidGate();

#do the forward pass
def forwardNeuron() :
  ax = mulg0.forward(a, x); #a*x = -1
  by = mulg1.forward(b, y); #b*y = 6
  axpby = addg0.forward(ax, by); #a*x + b*y = 5
  axpbypc = addg1.forward(axpby, c); #a*x + b*y + c = 2
  return sg0.forward(axpbypc); #sig(a*x + b*y + c) = 0.8808

s=forwardNeuron();

print('circuit output: ' + str(s.value)); #prints 0.8808



# A circuit: it takes 5 Units (x,y,a,b,c) and outputs a single Unit
# It can also compute the gradient w.r.t. its inputs
class Circuit() :
  # create some gates
  def __init__(self):
    self.mulg0 = multiplyGate()
    self.mulg1 = multiplyGate()
    self.addg0 = addGate()
    self.addg1 = addGate()

  def forward(self, x,y,a,b,c) :
    self.ax = self.mulg0.forward(a, x) # a*x
    self.by = self.mulg1.forward(b, y) # b*y
    self.axpby = self.addg0.forward(self.ax, self.by) # a*x + b*y
    self.axpbypc = self.addg1.forward(self.axpby, c) # a*x + b*y + c
    return self.axpbypc

  def backward(self, gradient_top) : # takes pull from above
    self.axpbypc.grad = gradient_top
    self.addg1.backward() # sets gradient in axpby and c
    self.addg0.backward() # sets gradient in ax and by
    self.mulg1.backward() # sets gradient in b and y
    self.mulg0.backward() # sets gradient in a and x
  

#
# That's a circuit that simply computes a*x + b*y + c and can also compute the gradient. It uses the gates code we developed in Chapter 1. Now lets write the SVM, which doesn't care about the actual circuit. It is only concerned with the values that come out of it, and it pulls on the circuit.

# SVM class
class SVM() :
  def __init__(self):
  # random initial parameter values
    self.a = Unit(1.0, 0.0)
    self.b = Unit(-2.0, 0.0)
    self.c = Unit(-1.0, 0.0)
    self.step_size = 0.01

    self.circuit = Circuit()


  def forward(self, x, y) : # assume x and y are Units
    self.unit_out = self.circuit.forward(x, y, self.a, self.b, self.c)
    return self.unit_out

  def backward(self, label) : # label is +1 or -1

    # reset pulls on a,b,c
    self.a.grad = 0.0
    self.b.grad = 0.0
    self.c.grad = 0.0

    # compute the pull based on what the circuit output was
    pull = 0.0
    if(label == 1 and self.unit_out.value < 1) :
      pull = 1 # the score was too low: pull up
    
    if(label == -1 and self.unit_out.value > -1) :
      pull = -1 # the score was too high for a positive example, pull down

    if self.unit_out.value < -2:
      pull = 1

    if self.unit_out.value > 2:
      pull = -1

    self.circuit.backward(pull) # writes gradient into x,y,a,b,c

    # add regularization pull for parameters: towards zero and proportional to value
    self.a.grad += -self.a.value
    self.b.grad += -self.b.value

  def learnFrom(self, x, y, label) :
    self.forward(x, y) # forward pass (set .value in all Units)
    self.backward(label) # backward pass (set .grad in all Units)
    self.parameterUpdate() # parameters respond to tug

  def parameterUpdate(self) :
    step_size=self.step_size
    self.a.value += step_size * self.a.grad
    self.b.value += step_size * self.b.grad
    self.c.value += step_size * self.c.grad
    self.step_size = step_size*.9999


# Now lets train the SVM with Stochastic Gradient Descent:

data = []
labels = []
data.append([1.2, 0.7]);labels.append(1)
data.append([-0.3, -0.5]); labels.append(-1)
data.append([3.0, 0.1]) ;labels.append(1)
data.append([-0.1, -1.0]); labels.append(-1)
data.append([-1.0, 1.1]) ;labels.append(-1)
data.append([2.1, -3]); labels.append(1)

svm = SVM()

# a function that computes the classification accuracy
def evalTrainingAccuracy() :
  num_correct = 0
  for i in range(0,len(data)):
    x = Unit(data[i][0], 0.0)
    y = Unit(data[i][1], 0.0)
    true_label = labels[i]
    # see if the prediction matches the provided label
    predicted_label = 1 if svm.forward(x, y).value > 0 else -1
    if(predicted_label == true_label) :
      num_correct=num_correct+1
  return num_correct*1. / len(data)

from random import randrange as rand
# the learning loop
for j in range(0,40000):
  # pick a random data point
  i = rand(len(data))
  x = Unit(data[i][0], 0.0)
  y = Unit(data[i][1], 0.0)
  label = labels[i]
  svm.learnFrom(x, y, label)

  if(j % 25 == 0) : # every 10 iterations...
    accuracy = evalTrainingAccuracy()
    print('training accuracy at iter ' + str(j) + ': ' + str(accuracy))
    if(accuracy==1.0):exit()
  

