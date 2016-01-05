#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

from tensorflow.python.ops.math_ops import sigmoid
import tensorflow as tf

sess = tf.InteractiveSession()

def sigma(x):
  return sigmoid(x) # 1./(1 + exp(-x))
  #σ(x)=1/1+e^−x

x0=tf.Variable(0.)
x=tf.placeholder(tf.float32)
ds=tf.gradients(sigma(x),x)[0]
ds0=tf.gradients(sigma(x0),x0)[0]
sess.run(tf.initialize_all_variables())
assert tf.equal(ds0,sigma(x0)-sigma(x0)*sigma(x0)).eval()


class Unit(object):
  def __init__(this,value, grad):
    # value computed in the forward pass
    this.value = value
    #the derivative of circuit output w.r.t this unit, computed in backward pass
    this.grad = grad

  def __str__(self): # for debuggin
    return "Unit: value: %f grad: %f"%(self.value,self.grad)
  # def __repr__(self):
  #   return str(self.value)


class multiplyGate:
  def forward(this,u0,u1):
    this.u0 = u0
    this.u1 = u1
    this.utop = Unit(u0.value * u1.value, 0.0)
    return this.utop

  def backward(this):
      this.u0.grad += this.u1.value * this.utop.grad
      this.u1.grad += this.u0.value * this.utop.grad


class addGate:
  def forward(this,u0,u1):
    this.u0 = u0
    this.u1 = u1
    this.utop = Unit(u0.value + u1.value, 0.0)
    return this.utop

  def backward(this):
      this.u0.grad += 1 * this.utop.grad
      this.u1.grad += 1 * this.utop.grad

def sig(x):
  return 1./(1 + math.exp(-x))

class sigGate:
  def forward(this,u0):
    this.u0 = u0
    this.utop = Unit(sig(u0.value), 0.0)
    return this.utop

  def backward(this):
      s = sig(this.u0.value) # 'gain? waste!!
      this.u0.grad += s *( 1 - s) * this.utop.grad


# create input units and 'edges'
a = Unit(1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)

# create the gates
mulg0 = multiplyGate()
mulg1 = multiplyGate()
addg0 = addGate()
addg1 = addGate()
sg0 = sigGate()

# do the forward pass
def forwardNeuron():
  ax = mulg0.forward(a, x) # a*x = -1
  by = mulg1.forward(b, y) # b*y = 6
  axpby = addg0.forward(ax, by) # a*x + b*y = 5
  axpbypc = addg1.forward(axpby, c) # a*x + b*y + c = 2
  s = sg0.forward(axpbypc) # sig(a*x + b*y + c) = 0.8808
  return s

s=forwardNeuron()

print('circuit output: %f' % s.value) # prints 0.8808 YAY!

s.grad = 1.0;
sg0.backward(); # writes gradient into axpbypc
addg1.backward(); # writes gradients into axpby and c
addg0.backward(); # writes gradients into ax and by
mulg1.backward(); # writes gradients into b and y
mulg0.backward(); # writes gradients into a and x

step_size = 0.01;
a.value += step_size * a.grad; # a.grad is -0.105
b.value += step_size * b.grad; # b.grad is 0.315
c.value += step_size * c.grad; # c.grad is 0.105
x.value += step_size * x.grad; # x.grad is 0.105
y.value += step_size * y.grad; # y.grad is 0.210

s=forwardNeuron();
print('circuit output after one backprop: %f' % s.value); # prints 0.8825

#  checking the numerical gradient:
def forwardCircuitFast(a,b,c,x,y):
  return 1/(1 + math.exp( - (a*x + b*y + c)));

a_grad1=a.grad
y_grad1=y.grad

a = 1
b = 2
c = -3
x = -1
y = 3;
h = 0.0001;
a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h;
y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h;
x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h;
print("Difference to numerical gradient:")
print(a_grad-a_grad1) #
# YEP!

