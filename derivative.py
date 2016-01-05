#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
# https://karpathy.github.io/neuralnets/

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.constant(1.0)
y = x * 2.0
z = y + y + y + y + y + y + y + y + y + y
grads = tf.gradients(z,[x, y])
assert 20.0 == grads[0].eval()
assert 10.0 == grads[1].eval()

# x=  -2, y = 3;
x = tf.Variable(-2., name="x")
y = tf.Variable( 3., name="y")

def derivative(f,x,h):
  return (f(x+h,y)-f(x-h,y)) / (2*h) # 'Tensor' object is not callable

# x = tf.placeholder(-2)
# y = tf.Variable( 3)

def forwardMultiplyGate(x, y):
  return x * y


f = forwardMultiplyGate#(x, y)  # before: -6
# computes the partial derivative of the tensor loss with respect to the tensor embed.
x_gradient, y_gradient = tf.gradients(f(x,y), [x,y]) # y,x; # by our complex mathematical derivation above
h = 1e-7
# x_gradient = (f(x+h,y)-f(x-h,y)) / (2*h) # by hand
# y_gradient = derivative(forwardMultiplyGate,y,1e-7)
step_size  = .01 # tf.Variable( .01, name="step_size")

update_x=tf.assign(x, x + x_gradient * step_size)
update_y=tf.assign(y, y + y_gradient * step_size)
sess.run(tf.initialize_all_variables())
# sess.run(tf.assign(x, x + tf.mul(x_gradient, step_size)))
# sess.run(tf.assign(y, y + y_gradient * step_size))
for i in range(100):
  x = x + x_gradient * step_size #
  y = y + y_gradient * step_size
  out_new = sess.run(forwardMultiplyGate(x, y))  # -5.87. Higher output! Nice.
  # _,_,out_new = sess.run([update_x,update_y, forwardMultiplyGate(x, y)])  # -5.87. Higher output! Nice.
  print(out_new)
