#!/usr/bin/env python
#!/usr/bin/env PYTHONIOENCODING="utf-8" python
# https://karpathy.github.io/neuralnets/
import tensorflow as tf

sess = tf.InteractiveSession()

# x=  -2, y = 3;
x = tf.Variable(-2., name="x")
y = tf.Variable( 3., name="y")
z = tf.Variable( 4., name="z")

def f(x, y):
  return x * y

def g(a,z):
  return a+z

def deepGate(x, y,z):
  return g(f(x,y),z)

# computes the partial derivative of the tensor loss with respect to the tensor embed.
x_gradient, y_gradient,z_gradient = tf.gradients(deepGate(x,y,z), [x,y,z]) # y,x; # by our complex mathematical derivation above

step_size  = .01 # tf.Variable( .01, name="step_size")

# update_x=tf.assign(x, x + x_gradient * step_size)
# update_y=tf.assign(y, y + y_gradient * step_size)
# update_z=tf.assign(z, z + z_gradient * step_size)
sess.run(tf.initialize_all_variables())

out_old=sess.run(deepGate(x, y,z))
print(out_old)

# // numerical gradient check
h = 0.0001
x_derivative = (deepGate(x+h,y,z) - deepGate(x,y,z)) / h; # -4
y_derivative = (deepGate(x,y+h,z) - deepGate(x,y,z)) / h; # -4
z_derivative = (deepGate(x,y,z+h) - deepGate(x,y,z)) / h; #  3
for a,b in zip([x_derivative,y_derivative,z_derivative],[x_gradient,y_gradient,z_gradient]):
  va= a.eval()
  vb= b.eval()
  check= abs(va-vb)<.01 # sess.run(tf.equal(a,b))
  assert check==True
  check= abs(a-b)<.01
  assert check.eval() # tensor!!

# sess.run(tf.assign(x, x + tf.mul(x_gradient, step_size)))
# sess.run(tf.assign(y, y + y_gradient * step_size))
x = x + x_gradient * step_size
y = y + y_gradient * step_size
z = z + z_gradient * step_size


out_new = sess.run(deepGate(x, y,z))  # -1.86 Higher output than 2! Nice.
print(out_new)
