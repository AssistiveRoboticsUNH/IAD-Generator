import tensorflow as tf

input = tf.Variable([3.0], dtype=tf.float32)

@tf.custom_gradient
def clip_grad_layer(x):
  def grad(dy):
    return dy * 2.0
  return tf.identity(x), grad

output_clip = clip_grad_layer(input)
grad_clip = tf.gradients(output_clip, input)

# output without gradient clipping in the backwards pass for comparison:
output = tf.identity(input)
grad = tf.gradients(output, input)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print("with clipping:", sess.run(grad_clip)[0])
  print("without clipping:", sess.run(grad)[0])