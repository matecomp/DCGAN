# -*- encoding: utf-8 -*-
# running on python3 and tensorflow 1.1.0
import loader
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    #If Matrix does not exits on scope, create a new variable
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    #If bias does not exits on scope, create a new variable
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    #Return tensor with layer result and their weigths and bias
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):

  with tf.variable_scope(name):
    #Depth of filter is equal to input depth and output_dim gives the depth of next layer
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", with_w=False):

  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w',
              [k_h, k_w, output_shape[-1],
              input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    deconv = tf.nn.conv2d_transpose(input_, w,
                    output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

class batch_norm(object):

  def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):

    with tf.variable_scope(name):
      self.epsilon = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):

    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

class Model():

  def __init__(self, data, epochs=60, batch_size=64, learning_rate=5e-4, is_inference=False):

    image_size = 64
    output_size = 64
    c_dim = 3
    z_dim = 100

    self.is_inference = is_inference
    self.is_training = not is_inference
    self.g_list = list()
    self.d_list = list()

    self.learning_rate = learning_rate
    self.epochs = epochs

    self.data = data
    self.N = len(data)
    self.batch_size = min(batch_size,self.N)

    self.x = data[:batch_size]

    self.dcgan_init(image_size=image_size, output_size=output_size, c_dim=c_dim, z_dim=z_dim)

  def inference(self):
    # scale back to [0, 255] range
    return tf.to_int32((self.G*127)+128)

  def loss(self):
    losses = [
      {'loss': self.d_loss, 'vars': self.d_vars},
      {'loss': self.g_loss, 'vars': self.g_vars}
    ]
    return losses

  def dcgan_init(self,image_size,output_size,z_dim,c_dim,gf_dim=64,df_dim=64,gfc_dim=1024,dfc_dim=1024):

    self.image_size = image_size
    self.output_size = output_size

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    self.soft_label_margin = 0.1

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.build_model()

  #Make foward on a batch of images
  def discriminator(self, image, y=None, reuse=False):

    #Get N value
    batch_size = self.batch_size

    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      #0º layer is a conv layer 64 features maps
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))

      #1º layer is a conv layer 128 features maps
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv'), train=self.is_training))

      #2º layer is a conv layer 256 features maps
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv'), train=self.is_training))

      #3º layer is a conv layer 512 features maps
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv'), train=self.is_training))

      #Calculate size for flatten 2D layer (64/16)^2 = 4^2 = 16, 16*64*8 = 8192
      h3_size = ((self.output_size // 16) ** 2) * self.df_dim * 8

      #4º layer is a flatten(512@4x4 to 8192 neurows) Fully connected a neuron that will be D(x) value
      h4 = linear(tf.reshape(h3, [self.batch_size, h3_size]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4

  #Make foward on a batch of z's
  def generator(self, z, y=None):


    with tf.variable_scope("generator") as scope:

      # 64 is the output size
      s = self.output_size

      # Layers sizes 32,16,8,4
      s2, s4, s8, s16 = int(s // 2), int(s // 4), int(s // 8), int(s // 16)

      #First layer is a fullyconnected layer
      self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s16 * s16, 'g_h0_lin', with_w=True)

      #Transform 1D layer in 2D layer because the next layer, this process is called feature maps
      self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0, train=self.is_training))

      #1º deconv(conv_transpose) layer [Samples, 8, 8, 256]
      self.h1, self.h1_w, self.h1_b = deconv2d(h0,
        [self.batch_size, s8, s8, self.gf_dim * 4], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1, train=self.is_training))

      #2º deconv(conv_transpose) layer [Samples, 16, 16, 128]
      h2, self.h2_w, self.h2_b = deconv2d(h1,
        [self.batch_size, s4, s4, self.gf_dim * 2], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2, train=self.is_training))

      #3º deconv(conv_transpose) layer [Samples, 32, 32, 34]
      h3, self.h3_w, self.h3_b = deconv2d(h2,
        [self.batch_size, s2, s2, self.gf_dim * 1], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3, train=self.is_training))

      #4º deconv(conv_transpose) layer [Samples, 64, 64, 3]
      h4, self.h4_w, self.h4_b = deconv2d(h3,
        [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

  def build_model(self):

    if not self.is_inference:
      # inputs G(z), D(images)
      self.images = tf.placeholder(tf.float32, shape=[self.batch_size,self.image_size,self.image_size,self.c_dim], name='images')
      self.z = tf.random_normal(mean=0.0, stddev=1.0, shape=[self.batch_size, self.z_dim], dtype=tf.float32, seed=None, name='z')
      # create generator
      self.G = self.generator(self.z)
      # create an instance of the discriminator (real samples, fake samples)
      self.D, self.D_logits = self.discriminator(self.images, reuse=False)
      self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

      # we are using the cross entropy loss for all these losses
      # note the use of the soft label smoothing here to prevent D from getting overly confident
      # on real samples
      self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                        labels=tf.ones_like(self.D) - self.soft_label_margin,
                        name="loss_D_real"))
      self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                        labels=tf.zeros_like(self.D_),
                        name="loss_D_fake"))
      self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2.
      # the typical GAN set-up is that of a minimax game where D is trying to minimize its own error and G is trying to maximize D's err$
      # however note how we are flipping G labels here: instead of maximizing D's error, we are minimizing D's error on the 'wrong' label
      # this trick helps produce a stronger gradient
      self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                      labels=tf.ones_like(self.D_) + self.soft_label_margin,
                      name="loss_G"))

      # debug variables (summary - tensorboard)

      self.g_list.append(image_summary("G", self.G, max_outputs=10))
      self.g_list.append(histogram_summary("G_hist", self.G))
      self.g_list.append(scalar_summary("d_loss_fake", self.d_loss_fake))
      self.g_list.append(scalar_summary("g_loss", self.g_loss))

      self.d_list.append(image_summary("X", self.images, max_outputs=10))
      self.d_list.append(histogram_summary("X_hist", self.images))
      self.d_list.append(scalar_summary("d_loss_real", self.d_loss_real))
      self.d_list.append(scalar_summary("d_loss", self.d_loss))

      # all trainable variables
      t_vars = tf.trainable_variables()
      # G variables
      self.g_vars = [var for var in t_vars if 'g_' in var.name]
      # D variables
      self.d_vars = [var for var in t_vars if 'd_' in var.name]

      self.d_optim = tf.train.AdamOptimizer(self.learning_rate) \
        .minimize(self.d_loss, var_list=self.d_vars)
      self.g_optim = tf.train.AdamOptimizer(self.learning_rate) \
        .minimize(self.g_loss, var_list=self.g_vars)

      # Extra hook for debug: log chi-square distance between G's output histogram and the dataset's histogram
      value_range = [-1.0, 1.0]
      nbins = 100
      hist_g = tf.histogram_fixed_width(self.G, value_range, nbins=nbins, dtype=tf.float32) / nbins
      hist_images = tf.histogram_fixed_width(self.images, value_range, nbins=nbins, dtype=tf.float32) / nbins
      self.chi_square = tf.reduce_mean(tf.div(tf.square(hist_g - hist_images), hist_g + hist_images + 1e-5))
      self.d_list.append(scalar_summary("chi_square", self.chi_square))

      self.g_sum = merge_summary(self.g_list)
      self.d_sum = merge_summary(self.d_list)
    else:
        # Create only the generator
      self.z = tf.random_normal(shape=[self.batch_size, self.z_dim], dtype=tf.float32, seed=None, name='z')
      self.G = self.generator(self.z)

    init_op = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init_op)


  def load(self, path=None):
    weights_path = "weights/" + path + "/model.ckpt"

    # initialize train session
    saver = tf.train.Saver()
    try:
        saver.restore(self.sess, weights_path)
        print("Checkpoint found")
    except:
        print("No checkpoint found")
        pass

  def plot_sample(self):
    imagem = self.sess.run(self.inference())
    imgplot = plt.imshow(imagem[15])
    plt.show(imgplot)

  def train(self, restore=False, path=None):
    graph_path = "graph/" + path
    weights_path = "weights/" + path

    if not os.path.exists(graph_path):
      os.mkdir(graph_path)

    if not os.path.exists(weights_path):
      os.mkdir(weights_path)

    # create tensorboard instance
    writer = SummaryWriter(graph_path)
    saver = tf.train.Saver()

    if restore is True:
        try:
            saver.restore(self.sess, weights_path + "/model.ckpt")
            print("Checkpoint found")
        except:
            print("No checkpoint found")
        pass

    batches = self.N//self.batch_size

    for epoch in range(self.epochs):
        np.random.shuffle(self.data)
        for idx in range(batches):
            # get batch data indices
            ini = idx * self.batch_size
            end = ini + self.batch_size

            print("Get images[{0},{1}]".format(ini,end))

            # get batch data values
            batch_x = loader.files2images(self.data[ini:end])

            # update discriminator
            summary_str_d, _ = self.sess.run([self.d_sum, self.d_optim], feed_dict={self.images: batch_x})

            # update generator 2 times
            summary_str_g, _ = self.sess.run([self.g_sum, self.g_optim])
            summary_str_g, _ = self.sess.run([self.g_sum, self.g_optim])


        writer.add_summary(summary_str_d, epoch)
        writer.add_summary(summary_str_g, epoch)
        errD_fake = self.d_loss_fake.eval(session=self.sess)
        errD_real = self.d_loss_real.eval(session=self.sess, feed_dict={self.images: batch_x})
        errD = (errD_real+errD_fake)/2.
        errG = self.g_loss.eval(session=self.sess)
        save_path = saver.save(self.sess, weights_path)

        if epoch%1 == 0:
            print("At epoch {0}".format(epoch+1))
            print("Discriminator loss: {0}".format(errD))
            print("Generator loss: {0}".format(errG))
            print("\n")


if __name__ == '__main__':
  images = loader.get_celeb()
  model = Model(images[64], batch_size=64, epochs=5, is_inference=False)
  model.train(path="test1")



# if __name__ == "__main__":
#   # # Load Dataset
#     images, names, features = load_data(-1)
#     # print_subset(images, names)

#     # Before training, we need convert these images to float
#     images = np.array(images)

#     # print(images[0].shape)
#     # print(images[0])
#     # plt.imshow(images[0])

#     model = Model(images, batch_size=64,epochs=60,is_inference=False)
#     model.train(path="train_19-09-2017")

#     # model.load()
#     # model.plot_sample()
