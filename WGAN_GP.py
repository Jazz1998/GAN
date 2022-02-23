import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,losses,optimizers,datasets
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
#生成器
class Generator(keras.Model):
  def __init__(self):
    super(Generator,self).__init__()

    filter = 32

    #第一个卷积层
    self.conv1 = layers.Conv2DTranspose(filter*8,4,1,'valid',use_bias=False)
    self.bn1 = layers.BatchNormalization()

    #第二个卷积层
    self.conv2 = layers.Conv2DTranspose(filter*4,4,2,'same',use_bias=False)
    self.bn2 = layers.BatchNormalization()

    #第三个卷积层
    self.conv3 = layers.Conv2DTranspose(filter*2,4,2,'same',use_bias=False)
    self.bn3 = layers.BatchNormalization()

    #第四个卷积层
    self.conv4 = layers.Conv2DTranspose(3,4,2,'same',use_bias=False)

  def call(self, inputs, training=None):
    x = inputs
    x = tf.reshape(x,[x.shape[0],1,1,x.shape[1]])

    #卷积-bn-激活:(b,4,4,256)
    x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
    #卷积-bn-激活:(b,8,8,128)
    x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
    #卷积-bn-激活:(b,16,16,64)
    x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
    #卷积:(b,32,32,3)
    x = self.conv4(x)

    #使用tanh激活函数，它的值范围是-1～1，与数据集中的数据保持一致
    x = tf.nn.tanh(x)

    return x
#鉴别器
class Discriminator(keras.Model):
  def __init__(self):
    super(Discriminator,self).__init__()

    filter = 32

    #第一个卷积层
    self.conv1 = layers.Conv2D(filter*2,4,2,'same',use_bias=False)
    self.bn1 = layers.BatchNormalization()

    #第二个卷积层
    self.conv2 = layers.Conv2D(filter*4,4,2,'same',use_bias=False)
    self.bn2 = layers.BatchNormalization()

    #第三个卷积层
    self.conv3 = layers.Conv2D(filter*8,4,2,'same',use_bias=False)
    self.bn3 = layers.BatchNormalization()

    #第四个卷积层
    self.conv4 = layers.Conv2D(filter*16,4,2,'same',use_bias=False)
    self.bn4 = layers.BatchNormalization()

    self.conv5 = layers.Conv2D(1,2,1,'valid',use_bias=False)

    #池化层
    # self.pool = layers.GlobalAveragePooling2D()
    # #打平层
    # self.flatten = layers.Flatten()

    # #输出层
    # self.fc = layers.Dense(1)

  def call(self, inputs, training=None):
    #卷积-bn-激活:(b,16,16,64)
    x = tf.nn.leaky_relu(self.bn1(self.conv1(inputs), training=training))
    #卷积-bn-激活:(b,8,8,128)
    x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
    #卷积-bn-激活:(b,4,4,256)
    x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
    #卷积-bn-激活:(b,2,2,512)
    x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))


    # x = self.pool(x)
    # x = self.flatten(x)

    # logits = self.fc(x)

    x = self.conv5(x)
    logits = tf.reshape(x,[x.shape[0],-1])

    return logits
#WGAN-GP的梯度惩罚函数
def gradient_penalty(discriminator, batch_x, fake_image):
  #梯度惩罚项计算函数
  batchsz = batch_x.shape[0]

  #每个样本均随机采样t,用于插值
  t = tf.random.uniform([batchsz,1,1,1])
  #自动扩展为x的形状:[b,1,1,1] => [b,h,w,c]
  t = tf.broadcast_to(t, batch_x.shape)
  #在真假图片之间作线性插值
  interplate = t * batch_x + (1-t) * fake_image
  #在梯度环境中计算 D 对插值样本的梯度
  with tf.GradientTape() as tape:
    tape.watch([interplate])
    d_interplate_logits = discriminator(interplate)

  grads = tape.gradient(d_interplate_logits, interplate)

  #计算每个样本的梯度的范数:[b,h,w,c] => [b,-1]
  grads = tf.reshape(grads, [grads.shape[0],-1])
  gp = tf.norm(grads,axis=1)
  #计算梯度惩罚项
  gp = tf.reduce_mean((gp-1.)**2)

  return gp
#鉴别器的损失函数
def d_loss_fn(generator,discriminator,batch_z,batch_x,is_training):
  #计算判别器的损失函数

  #生成样本
  fake_image = generator(batch_z,is_training)
  #判别生成样本
  d_fake_logits = discriminator(fake_image,is_training)
  #判别真实样本
  d_real_logits = discriminator(batch_x,is_training)
  # 计算梯度惩罚项
  gp = gradient_penalty(discriminator,batch_x,fake_image)

  loss = tf.reduce_mean(d_fake_logits) - tf.reduce_mean(d_real_logits) + 10.*gp

  return loss, gp
# 生成器的损失函数
def g_loss_fn(generator,discriminator,batch_z,is_training):
  #计算生成器的损失函数
  fake_image = generator(batch_z,is_training)
  d_fake_logits = discriminator(fake_image,is_training)
  #最大化假样本的输出值
  loss = -tf.reduce_mean(d_fake_logits)

  return loss

# 定义超参数
epochs = 3000000          # 训练回合数
batch_sz = 64   # 批处理大小
learn_rate = 0.0001   # 学习率
is_training = True   # 是否训练的标志
z_dim = 100   # 采样向量的长度



(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(x_train.shape)

# 数据预处理
def preprocess(x):
  #预处理函数
  x = tf.cast(x,dtype=tf.float32)/127.5 - 1
  return x

#开始处理数据
train_db = tf.data.Dataset.from_tensor_slices((x_train))
train_db = train_db.shuffle(1000).map(preprocess).batch(batch_sz)

sample = next(iter(train_db))
print(sample.shape)


'''
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
print(x_train.shape)
#开始处理数据

def preprocess(x):
  #预处理函数
  x = x.reshape(x.shape[0], 28, 28, 1).astype('float32')
  x = tf.cast(x,dtype=tf.float32)/127.5 - 1
  return x

train_db = tf.data.Dataset.from_tensor_slices((x_train))
train_db = train_db.shuffle(1000).map(preprocess).batch(batch_sz)

sample = next(iter(train_db))
print(sample.shape)
'''

#生成器
generator = Generator()
#判别器
discriminator = Discriminator()

generator.build(input_shape=(4,z_dim))
generator.summary()

discriminator.build(input_shape=(4,32,32,3))
discriminator.summary()

#g_optimizer = optimizers.Adam(learn_rate,beta_1=0.5)
#d_optimizer = optimizers.Adam(learn_rate,beta_1=0.5)

#生成器的优化器
g_optimizer = optimizers.RMSprop(learn_rate)
#判别器的优化器
d_optimizer = optimizers.RMSprop(learn_rate)

#保存图片
def save_image(imgs,name):
  new_imgs = Image.new('RGB',(320,320))

  index = 0
  for i in range(0,320,32):
    for j in range(0,320,32):

      img = imgs[index]
      img = ((img+1)*127.5).astype(np.uint8)
      img = Image.fromarray(img,mode='RGB')

      new_imgs.paste(img,(i,j))

      index += 1

  new_imgs.save(name)

d = []
g = []
#开始训练
for epoch in range(epochs):
  for _ in range(1):
    batch_z = tf.random.normal([batch_sz,z_dim])
    batch_x = next(iter(train_db))

    with tf.GradientTape() as tape:
      d_loss = d_loss_fn(generator,discriminator,batch_z,batch_x,is_training)[0]
      d.append(d_loss)
    grads = tape.gradient(d_loss,discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads,discriminator.trainable_variables))

  batch_z = tf.random.normal([batch_sz,z_dim])
  with tf.GradientTape() as tape:
    g_loss = g_loss_fn(generator,discriminator,batch_z,is_training)
    g.append(g_loss)
  grads = tape.gradient(g_loss,generator.trainable_variables)
  g_optimizer.apply_gradients(zip(grads,generator.trainable_variables))

  if epoch!=0 and epoch %100 ==0:
    print(epoch, ' d-loss:',float(d_loss), ' g_loss:',float(g_loss))

    batch_z = tf.random.normal([100, z_dim])
    fake_images = generator(batch_z, False)
    #save_image(fake_images.numpy(), 'wgan-gp%d.png'%epoch)

e = range(1, len(d) + 1)
plt.plot(e, d, 'b', color='blue', label='D loss')
plt.plot(e, g, 'b', color='blue', label='G loss')
plt.show()