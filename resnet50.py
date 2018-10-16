from keras.applications.resnet50 import ResNet50
import numpy as np
import tensorflow as tf

# only use keras to get pretrained weights
class resnet50:
  def __init__(self,using_weights=True):
    self.build_initializers(using_weights)
    self.hwc=(224,224,3)
    self.class_num=22
    self.x=tf.placeholder(tf.float32,shape=[None,self.hwc[0],self.hwc[1],self.hwc[2]])
    self.y=tf.placeholder(tf.float32,shape=[None,self.hwc[0],self.hwc[1],self.class_num])
    self.logits=self.forward(self.x)
    self.loss=tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits),axis=[1,2]),axis=0)
    
  def build_initializers(self,using_weights):
    model = ResNet50(weights='imagenet')
    namelist=names = [weight.name for layer in model.layers for weight in layer.weights]
    weights=model.get_weights()
    self.init_dict={}
    for i in range(len(namelist)):
      if using_weights:
        self.init_dict[namelist[i]]=tf.constant_initializer(weights[i])
      else:
        self.init_dict[namelist[i]]=None
  
  def res_block(self,inputs,block_name="2a",filters=[64,64,256],kernel_size=[1,3,1],firstconv_strides=1):
    # here we suppose the block contain 3 consequtive convolution layers and one shortcut
    res_branch1=tf.layers.conv2d(inputs,filters=filters[-1],kernel_size=1,strides=firstconv_strides)
    
    #res_branch1=tf.layers.batch_normalization(res_branch1,training=True,beta_initializer=self.init_dict["bn"+block_name+"_branch1/beta:0"],gamma_initializer=self.init_dict["bn"+block_name+"_branch1/gamma:0"],moving_mean_initializer=self.init_dict["bn"+block_name+"_branch1/moving_mean:0"],moving_variance_initializer=self.init_dict["bn"+block_name+"_branch1/moving_variance:0"],name="bn"+block_name+"_branch1")
    res_branch1=tf.nn.relu(res_branch1)
    
    res_branch2=tf.layers.conv2d(inputs,filters=filters[0],kernel_size=kernel_size[0],strides=firstconv_strides,padding="same",kernel_initializer=self.init_dict["res"+block_name+"_branch2a/kernel:0"],bias_initializer=self.init_dict["res"+block_name+"_branch2a/bias:0"],trainable=True,name="res"+block_name+"_branch2a")
    res_branch2=tf.layers.batch_normalization(res_branch2,training=True,beta_initializer=self.init_dict["bn"+block_name+"_branch2a/beta:0"],gamma_initializer=self.init_dict["bn"+block_name+"_branch2a/gamma:0"],moving_mean_initializer=self.init_dict["bn"+block_name+"_branch2a/moving_mean:0"],moving_variance_initializer=self.init_dict["bn"+block_name+"_branch2a/moving_variance:0"],name="bn"+block_name+"_branch2a")
    res_branch2=tf.nn.relu(res_branch2)
    
    res_branch2=tf.layers.conv2d(res_branch2,filters=filters[1],kernel_size=kernel_size[1],strides=1,padding="same",kernel_initializer=self.init_dict["res"+block_name+"_branch2b/kernel:0"],bias_initializer=self.init_dict["res"+block_name+"_branch2b/bias:0"],trainable=True,name="res"+block_name+"_branch2b")
    res_branch2=tf.layers.batch_normalization(res_branch2,training=True,beta_initializer=self.init_dict["bn"+block_name+"_branch2b/beta:0"],gamma_initializer=self.init_dict["bn"+block_name+"_branch2b/gamma:0"],moving_mean_initializer=self.init_dict["bn"+block_name+"_branch2b/moving_mean:0"],moving_variance_initializer=self.init_dict["bn"+block_name+"_branch2b/moving_variance:0"],name="bn"+block_name+"_branch2b")
    res_branch2=tf.nn.relu(res_branch2)
    
    res_branch2=tf.layers.conv2d(res_branch2,filters=filters[2],kernel_size=kernel_size[2],strides=1,padding="same",kernel_initializer=self.init_dict["res"+block_name+"_branch2c/kernel:0"],bias_initializer=self.init_dict["res"+block_name+"_branch2c/bias:0"],trainable=True,name="res"+block_name+"_branch2c")
    res_branch2=tf.layers.batch_normalization(res_branch2,training=True,beta_initializer=self.init_dict["bn"+block_name+"_branch2c/beta:0"],gamma_initializer=self.init_dict["bn"+block_name+"_branch2c/gamma:0"],moving_mean_initializer=self.init_dict["bn"+block_name+"_branch2c/moving_mean:0"],moving_variance_initializer=self.init_dict["bn"+block_name+"_branch2c/moving_variance:0"],name="bn"+block_name+"_branch2c")
    res_branch2=tf.nn.relu(res_branch2)
    
    block=tf.add(res_branch1,res_branch2)
    return block
    
  def forward(self,inputs):
    with tf.variable_scope("resnet50",reuse=tf.AUTO_REUSE) as scope:
      conv1=tf.layers.conv2d(inputs,filters=64,kernel_size=7,strides=2,padding="same",kernel_initializer=self.init_dict["conv1/kernel:0"],bias_initializer=self.init_dict["conv1/bias:0"],trainable=True,name="conv1")
      bn_conv1=tf.layers.batch_normalization(conv1,training=True,beta_initializer=self.init_dict["bn_conv1/beta:0"],gamma_initializer=self.init_dict["bn_conv1/gamma:0"],moving_mean_initializer=self.init_dict["bn_conv1/moving_mean:0"],moving_variance_initializer=self.init_dict["bn_conv1/moving_variance:0"],name="bn_conv1")
      pool1=tf.layers.max_pooling2d(bn_conv1,pool_size=3,strides=2,padding="same")
      
      res2a=self.res_block(pool1,block_name="2a",filters=[64,64,256],kernel_size=[1,3,1],firstconv_strides=2)
      
      res2b=self.res_block(res2a,block_name="2b",filters=[64,64,256],kernel_size=[1,3,1])
      res2c=self.res_block(res2b,block_name="2c",filters=[64,64,256],kernel_size=[1,3,1])
      
      res3a=self.res_block(res2c,block_name="3a",filters=[128,128,512],kernel_size=[1,3,1],firstconv_strides=2)
      res3b=self.res_block(res3a,block_name="3b",filters=[128,128,512],kernel_size=[1,3,1])
      res3c=self.res_block(res3b,block_name="3c",filters=[128,128,512],kernel_size=[1,3,1])
      res3d=self.res_block(res3c,block_name="3d",filters=[128,128,512],kernel_size=[1,3,1])
      
      res4a=self.res_block(res3d,block_name="4a",filters=[256,256,1024],kernel_size=[1,3,1],firstconv_strides=2)
      res4b=self.res_block(res4a,block_name="4b",filters=[256,256,1024],kernel_size=[1,3,1])
      res4c=self.res_block(res4b,block_name="4c",filters=[256,256,1024],kernel_size=[1,3,1])
      res4d=self.res_block(res4c,block_name="4d",filters=[256,256,1024],kernel_size=[1,3,1])
      res4e=self.res_block(res4d,block_name="4e",filters=[256,256,1024],kernel_size=[1,3,1])
      res4f=self.res_block(res4e,block_name="4f",filters=[256,256,1024],kernel_size=[1,3,1])
      """
      res5a=self.res_block(res4f,block_name="5a",filters=[512,512,2048],kernel_size=[1,3,1],firstconv_strides=2)
      res5b=self.res_block(res5a,block_name="5b",filters=[512,512,2048],kernel_size=[1,3,1])
      res5c=self.res_block(res5b,block_name="5c",filters=[512,512,2048],kernel_size=[1,3,1])
      """
      # by here we should get 7*7 images if input is 224*224
      conv6=tf.layers.conv2d(res4f,filters=512,kernel_size=7,padding="same")
      conv6=tf.nn.leaky_relu(conv6)
      conv7=tf.layers.conv2d(conv6,filters=512,kernel_size=1,padding="same")
      conv7=tf.nn.leaky_relu(conv7)
      upsampling=tf.image.resize_bilinear(conv7,size=[tf.shape(inputs)[1],tf.shape(inputs)[2]])
      scoring=tf.layers.conv2d(upsampling,filters=self.class_num,kernel_size=1,padding="same")
      
      return scoring
