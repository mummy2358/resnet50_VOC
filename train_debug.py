import tensorflow as tf
import numpy as np
import resnet50
import data_loader
import os
import utils
import cv2

def scaling(inputs):
  # scale uint8 images to [0,1] float
  inputs=np.array(inputs)
  inputs=inputs/255.0
  return inputs
  
def train(max_epoch=10000,batch_size=2,restore_iter=0):
  model=resnet50.resnet50(using_weights=True)
  loader=data_loader.VOC_loader()
  optimizer=tf.train.AdamOptimizer(1e-4)
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step=optimizer.minimize(model.loss)
    
  saver=tf.train.Saver()
  
  init=tf.global_variables_initializer()
  sess=tf.Session()
  sess.run(init)
  
  batch_per_epoch=len(loader.trainlist)//batch_size
  tool0=utils.toolbox()
  if restore_iter!=0:
    saver.restore(sess,"./models/__iter__"+str(restore_iter)+".ckpt")
    print("restoring")
  for e in range(restore_iter,max_epoch):
    train_iterator=loader.train_next_batch(batch_size)
    loss_avg=0
    for b in range(batch_per_epoch):
      batch_x,batch_y=next(train_iterator)
      # scaling
      batch_x=scaling(batch_x)
      sess.run(train_step,feed_dict={model.x:batch_x,model.y:batch_y})
      batch_loss=sess.run(model.loss,feed_dict={model.x:batch_x,model.y:batch_y})
      loss_avg+=batch_loss
    loss_avg/=batch_per_epoch
    print("epoch"+str(e+1)+":"+str(loss_avg))
    if not os.path.exists("./models"):
      os.system("mkdir ./models")
    if (e+1)%50==0:
      saver.save(sess,"./models/__iter__"+str(e+1)+".ckpt")
      pred_imgs,pred_labels=loader.prepare_val_batch([0,10])
      pred_imgs=scaling(pred_imgs)
      for i in range(len(pred_imgs)):
        pred_i=sess.run(model.logits,feed_dict={model.x:[pred_imgs[i]]})
        pred_i=pred_i[0]
        prediction=np.argmax(pred_i,axis=-1)
        pred_i=tool0.convert_image(pred_i,22)
        label_i=tool0.convert_image(pred_labels[i],22)
        if i<6:
          cv2.imshow("win0",pred_i)
          cv2.imshow("win1",label_i.astype(np.uint8))
          cv2.waitKey()
          cv2.destroyAllWindows()
if __name__=="__main__":
  train(restore_iter=100)
