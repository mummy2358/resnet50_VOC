import numpy as np
from skimage import io
from skimage import transform
from skimage import img_as_ubyte
import os
import utils
import cv2

def skimage_read_BGR(img_dir):
  img=io.imread(img_dir)
  dur=img[:,:,0]
  img[:,:,0]=img[:,:,2]
  img[:,:,2]=dur
  return img

class VOC_loader:
  def __init__(self,root="../VOCdevkit/VOC2012/"):
    self.classnum=22
    self.root=root
    self.trainlist=open(root+"ImageSets/Segmentation/train.txt").read().split("\n")
    self.trainlist=self.trainlist[0:100]
    self.trainlist=[x for x in self.trainlist if x is not None]
    self.vallist=open(root+"ImageSets/Segmentation/val.txt").read().split("\n")
    self.vallist=[x for x in self.vallist if x is not None]
    self.sample_counter=0
    self.indices=list(range(np.shape(self.trainlist)[0]))
    self.hwc=(224,224,3)
    
  def train_next_batch(self,batch_size=10):
    self.sample_counter=0
    batch_x=[]
    batch_y=[]
    while True:
      print(self.trainlist)
      input_image=cv2.imread(self.root+"JPEGImages/"+str(self.trainlist[self.indices[self.sample_counter]])+".jpg")
      label_image=cv2.imread(self.root+"SegmentationClass/"+str(self.trainlist[self.indices[self.sample_counter]])+".png")
      input_image=self.preprocess(input_image)
      label=self.label_convertion(label_image)
      batch_x.append(input_image)
      batch_y.append(label)
      self.sample_counter+=1
      if len(batch_x)==batch_size or self.sample_counter==len(self.trainlist)-1:
        yield batch_x,batch_y
        batch_x=[]
        batch_y=[]
      if self.sample_counter==len(self.trainlist)-1:
        # shuffle indices
        self.indices=np.random.shuffle(self.indices)
        self.sample_counter=0
        
  def test_func(self):
    img=skimage_read_BGR(self.root+"SegmentationClass/"+self.trainlist[0]+".png")
    img=cv2.imread(self.root+"SegmentationClass/"+self.trainlist[0]+".png")
    img=self.preprocess(img)
    tool0=utils.toolbox()
    label=tool0.convert_label(img,self.classnum)
    #print(label)
    label_image=tool0.convert_image(label,self.classnum)
    cv2.imshow("win1",label_image.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
  def preprocess(self,img):
    # preprocess for input images such as resize, normalize, change type, scaling
    # remember to keep data type in mind
    img=transform.resize(img,[self.hwc[0],self.hwc[1]])
    img=img_as_ubyte(img)
    return img
  
  def label_convertion(self,label_image):
    # convert from raw images to pixel-wise labels
    label_image=transform.resize(label_image,[self.hwc[0],self.hwc[1]])
    label_image=img_as_ubyte(label_image)
    tool0=utils.toolbox()
    label=tool0.convert_label(label_image,self.classnum)
    return label
  
  def prepare_val_batch(self,index=[0,10]):
    # input an [lower,upper] index range to get val images and labels from self.vallist
    batch_x=[]
    batch_y=[]
    for name in self.vallist[index[0]:index[1]]:
      input_image=cv2.imread(self.root+"JPEGImages/"+name+".jpg")
      label_image=cv2.imread(self.root+"SegmentationClass/"+name+".png")
      input_image=self.preprocess(input_image)
      label_image=self.label_convertion(label_image)
      batch_x.append(input_image)
      batch_y.append(label_image)
    return batch_x,batch_y
  
  def prepare_train_batch(self,index=[0,10]):
    # input an [lower,upper] index range to get val images and labels from self.vallist
    batch_x=[]
    batch_y=[]
    for name in self.trainlist[index[0]:index[1]]:
      input_image=cv2.imread(self.root+"JPEGImages/"+name+".jpg")
      label_image=cv2.imread(self.root+"SegmentationClass/"+name+".png")
      input_image=self.preprocess(input_image)
      label_image=self.label_convertion(label_image)
      batch_x.append(input_image)
      batch_y.append(label_image)
    return batch_x,batch_y

loader=VOC_loader()
train=loader.train_next_batch(batch_size=2)
tool0=utils.toolbox()
for i in range(3):
  batch_x,batch_y=next(train)
  label_img=tool0.convert_image(batch_y[0],22)
  cv2.imshow("image",batch_x[0])
  cv2.imshow("label",label_img.astype(np.uint8))
  cv2.waitKey()
  cv2.destroyAllWindows()

