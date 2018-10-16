from os import listdir
import numpy as np

class toolbox:
  def __init__(self):
    self.class_names=['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    self.class_num=np.shape(self.class_names)[0]
    self.cmap=self.get_colormap(256)
    self.hashmap=np.zeros([256,256,256])
    for i in range(len(self.class_names)-1):
      for var0 in range(-1,2):
        for var1 in range(-1,2):
          for var2 in range(-1,2):
            self.hashmap[self.truncate(int(self.cmap[i,0])+var0,[0,255]),self.truncate(int(self.cmap[i,1])+var1,[0,255]),self.truncate(int(self.cmap[i,2])+var2,[0,255])]=i
    self.hashmap[int(self.cmap[-1,0]),int(self.cmap[-1,1]),int(self.cmap[-1,2])]=self.class_num-1
    #### notice that cmap is in RGB order and cmap[21] is not(actually it's cmap[-1]) the color for edge !!!!


  def truncate(self,x,scale):
    # scale is [min,max]
    if x<scale[0]:
      return scale[0]
    elif x>scale[1]:
      return scale[1]
    else:
      return x
  
  def get_colormap(self,N):
    def get_bit(bitmap,place):
      return (bitmap&(1<<place)!=0)
    cmap=np.zeros([N,3])
    for i in range(N):
      idx=i
      r=0
      g=0
      b=0
      for j in range(8):
        r=r|get_bit(idx,0)<<(7-j)
        g=g|get_bit(idx,1)<<(7-j)
        b=b|get_bit(idx,2)<<(7-j)
        idx=idx>>3
      cmap[i,0]=r
      cmap[i,1]=g
      cmap[i,2]=b
    return cmap

  def n2onehot(self,n,N):
    # convert index n to N-dim one-hot vector
    ans=np.zeros(N)
    ans[n]=1
    return ans

  def convert_label(self,img,N):
    # read in label image and convert to one-hot numpy arrays
    # img: HWC; BGR order!!!!
    sh=np.shape(img)
    label=np.zeros([sh[0],sh[1],N])
    for h in range(sh[0]):
      for w in range(sh[1]):
        label[h,w]=self.n2onehot(int(self.hashmap[int(img[h,w,2]),int(img[h,w,1]),int(img[h,w,0])]),N)
    return label

  def convert_image(self,label,N):
    # read in one-hot label and convert to BGR color image
    # img: HWC
    sh=np.shape(label)
    img=np.zeros([sh[0],sh[1],3])
    for h in range(sh[0]):
      for w in range(sh[1]):
        label_num=np.argmax(label[h,w])
        if label_num==self.class_num-1:
          dur=self.cmap[-1]
        else:
          dur=self.cmap[label_num]
        img[h,w]=[dur[2],dur[1],dur[0]]
    return img
