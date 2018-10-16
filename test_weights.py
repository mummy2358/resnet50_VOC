from keras.applications.resnet50 import ResNet50
import numpy as np

model = ResNet50(weights='imagenet')

names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()

for name, weight in zip(names, weights):
  if "branch1" in name:
    print(name, weight.shape)
np.save("resnet50_weights.npy",weights)



