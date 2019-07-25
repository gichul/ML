import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
import numpy as np
from PIL import Image
from processing import *

def img_show(img):
	pil_img=Image.fromarray(np.uint8(img))
	pil_img.show()
'''
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False)

	
#print(x_train.shape)
#print(t_train.shape)
#print(x_test.shape)
#print(t_test.shape)

img =x_test[0]
print(x_test[0])
label=t_test[0]
print(label)

print(img.shape)
img=img.reshape(28,28)
print(img.shape)

img_show(img)

'''

'''
x,y=get_data()
print(x[0])
#print(np.sum(np.array(x[0])))
img=x[0]
img=img.reshape(28,28)
print(img.shape)
img_show(img)

label=y[0]
print(label)
'''

accuracy()
