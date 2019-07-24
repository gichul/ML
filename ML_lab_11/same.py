import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
#plt.imshow(image.reshape(2,2))
#plt.show()
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
print(pool.shape)
#print(pool.eval())
