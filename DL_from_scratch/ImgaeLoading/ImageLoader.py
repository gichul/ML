import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./bird.png')

plt.imshow(img)
plt.show()

