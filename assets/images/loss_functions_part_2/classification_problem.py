from PIL import Image
import numpy as np
npimg = np.zeros((500,500))

npimg[50:100, 250:300] = 255

im = Image.fromarray(npimg.astype(np.uint8))

im.save('classification_problem.png')
