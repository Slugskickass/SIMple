import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def generate_patter(size, K_vector, ):
    t = size
    u = np.linspace(0, t-1, t)
    v = np.linspace(0, t-1, t)
    [U, V] = np.meshgrid(u, v)
    to = t/2
    i = np.complex(0, 1)
    S1aT = np.exp(-1*i * 2*np.pi*( K_vector[0]/t*(U-to)+K_vector[1]/t*(V-to)))
    return(S1aT)

def get_image(filename, frame):
    image = Image.open(filename)
    image.seek(frame)
    data = np.asarray(image)
    return(data)

def generate_PSF(NA, lamda, pixel_size, frame_size):

    return(image)

filename = '/Users/Ashley/PycharmProjects/SIMple/Data/SLM-SIM_Tetraspeck200_680nm.tif'

image_data = get_image(filename, 1)
K_vector = [10, 10]
S1aT = generate_patter(512, K_vector)
plt.imshow(np.real(S1aT))
plt.show()


plt.imshow(np.real(image_data))
plt.show()





