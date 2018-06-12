import cv2
import numpy as np

def make_noise(image):
    mean = 0
    sigma = 20
    gauss = np.random.normal(mean,sigma,image.shape)
    noisy = image + gauss
    noisy = np.clip(noisy,0,255)
    return noisy.astype(np.uint8)

for i in range(5000):
    print(f'working on {i+1:04d}_board.png')
    img = cv2.imread(f'{i+1:04d}_board.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    noisy = make_noise(img)
    # print(noisy.shape)
    cv2.imwrite(f'{i+1:04d}_noisy_board.png', cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))
