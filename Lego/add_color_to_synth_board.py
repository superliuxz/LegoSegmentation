import cv2
import numpy as np

for i in range(5000):
    print(f'working on {i+1:04d}.png')
    img = cv2.imread(f'{i+1:04d}x.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    blue = img.copy()
    yellow = img.copy()

    blue[(img==[65, 105, 225]).all(axis=-1)] = [255,255,255]
    blue[(img!=[65, 105, 225]).all(axis=-1)] = [0,0,0]
    yellow[(img==[255, 215, 0]).all(axis=-1)] = [255,255,255]
    yellow[(img!=[255, 215, 0]).all(axis=-1)] = [0,0,0]

    cv2.imwrite(f'{i+1:04d}_blue.png', cv2.cvtColor(blue, cv2.COLOR_RGB2GRAY))
    cv2.imwrite(f'{i+1:04d}_yellow.png', cv2.cvtColor(yellow, cv2.COLOR_RGB2GRAY))

    board = cv2.imread(f'board.png')
    board = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)

    board[(img==[65, 105, 225]).all(axis=-1)] = [65, 105, 225]
    board[(img==[255, 215, 0]).all(axis=-1)] = [255, 215, 0]

    cv2.imwrite(f'{i+1:04d}_board.png', cv2.cvtColor(board, cv2.COLOR_RGB2BGR))

print('done')
