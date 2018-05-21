import matplotlib.pyplot as plt
import numpy as np

import csv


color = {0: 'royalblue', 1: 'gold'}


def pretty_print(board):
    print('\n'.join([''.join([f'{ele:2d}' if isinstance(ele, int) else f'{ele: >2}' for ele in row]) for row in board]))


def gen_board(filename):
    base = np.zeros((16, 32))

    fig, ax = plt.subplots(figsize=(4, 3))

    plt.imshow(base, cmap='Greys')

    r = plt.Rectangle((30.5, -0.5), 1, 1, fill=True, color=color[0])
    base[0][31] = 200
    ax.add_artist(r)

    for i in range(12):
        l = np.random.randint(1, 6)
        x = np.random.randint(0, 32-l)-0.5
        y = np.random.randint(0, 16)-0.5
        c = np.random.randint(0, 2)
        r = plt.Rectangle((x, y), l, 1, fill=True, color=color[c])
        for j in range(l):
            if c == 0:
                base[int(y+0.5)][int(x+0.5+j)] = 200
            elif c == 1:
                base[int(y+0.5)][int(x+0.5+j)] = 100
        ax.add_artist(r)

    for row in range(base.shape[0]):
        for col in range(base.shape[1]):
            c = plt.Circle((col, row), 0.25, fill=False, lw=0.3)
            ax.add_artist(c)

    ax.xaxis.set_ticks_position('none')
    ax.set_xticklabels([])
    ax.yaxis.set_ticks_position('none')
    ax.set_yticklabels([])

    plt.tight_layout(pad=0.4)
    plt.savefig(f'{filename}.jpg', dpi=64)
    plt.cla()
    plt.close(fig)

    return base.astype(np.int).flatten()


if __name__ == '__main__':
    res = []
    
    for i in range(5000):
        r = gen_board(str(i+1).zfill(5))
        res.append(r)
    with open('output.txt', 'w') as fout:
        writer = csv.writer(fout, delimiter=',')
        writer.writerows(res)
