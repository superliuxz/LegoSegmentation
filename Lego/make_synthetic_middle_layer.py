import matplotlib.pyplot as plt
import numpy as np


color = {0: 'royalblue', 1: 'gold'}


def gen_board(filename):
    base = np.zeros((16, 32))

    fig, ax = plt.subplots(figsize=(4, 3))
    # fig.patch.set_facecolor(np.random.rand(4))

    plt.imshow(base, cmap='Greys')

    base[1][30] = 200

    for i in range(np.random.randint(1, 11)):
        l = np.random.randint(1, 7)
        x = np.random.randint(1, 31-l)-0.5
        y = np.random.randint(1, 15)-0.5
        c = np.random.randint(0, 2)
        r = plt.Rectangle((x, y), l, 1, fill=True, color=color[c], antialiased=False)
        ax.add_artist(r)

    ax.xaxis.set_ticks_position('none')
    ax.set_xticklabels([])
    ax.yaxis.set_ticks_position('none')
    ax.set_yticklabels([])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0)

    plt.tight_layout(pad=0.4)

    plt.savefig(f'{filename}.png', dpi=64, facecolor=fig.get_facecolor())

    plt.cla()
    plt.close(fig)


if __name__ == '__main__':
    for i in range(5000):
        print(f'working on {i+1}...')
        gen_board(f'{i+1:04d}')
    print('done')
