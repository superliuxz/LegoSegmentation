import argparse
import csv
from functools import reduce


NUM_BLUE = 16
NUM_YELLOW = 9
BOARD_SIZE = (16, 32)


def pretty_print(board):
    print('\n'.join([''.join([f'{ele:2d}' if isinstance(ele, int) else f'{ele: >2}' for ele in row]) for row in board]))


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='the input file\n make sure each line of the input is formatted as:\n'
                                            '<image filename>,<brick row>,<brick col>,<brick span>,<brick color>,'
                                            '<repeat for another brick>')
parser.add_argument('output', type=str, help='the output file name')
args = parser.parse_args()

label = []

with open(args.input, 'r') as fin:
    for line in fin:
        line = line.strip().split(',')
        pos = 1
        board = [['0' for col in range(BOARD_SIZE[1])] for row in range(BOARD_SIZE[0])]
        blue = yellow = 0
        while pos+3 < len(line):
            col = int(line[pos])
            row = int(line[pos+1])
            span = int(line[pos+2])
            clr = line[pos+3]
            for s in range(span):
                if clr == 'y':
                    board[col - 1][row + s - 1] = 100
                    yellow += 1
                elif clr == 'b':
                    board[col - 1][row + s - 1] = 200
                    blue += 1
                else:
                    raise ValueError(f'file {line[0]} ({col},{row},{span},{clr}), clr must be "y" or "b"')
            pos += 4

        assert blue == NUM_BLUE, f'{line[0]}, blue={blue}'
        assert yellow == NUM_YELLOW, f'{line[0]}, yellow={yellow}'

        # pretty_print(board)
        # input(f'{line[0]} looks good?\n')

        label.append(reduce(lambda x, y: x+y, board))

with open(args.output, 'w') as fout:
    writer = csv.writer(fout, delimiter=',')
    writer.writerows(label)
