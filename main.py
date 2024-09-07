import numpy as np
import time
from numba import jit, vectorize, complex64, int16

GRID_DEF = 2 ** 7
MAX_ITER = 100

@jit(nopython=True)
def init_grid(grid_def: int) -> np.array:
    arr_length = grid_def+1
    res: np.array = np.empty((arr_length, arr_length), dtype=np.complex64)

    for x in range(arr_length):
        for y in range(arr_length):
            res[y][x] = (4 * x / grid_def - 2) + (4 * y / grid_def - 2) * 1j

    return res

@vectorize([int16(complex64)])
def iterate(point: np.complex64):
    curr = 0

    output = 0
    for _ in range(MAX_ITER):
        curr = curr*curr + point

        if np.absolute(curr) > 2:
            break

        output = output+1

    return output

def main():
    grid = init_grid(GRID_DEF)
    output = iterate(grid)

if __name__ == "__main__":
    main()