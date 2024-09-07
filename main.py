import numpy as np
from numba import jit, vectorize, complex64, int16

COLOR_GREEN = (106, 123, 106)
COLOR_RED = (123, 106, 106)

GRID_DEF = 2 ** 11
MAX_ITER = 200
MAX_HEIGHT = 0.1

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

def print_dimensions(file):
    vertices = (GRID_DEF + 1) ** 2
    sides = GRID_DEF ** 2

    print(f"{vertices} {sides} 0", file=file)

def print_vertices(grid, output, file):
    for compl, out in zip(np.ravel(grid), np.ravel(output)):
        print(f"{np.real(compl)} {np.imag(compl)} {np.sin(out / MAX_ITER * MAX_HEIGHT * np.pi/2)}", file=file)

def avg_height(output, vertices):
    return np.mean([output[i] for i in vertices])

def print_side(output, vertices, file):
    print(4, *vertices, end=" ", file=file)

    avg_h = avg_height(output, vertices)
    factor = 1 - (avg_h / MAX_ITER)
    color = (int(c1*factor + c2*(1-factor))
             for c1, c2 in zip(COLOR_GREEN, COLOR_RED))

    print(*color, file=file)

def print_all_sides(output, file):
    for i in range(GRID_DEF):
        for j in range(GRID_DEF):
            vertices = (i * (GRID_DEF + 1) + j,
                        i * (GRID_DEF + 1) + j + 1,
                        i * (GRID_DEF + 1) + j + 1 + GRID_DEF,
                        i * (GRID_DEF + 1) + j + GRID_DEF)
            print_side(np.ravel(output), vertices, file)

def main():
    grid = init_grid(GRID_DEF)
    output = iterate(grid)

    with open("out1.off", "w") as f:
        print("OFF", file=f)
        print_dimensions(grid, f)
        print_vertices(grid, output, f)
        print_all_sides(output, f)

if __name__ == "__main__":
    main()