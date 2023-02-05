#!/usr/bin/env python
# coding: utf-8
import numpy as np
import exact_cover as ec
import pprint

def checkline(l):
    standard = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    l = l.flatten()
    return np.linalg.norm(sorted(l) - standard) == 0


def is_correct(matrix):  # input  9 * 9 matrix
    matrix = np.array(matrix)
    for i in range(9):
        if not checkline(matrix[i, :]):
            return False
        if not checkline(matrix[:, i]):
            return False
        blx, bly = i // 3, i % 3
        if not checkline(matrix[blx * 3:blx * 3 + 3, bly * 3:bly * 3 + 3]):
            return False
    return True


def construct_matrix(matrix):
    index = np.array(range(81)).reshape(9, 9)

    # 729维向量
    zero_one = []

    for i in range(81):  # 每个格子只能选一个数, 81个限制
        # 格子i的范围 9*i～9*i+9, 9*i + j 中的值为1 代表格子i选择了j
        zero_one.append([0] * 729)
        for j in range(9 * i, 9 * i + 9):
            zero_one[i][j] = 1

    col_constraint = []
    for k in range(9):  # 第k列
        hits = index[:, k]  # 第k列的格子的index

        for j in range(9):  # 第k列中关于j的限制

            # 每列包含1～9
            col_constraint.append([0] * 729)

            cur = len(col_constraint) - 1

            for i in hits:
                col_constraint[cur][9 * i + j] = 1

    row_constraint = []
    for k in range(9):  # 第k行
        hits = index[k, :]  # 第k行的格子的index

        for j in range(9):  # 第k列中关于j的限制

            # 每列包含1～9
            row_constraint.append([0] * 729)

            cur = len(row_constraint) - 1

            for i in hits:
                row_constraint[cur][9 * i + j] = 1

    block_constraint = []
    for k in range(9):  # 第k块
        blx, bly = k // 3, k % 3

        hits = index[blx * 3:blx * 3 + 3, bly * 3:bly * 3 + 3].flatten()
        # 第k块的格子的index

        for j in range(9):  # 第k块中关于j的限制

            # 每列包含1～9
            block_constraint.append([0] * 729)

            cur = len(block_constraint) - 1

            for i in hits:
                block_constraint[cur][9 * i + j] = 1

    general = np.array(zero_one + col_constraint + row_constraint + block_constraint).T
    # 最终需要转置

    pos = 0
    pos_index = {}
    final = []
    for i in range(81):
        row, col = i // 9, i % 9
        pos_index[i] = pos  # start position
        if matrix[row][col] != 0:  # already has value
            val = matrix[row][col]
            final.append(general[9 * i + val - 1])
            pos += 1
        else:  # no value
            for j in range(9):
                final.append(general[9 * i + j])
            pos += 9

    return np.array(final), pos_index


def get_solution(matrix):
    m, pos_index = construct_matrix(matrix)
    # print(pos_index)
    res = ec.get_exact_cover(m)
    # print(res)

    pos = 0
    solution = np.zeros((9, 9))
    for i in range(81):
        row, col = i // 9, i % 9

        if matrix[row][col] != 0:
            solution[row][col] = matrix[row][col]
        else:
            for j in range(pos_index[i], pos_index[i] + 9):
                if j in res:
                    solution[row][col] = j - pos_index[i] + 1
    return solution.astype(int).tolist()


if __name__ == '__main__':
    demo = [[8, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 6, 0, 0, 0, 0, 0],
            [0, 7, 0, 0, 9, 0, 2, 0, 0],
            [0, 5, 0, 0, 0, 7, 0, 0, 0],
            [0, 0, 0, 0, 4, 5, 7, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 3, 0],
            [0, 0, 1, 0, 0, 0, 0, 6, 8],
            [0, 0, 8, 5, 0, 0, 0, 1, 0],
            [0, 9, 0, 0, 0, 0, 4, 0, 0]]

    pprint.pprint(get_solution(demo))
