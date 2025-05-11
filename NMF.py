import numpy as np


def spa(x, r):
    """
    :param r: Input data
    :param x: Rank
    :return: Column indices sufficing local-dominance
    """
    m, n = x.shape
    max_x = np.max(np.max(x))
    threshold = max_x / 1e6
    sum_x = np.sum(np.square(x), axis=0)
    z = np.zeros([m, n])

    for nn in range(0, n):
        if np.sum(x[:, nn]) < threshold:
            z[:, nn] = z[:, nn] * 1
        else:
            z[:, nn] = x[:, nn] / np.sum(x[:, nn])

    rmat = z.copy()
    idx = []

    for rr in range(0, r):
        rmat_l2 = np.sum(np.square(rmat), axis=0)
        max_l2 = np.max(rmat_l2)
        index = np.where(rmat_l2 == max_l2)[0]
        select_mat = sum_x[index] * 1
        indexmax = np.where(select_mat == np.max(select_mat))[0]
        indexrr = index[indexmax[0]]
        if rr == 0:
            idx = indexrr
        else:
            idx = np.append(idx, indexrr)

        # projection
        rmat_2 = np.square(np.linalg.norm(rmat[:, indexrr], ord=2))
        rmat_1 = np.eye(m) - np.outer(rmat[:, indexrr], rmat[:, indexrr]) / rmat_2
        rmat = np.dot(rmat_1, rmat)

    return idx


def tensor_unfold_m3(x):
    i, j, k = x.shape

    x3 = np.zeros([i * j, k])

    for ii in range(0, i):

        for jj in range(0, j):
            x_fiber = x[ii, jj, :]
            idx = ii + jj * i
            x3[idx, :] = x_fiber

    return x3


def get_unfold_location(coor, i):
    coornum, dim = coor.shape
    unfoled_idx = np.zeros([coornum, 1])
    for mm in range(0, coornum):
        i_idx = coor[mm, 0]
        j_idx = coor[mm, 1]
        unfoled_idx[mm] = i_idx - 1 + (j_idx - 1) * i

    return unfoled_idx
