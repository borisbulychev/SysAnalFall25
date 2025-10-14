from typing import Tuple, List
import re
import numpy as np
from collections import deque
import pandas as pd


def main(s: str, e: str) -> Tuple[
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
    List[List[bool]],
]:
    """
    Построение пяти матриц отношений для ориентированного графа,
    заданного парами натуральных чисел в строке s.

    Параметры
    ----------
    s : str
        Текст/CSV-строка со списком рёбер. Извлекаются все натуральные
        числа и группируются по два подряд как пары (u, v).
    e : str
        Идентификатор корневого узла.
        Если корень отсутствует среди вершин или парсинг неудачен —
        возвращается ([], [], [], [], []).

    Возвращаемое значение
    -------
    Tuple из пяти матриц (List[List[bool]]): (A_r1, A_r2, A_r3, A_r4, A_r5).
    """
    if not isinstance(s, str) or not isinstance(e, str):
        return [], [], [], [], []

    try:
        root = int(e)
    except Exception:
        return [], [], [], [], []

    nums = re.findall(r'\d+', s)
    if not nums:
        return [], [], [], [], []

    verts_all = [int(x) for x in nums]
    if len(verts_all) % 2 == 1:
        verts_all = verts_all[:-1]

    edges = [(verts_all[i], verts_all[i + 1])
             for i in range(0, len(verts_all), 2)]
    if not edges:
        return [], [], [], [], []

    vertices = sorted({v for pair in edges for v in pair})
    if root not in vertices:
        return [], [], [], [], []

    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    root_idx = idx[root]

    A_r1 = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        A_r1[idx[u], idx[v]] = True

    A_r2 = A_r1.T.copy()

    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[idx[u]].append(idx[v])

    depth = np.full(n, -1, dtype=int)
    dq = deque([root_idx])
    depth[root_idx] = 0
    while dq:
        u = dq.popleft()
        for w in adj_list[u]:
            if depth[w] == -1:
                depth[w] = depth[u] + 1
                dq.append(w)

    R = A_r1.copy()
    for k in range(n):
        col = R[:, k].reshape((n, 1))
        row = R[k, :].reshape((1, n))
        R = np.logical_or(R, np.logical_and(col, row))

    R_ge2 = np.logical_and(R, np.logical_not(A_r1))

    depth_arr = np.array([-1 if d is None else d for d in depth], dtype=int)
    mask_depth = (depth_arr[:, None] >= 0) & (depth_arr[None, :] >= 0) & (depth_arr[:, None] < depth_arr[None, :])

    A_r3 = np.logical_and(R_ge2, mask_depth)

    A_r4 = A_r3.T.copy()

    r2_no_diag = A_r2.copy()
    np.fill_diagonal(r2_no_diag, False)
    shared_counts = r2_no_diag.astype(int) @ r2_no_diag.astype(int).T
    A_r5 = shared_counts > 0
    np.fill_diagonal(A_r5, False)

    return (
        A_r1.tolist(),
        A_r2.tolist(),
        A_r3.tolist(),
        A_r4.tolist(),
        A_r5.tolist(),
    )


csv_path = "../task0/task2.csv"
root = "1"

with open(csv_path, "r", encoding="utf-8") as f:
    text = f.read()

A_r1, A_r2, A_r3, A_r4, A_r5 = main(text, root)

if not A_r1:
    print("main вернула пустой результат — проверьте корень или формат файла.")
else:
    nums = re.findall(r"\d+", text)
    verts_all = [int(x) for x in nums] if nums else []
    if len(verts_all) % 2 == 1:
        verts_all = verts_all[:-1]
    edges = [(verts_all[i], verts_all[i+1]) for i in range(0, len(verts_all), 2)]
    vertices = sorted({v for e in edges for v in e}) if edges else []

    for name, M in zip(["r1", "r2", "r3", "r4", "r5"], (A_r1, A_r2, A_r3, A_r4, A_r5)):
        df = pd.DataFrame(M, index=vertices, columns=vertices, dtype=bool)
        print(f"\nМатрица {name}:")
        print(df.astype(int))
