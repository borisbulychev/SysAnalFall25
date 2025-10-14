import pandas as pd
import re


def main(input_str: str) -> list[list[int]]:
    """
    Парсинг строки со списком рёбер и построение матрицы смежности графа.

    Параметры
    ----------
    input_str : str
        Строка, введённая в консоль (может содержать переводы строк).

    Возвращаемое значение
    ---------------------
    list[list[int]]
        Квадратичная матрица смежности (список списков из 0/1).
        Пустой список при отсутствии корректных рёбер.
    """

    if not input_str or input_str.strip() == "":
        return []

    nums = re.findall(r'\d+', input_str)
    if not nums:
        return []

    verts_all = [int(x) for x in nums]

    if len(verts_all) % 2 == 1:
        verts_all = verts_all[:-1]

    edges: list[tuple[int, int]] = []
    for i in range(0, len(verts_all), 2):
        v_j = verts_all[i]
        v_k = verts_all[i + 1]
        edges.append((v_j, v_k))

    if not edges:
        return []

    vertices = sorted({v for e in edges for v in e})

    idx = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    G: list[list[int]] = [[0] * n for _ in range(n)]

    for e_j, e_k in edges:
        i = idx[e_j]
        j = idx[e_k]
        G[i][j] = 1
        G[j][i] = 1

    return G


csv_path = 'task0/task2.csv'

with open(csv_path, 'r', encoding='utf-8') as f:
    text = f.read()

G = main(text)

nums = re.findall(r'\d+', text)
verts_all = [int(x) for x in nums] if nums else []
if len(verts_all) % 2 == 1:
    verts_all = verts_all[:-1]
edges = [(verts_all[i], verts_all[i+1]) for i in range(0, len(verts_all), 2)]
vertices = sorted({v for e in edges for v in e}) if edges else []

df = pd.DataFrame(G, index=vertices, columns=vertices)
print(df)
