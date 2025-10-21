from typing import Tuple
import math
import pandas as pd
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from task1.task import main as build_relations


def main(s: str, e: str) -> Tuple[float, float]:
    """
    Вычисление энтропии структуры графа и нормированной оценки структурной сложности.

    Параметры
    ----------
    s : str
        Текст/CSV-строка со списком рёбер. Извлекаются все натуральные
        числа и группируются по два подряд как пары (u, v).
    e : str
        Идентификатор корневого узла.

    Возвращаемое значение
    -------
    Tuple[float, float]
        (entropy, normalized_complexity), округлённые до 1 знака после запятой.
    """
    relations = build_relations(s, e)
    lines = [ln.strip() for ln in s.splitlines() if ln.strip() != ""]
    edges = [(p.strip().split(",")[0], p.strip().split(",")[1]) for p in lines]
    vertices = sorted({v for edge in edges for v in edge})
    n = len(vertices)
    m = len(relations)
    max_out = n - 1
    H_total = 0.0
    for v_idx in range(n):
        for mat in relations:
            l_ij = sum(1 for val in mat[v_idx] if val)
            if l_ij == 0:
                continue
            p = l_ij / max_out
            if p > 0.0:
                H_total += -p * math.log2(p)
    c = 1.0 / (math.e * math.log(2.0))
    H_ref = c * n * m
    h_norm = H_total / H_ref
    entropy = round(float(H_total), 1)
    normalized_complexity = round(float(h_norm), 1)
    return entropy, normalized_complexity


csv_path = "../task0/task2.csv"
root = "1"

with open(csv_path, "r", encoding="utf-8") as f:
    text = f.read()

try:
    relations = build_relations(text, root)
except Exception as exc:
    print("Ошибка при построении матриц отношений (task1):", exc)
    relations = None

if relations and isinstance(relations, tuple) and len(relations) == 5 and relations[0]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() != ""]
    edges = [tuple(p.strip() for p in ln.split(",", 1)) for ln in lines]
    vertices = sorted({v for e in edges for v in e})

try:
    entropy, norm_complexity = main(text, root)
    print(f"\nEntropy: {entropy}, Normalized complexity: {norm_complexity}")
except Exception as exc:
    print("Ошибка при вычислении энтропии/нормированной сложности:", exc)
