import json
import numpy as np
from pprint import pprint


def main(json_a: str, json_b: str) -> str:
    """
    Принимает две JSON-строки с кластерными ранжировками (каждая - JSON array,
    элементы массива либо скаляры, либо списки - кластеры).

    Возвращает JSON-строку:
    - при успехе: сериализация списка пар [x_i, x_j] — ядро противоречий;
    - при ошибке: {"error": "<описание>"}.

    Правило для матрицы Y: y[i][j] = 1 <=> позиция элемента i >= позиции элемента j.
    """

    def _parse_obj(obj):
        clusters = []
        seen = set()
        order = []
        for idx, item in enumerate(obj):
            if isinstance(item, list):
                if len(item) == 0:
                    raise ValueError(f"пустой кластер на месте {idx}.")
                for val in item:
                    if val in seen:
                        raise ValueError(f"повторяющийся элемент '{val}' в ранжировке")
                    seen.add(val)
                    order.append(val)
                clusters.append(list(item))
            else:
                val = item
                if val in seen:
                    raise ValueError(f"повторяющийся элемент '{val}' в ранжировке")
                seen.add(val)
                clusters.append([val])
                order.append(val)
        return clusters, order

    try:
        obj_a = json.loads(json_a)
    except Exception as exc:
        return json.dumps({"ошибка": f"недопустимый JSON для первой ранжировки: {exc}"}, ensure_ascii=False)
    try:
        obj_b = json.loads(json_b)
    except Exception as exc:
        return json.dumps({"ошибка": f"недопустимый JSON для второй ранжировки: {exc}"}, ensure_ascii=False)

    try:
        clusters_a, order_a = _parse_obj(obj_a)
    except ValueError as exc:
        return json.dumps({"ошибка": f"ошибка в первой ранжировке: {exc}"}, ensure_ascii=False)
    try:
        clusters_b, order_b = _parse_obj(obj_b)
    except ValueError as exc:
        return json.dumps({"ошибка": f"ошибка во второй ранжировке: {exc}"}, ensure_ascii=False)

    set_a = set(order_a)
    set_b = set(order_b)
    if set_a != set_b:
        return json.dumps({"ошибка": "ранжировки должны содержать одинаковые объекты"}, ensure_ascii=False)

    order = order_a[:]
    n = len(order)
    if n == 0:
        return json.dumps({"ошибка": "пустая ранжировка"}, ensure_ascii=False)
    pos_a = {}
    for ci, cluster in enumerate(clusters_a):
        for el in cluster:
            pos_a[el] = ci
    pos_b = {}
    for ci, cluster in enumerate(clusters_b):
        for el in cluster:
            pos_b[el] = ci
    pos_arr_a = np.array([pos_a[e] for e in order], dtype=int)
    pos_arr_b = np.array([pos_b[e] for e in order], dtype=int)
    YA = pos_arr_a[:, None] >= pos_arr_a[None, :]
    YB = pos_arr_b[:, None] >= pos_arr_b[None, :]
    YAB = np.logical_and(YA, YB)
    YT_AB = np.logical_and(YA.T, YB.T)
    mask = np.logical_and(~YAB, ~YT_AB)
    diag = np.eye(n, dtype=bool)
    mask = np.logical_and(mask, ~diag)
    contradictions = [[order[i], order[j]] for i, j in np.argwhere(np.triu(mask, k=1))]

    return json.dumps(contradictions, ensure_ascii=False)


json_a_path = "Ранжировка  A.json"
json_b_path = "Ранжировка  B.json"
expected_path = "Ядро противоречий AB.json"

try:
    with open(json_a_path, "r", encoding="utf-8") as f:
        json_a = f.read()
except Exception as exc:
    json_a = '[1,[2,3],4,[5,6,7],8,9,10]'

try:
    with open(json_b_path, "r", encoding="utf-8") as f:
        json_b = f.read()
except Exception as exc:
    json_b = '[[1,2],[3,4,5,],6,7,9,[8,10]]'

try:
    with open(expected_path, "r", encoding="utf-8") as f:
        expected_str = f.read()
except Exception as exc:
    expected_str = '[[8,9]]'

print("\n--- inputs ---")
print("json_a:", json_a)
print("json_b:", json_b)
print("expected:", expected_str)

try:
    result_str = main(json_a, json_b)
except Exception as exc:
    raise

print("\n--- main returned ---")
print(result_str)

result = json.loads(result_str)
expected = json.loads(expected_str)

if isinstance(result, dict) and "error" in result:
    print("\nmain() вернула объект ошибки:")
    pprint(result)
    raise SystemExit(2)


def normalize_pairs(lst):
    try:
        normalized = []
        for pair in lst:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return None
            a, b = pair
            normalized.append((str(a), str(b)))
        return set(normalized)
    except Exception:
        return None


norm_res = normalize_pairs(result)
norm_exp = normalize_pairs(expected)

print("\n--- сomparison ---")
if norm_res is None:
    print("формат результата неожиданный")
    pprint(result)
    raise SystemExit(2)
if norm_exp is None:
    print("формат expected неожиданный")
    pprint(expected)
    raise SystemExit(2)

if norm_res == norm_exp:
    print("результат совпадает с ожидаемым")
    pprint(sorted(norm_res))
    raise SystemExit(0)
else:
    only_in_res = norm_res - norm_exp
    only_in_exp = norm_exp - norm_res
    print("результат отличается от ожидаемого.")
    print("только в результате:")
    pprint(sorted(only_in_res))
    print("только в expected:")
    pprint(sorted(only_in_exp))
    raise SystemExit(1)
