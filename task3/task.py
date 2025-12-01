import json
import numpy as np
from pprint import pprint


def main(json_a: str, json_b: str):
    """
    Принимает две JSON-строки с кластерными ранжировками (каждая - JSON array,
    элементы массива либо скаляры, либо списки - кластеры).

    Возвращает кортеж из двух JSON-строк:
      ( contradictions_json, consensus_ranking_json )

    - contradictions_json: сериализация списка неупорядоченных пар [[x_i,x_j], ...] — ядро противоречий;
    - consensus_ranking_json: сериализация согласованной кластерной ранжировки (список кластеров, каждый кластер — список элементов).

    В случае ошибки возвращает два одинаковых JSON-строки с объектом {"ошибка": "<описание>"}.
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
                    if isinstance(val, list):
                        raise ValueError(f"вложенные кластеры не поддерживаются (позиция {idx}).")
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
        err = json.dumps({"ошибка": f"недопустимый JSON для первой ранжировки: {exc}"}, ensure_ascii=False)
        return err, err
    try:
        obj_b = json.loads(json_b)
    except Exception as exc:
        err = json.dumps({"ошибка": f"недопустимый JSON для второй ранжировки: {exc}"}, ensure_ascii=False)
        return err, err

    try:
        clusters_a, order_a = _parse_obj(obj_a)
    except ValueError as exc:
        err = json.dumps({"ошибка": f"ошибка в первой ранжировке: {exc}"}, ensure_ascii=False)
        return err, err
    try:
        clusters_b, order_b = _parse_obj(obj_b)
    except ValueError as exc:
        err = json.dumps({"ошибка": f"ошибка во второй ранжировке: {exc}"}, ensure_ascii=False)
        return err, err

    set_a = set(order_a)
    set_b = set(order_b)
    if set_a != set_b:
        err = json.dumps({"ошибка": "ранжировки должны содержать одинаковые объекты"}, ensure_ascii=False)
        return err, err

    order = order_a[:]
    n = len(order)
    if n == 0:
        err = json.dumps({"ошибка": "пустая ранжировка"}, ensure_ascii=False)
        return err, err
    pos_a = {}
    for ci, cluster in enumerate(clusters_a):
        for el in cluster:
            pos_a[el] = ci
    pos_b = {}
    for ci, cluster in enumerate(clusters_b):
        for el in cluster:
            pos_b[el] = ci
    try:
        pos_arr_a = np.array([pos_a[e] for e in order], dtype=int)
        pos_arr_b = np.array([pos_b[e] for e in order], dtype=int)
    except KeyError as exc:
        err = json.dumps({"ошибка": f"элемент отсутствует в отображении позиций: {exc}"}, ensure_ascii=False)
        return err, err

    YA = pos_arr_a[:, None] >= pos_arr_a[None, :]
    YB = pos_arr_b[:, None] >= pos_arr_b[None, :]
    YAB = np.logical_and(YA, YB)
    YT_AB = np.logical_and(YA.T, YB.T)
    mask = np.logical_and(~YAB, ~YT_AB)
    diag = np.eye(n, dtype=bool)
    mask = np.logical_and(mask, ~diag)
    pairs_idx = np.argwhere(np.triu(mask, k=1))
    contradictions = [[order[int(i)], order[int(j)]] for i, j in pairs_idx]
    contradictions.sort(key=lambda p: (str(p[0]), str(p[1])))

    C = np.logical_and(YA, YB).copy()

    for i, j in pairs_idx:
        C[int(i), int(j)] = True
        C[int(j), int(i)] = True

    E = np.logical_and(C, C.T)

    E_star = E.copy()
    for k in range(n):
        col = E_star[:, k].reshape((n, 1))
        row = E_star[k, :].reshape((1, n))
        E_star = np.logical_or(E_star, np.logical_and(col, row))
    visited = np.zeros(n, dtype=bool)
    clusters_idx = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            neigh = np.nonzero(E_star[u])[0]
            for v in neigh:
                if not visited[int(v)]:
                    visited[int(v)] = True
                    stack.append(int(v))
        comp.sort()
        clusters_idx.append(comp)

    avg_positions = []
    for comp in clusters_idx:
        pos_mean_a = np.mean(pos_arr_a[comp]) if len(comp) > 0 else 0.0
        pos_mean_b = np.mean(pos_arr_b[comp]) if len(comp) > 0 else 0.0
        avg = 0.5 * (pos_mean_a + pos_mean_b)
        min_idx = int(min(comp)) if comp else 0
        avg_positions.append((avg, min_idx))

    order_clusters = sorted(range(len(clusters_idx)), key=lambda k: (avg_positions[k][0], avg_positions[k][1]))

    consensus_clusters = []
    for idx in order_clusters:
        comp = clusters_idx[idx]
        cluster_elems = [order[int(i)] for i in comp]
        consensus_clusters.append(cluster_elems)

    try:
        contradictions_json = json.dumps(contradictions, ensure_ascii=False)
        consensus_json = json.dumps(consensus_clusters, ensure_ascii=False)
    except Exception as exc:
        err = json.dumps({"ошибка": f"ошибка сериализации результата: {exc}"}, ensure_ascii=False)
        return err, err

    return contradictions_json, consensus_json


json_a_path = "Ранжировка  A.json"
json_b_path = "Ранжировка  B.json"
expected_contr_path = "Ядро противоречий AB.json"
expected_consensus_path = "Согласованная кластерная ранжировка AB.json"

# Чтение входов (если файла нет — используем встроенные строки)
try:
    with open(json_a_path, "r", encoding="utf-8") as f:
        json_a = f.read()
except Exception as exc:
    print(f"Не удалось прочитать {json_a_path}: {exc}")
    json_a = '[1,[2,3],4,[5,6,7],8,9,10]'

try:
    with open(json_b_path, "r", encoding="utf-8") as f:
        json_b = f.read()
except Exception as exc:
    print(f"Не удалось прочитать {json_b_path}: {exc}")
    json_b = '[[1,2],[3,4,5,],6,7,9,[8,10]]'

# Ожидаемое ядро противоречий
try:
    with open(expected_contr_path, "r", encoding="utf-8") as f:
        expected_contr_str = f.read()
except Exception as exc:
    print(f"Не удалось прочитать {expected_contr_path}: {exc}")
    expected_contr_str = '[[8,9]]'

# Ожидаемая согласованная ранжировка (если есть)
expected_consensus_str = None
try:
    with open(expected_consensus_path, "r", encoding="utf-8") as f:
        expected_consensus_str = f.read()
except Exception as exc:
    print(f"Не удалось прочитать {expected_consensus_path}: {exc}")
    # оставляем expected_consensus_str = None, тогда проверка консенсуса будет пропущена

print("\n--- inputs ---")
print("json_a:", json_a)
print("json_b:", json_b)
print("expected (contradictions):", expected_contr_str)
if expected_consensus_str is not None:
    print("expected (consensus):", expected_consensus_str)
else:
    print("expected (consensus): <файл не найден — проверка консенсуса пропущена>")

# Вызов main — ожидаем два результата (contradictions_json, consensus_json)
try:
    res = main(json_a, json_b)
except Exception as exc:
    print("\nmain вызвала исключение:")
    raise

if isinstance(res, (list, tuple)) and len(res) == 2:
    contr_str, consensus_str = res[0], res[1]
else:
    contr_str = res
    consensus_str = None

print("\n--- main returned (raw) ---")
print("contradictions:", contr_str)
print("consensus:", consensus_str)

result_contr = json.loads(contr_str)
result_consensus = json.loads(consensus_str) if consensus_str is not None else None
expected_contr = json.loads(expected_contr_str)
expected_consensus = json.loads(expected_consensus_str) if expected_consensus_str is not None else None

if isinstance(result_contr, dict) and "ошибка" in result_contr:
    print("\nmain() вернула объект ошибки для ядра противоречий:")
    pprint(result_contr)
    raise SystemExit(2)
if result_consensus is not None and isinstance(result_consensus, dict) and "ошибка" in result_consensus:
    print("\nmain() вернула объект ошибки для согласованной ранжировки:")
    pprint(result_consensus)
    raise SystemExit(2)

if result_contr is None:
    print("\nНевозможно распарсить вывод main() (ядро противоречий) как JSON — тест провален.")
    raise SystemExit(2)
if expected_contr is None:
    print("\nНевозможно распарсить expected (ядро противоречий) — проверьте файл.")
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


norm_res = normalize_pairs(result_contr)
norm_exp = normalize_pairs(expected_contr)

print("\n--- Comparison (contradictions) ---")
if norm_res is None:
    print("Формат результата (ядро) неожиданный:")
    pprint(result_contr)
    raise SystemExit(2)
if norm_exp is None:
    print("Формат expected (ядро) неожиданный:")
    pprint(expected_contr)
    raise SystemExit(2)

if norm_res == norm_exp:
    print("PASS: ядро противоречий совпадает с ожидаемым")
    pprint(sorted(norm_res))
    contr_ok = True
else:
    only_in_res = norm_res - norm_exp
    only_in_exp = norm_exp - norm_res
    print("FAIL: ядро противоречий отличается от ожидаемого.")
    print("Только в результате:")
    pprint(sorted(only_in_res))
    print("Только в expected:")
    pprint(sorted(only_in_exp))
    contr_ok = False

consensus_ok = None
if expected_consensus is not None:
    if result_consensus is None:
        print("\nОжидался файл с согласованной ранжировкой, но main не вернула его.")
        consensus_ok = False
    else:
        def normalize_clusters(clusters):
            """
            Преобразует представление кластерной ранжировки в множество frozenset строк-элементов.
            Поддерживает форматы:
            - [1,2,[3,4],5]  (скалярные элементы трактуются как одиночные кластеры)
            - [[1],[2],[3,4]] (все кластеры как списки)
            Возвращает None при некорректном формате.
            """
            try:
                normalized = set()
                for cl in clusters:
                    if not isinstance(cl, (list, tuple)):
                        elems = [cl]
                    else:
                        elems = list(cl)
                    for x in elems:
                        if isinstance(x, (list, tuple)):
                            return None
                    normalized.add(frozenset(str(x) for x in elems))
                return normalized
            except Exception:
                return None

        norm_cons_res = normalize_clusters(result_consensus)
        norm_cons_exp = normalize_clusters(expected_consensus)

        print("\n--- Comparison (consensus clustering) ---")
        if norm_cons_res is None:
            print("Формат результата (consensus) неожиданный:")
            pprint(result_consensus)
            raise SystemExit(2)
        if norm_cons_exp is None:
            print("Формат expected (consensus) неожиданный:")
            pprint(expected_consensus)
            raise SystemExit(2)

        if norm_cons_res == norm_cons_exp:
            try:
                parsed_a_flat = []
                parsed_a = json.loads(json_a)
                for item in parsed_a:
                    if isinstance(item, (list, tuple)):
                        for v in item:
                            parsed_a_flat.append(str(v))
                    else:
                        parsed_a_flat.append(str(item))
                order_map = {v: i for i, v in enumerate(parsed_a_flat)}
            except Exception:
                order_map = {}

            def ordered_clusters_from_set(cluster_set):
                lst = [sorted(list(s), key=lambda x: order_map.get(x, 10**9)) for s in cluster_set]
                lst.sort(key=lambda cl: order_map.get(cl[0], 10**9))
                return lst

            print("PASS: согласованная ранжировка совпадает с ожидаемым")
            ordered_res_display = ordered_clusters_from_set(norm_cons_res)
            pprint(ordered_res_display)
            consensus_ok = True
        else:
            only_in_res = norm_cons_res - norm_cons_exp
            only_in_exp = norm_cons_exp - norm_cons_res
            print("FAIL: согласованная ранжировка отличается от ожидаемой.")
            print("Только в результате (кластеры):")
            pprint([sorted(list(s)) for s in sorted(only_in_res, key=lambda x: sorted(list(x)))])
            print("Только в expected (кластеры):")
            pprint([sorted(list(s)) for s in sorted(only_in_exp, key=lambda x: sorted(list(x)))])
            consensus_ok = False

if consensus_ok is None:
    if contr_ok:
        raise SystemExit(0)
    else:
        raise SystemExit(1)
else:
    if contr_ok and consensus_ok:
        raise SystemExit(0)
    else:
        raise SystemExit(1)
