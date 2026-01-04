import json
import math
import numpy as np
from typing import List, Dict, Any


def main(
    json_temp_membership: str,
    json_heating_membership: str,
    json_rules: str,
    temp_value: float,
) -> float:
    """
    Функция вычисляет значение управления (уровень нагрева) по методу Мамдани:
    1) фаззификация входной величины (температура),
    2) применение правил,
    3) усечение выходных функций принадлежности по уровням активации,
    4) аккумулирование всех усечённых выходов,
    5) дефаззификация методом первого максимума.

    Параметры
    ----------
    json_temp_membership : str
        JSON-строка, описывающая функции принадлежности термов переменной "температура"
        Формат: {"температура": [ {"id":"холодно", "points":[[x0,y0],[x1,y1],...]}, ... ] }

    json_heating_membership : str
        JSON-строка, описывающая функции принадлежности термов переменной "уровень нагрева"
        Формат аналогичен

    json_rules : str
        JSON-строка, описывающая правила вида [ [input_term_id, output_term_id], ... ]

    temp_value : float
        Текущее значение температуры (градусы Цельсия)

    Возвращаемое значение
    ---------------------
    float
        Чёткое значение уровня нагрева. Если нет активированных правил или входные данные некорректны, возвращает 0.0
    """
    try:
        parsed_temp = json.loads(json_temp_membership)
        parsed_heat = json.loads(json_heating_membership)
        parsed_rules = json.loads(json_rules)
    except Exception:
        return 0.0

    def extract_terms(obj: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        seq = None
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    seq = v
                    break
        elif isinstance(obj, list):
            seq = obj
        if seq is None:
            return {}

        terms: Dict[str, Dict[str, np.ndarray]] = {}
        for item in seq:
            if not isinstance(item, dict):
                continue
            tid = item.get("id")
            pts = item.get("points")
            if tid is None or not isinstance(pts, list) or not pts:
                continue
            xs = []
            ys = []
            ok = True
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    ok = False
                    break
                try:
                    xs.append(float(p[0]))
                    ys.append(float(p[1]))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            arr = sorted(zip(xs, ys), key=lambda t: t[0])
            xs_sorted = np.array([t[0] for t in arr], dtype=float)
            ys_sorted = np.array([t[1] for t in arr], dtype=float)
            key = str(tid).strip().lower()
            terms[key] = {"x": xs_sorted, "y": ys_sorted, "orig_id": str(tid)}
        return terms

    temp_terms = extract_terms(parsed_temp)
    heat_terms = extract_terms(parsed_heat)
    if not temp_terms or not heat_terms:
        return 0.0

    rules: List[List[Any]] = []
    if isinstance(parsed_rules, list):
        for it in parsed_rules:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                in_norm = str(it[0]).strip().lower()
                out_norm = str(it[1]).strip().lower()
                rules.append([in_norm, out_norm])
    if not rules:
        return 0.0

    mu_temp: Dict[str, float] = {}
    for tid, data in temp_terms.items():
        xs = data["x"]
        ys = data["y"]
        mu = float(np.interp(float(temp_value), xs, ys))
        mu_temp[tid] = mu

    heat_keys = list(heat_terms.keys())
    out_map = {}
    n_suffix = 3
    threshold = 0.5

    def lcp_length(a: str, b: str) -> int:
        L = min(len(a), len(b))
        i = 0
        while i < L and a[i] == b[i]:
            i += 1
        return i

    for _, out_term in rules:
        if out_term in out_map:
            continue
        if out_term in heat_terms:
            out_map[out_term] = out_term
            continue
        best = None
        best_lcp = -1
        for k in heat_keys:
            lcp = lcp_length(out_term, k)
            if lcp > best_lcp:
                best_lcp = lcp
                best = k
        if best is None:
            out_map[out_term] = None
            continue
        min_len = min(len(out_term), len(best))
        cond1 = (best_lcp >= max(0, min_len - n_suffix))
        cond2 = (best_lcp >= math.ceil(min_len * threshold))
        out_map[out_term] = best if (cond1 or cond2) else None

    activations: Dict[str, float] = {}
    for in_term, out_term in rules:
        alpha = mu_temp.get(in_term, 0.0)
        if alpha <= 0.0:
            continue
        mapped_key = out_map.get(out_term)
        if mapped_key is None:
            continue
        activations[mapped_key] = max(activations.get(mapped_key, 0.0), float(alpha))

    if not activations:
        return 0.0

    segments: List[Dict[str, float]] = []
    eps_small = 1e-12

    for heat_key, alpha in activations.items():
        term = heat_terms.get(heat_key)
        if term is None:
            continue
        xs = term["x"]
        ys = term["y"]
        for i in range(len(xs) - 1):
            x0 = float(xs[i])
            x1 = float(xs[i + 1])
            y0 = float(ys[i])
            y1 = float(ys[i + 1])
            if x1 == x0:
                continue
            if y0 <= alpha + eps_small and y1 <= alpha + eps_small:
                m = (y1 - y0) / (x1 - x0)
                c = y0 - m * x0
                segments.append({"a": x0, "b": x1, "m": m, "c": c})
            elif y0 >= alpha - eps_small and y1 >= alpha - eps_small:
                segments.append({"a": x0, "b": x1, "m": 0.0, "c": float(alpha)})
            else:
                m_orig = (y1 - y0) / (x1 - x0)
                c_orig = y0 - m_orig * x0
                if abs(m_orig) < 1e-15:
                    if y0 > alpha:
                        segments.append({"a": x0, "b": x1, "m": 0.0, "c": float(alpha)})
                    else:
                        segments.append({"a": x0, "b": x1, "m": 0.0, "c": float(max(0.0, min(y0, alpha)))})
                    continue
                x_cross = (alpha - c_orig) / m_orig
                x_cross = max(min(x_cross, x1), x0)
                if y0 < alpha and y1 > alpha:
                    segments.append({"a": x0, "b": x_cross, "m": m_orig, "c": c_orig})
                    segments.append({"a": x_cross, "b": x1, "m": 0.0, "c": float(alpha)})
                else:
                    segments.append({"a": x0, "b": x_cross, "m": 0.0, "c": float(alpha)})
                    segments.append({"a": x_cross, "b": x1, "m": m_orig, "c": c_orig})

    if not segments:
        return 0.0

    cand_x = set()
    for s in segments:
        cand_x.add(s["a"])
        cand_x.add(s["b"])

    S = len(segments)
    for i in range(S):
        si = segments[i]
        ai, bi, mi, ci = si["a"], si["b"], si["m"], si["c"]
        for j in range(i + 1, S):
            sj = segments[j]
            aj, bj, mj, cj = sj["a"], sj["b"], sj["m"], sj["c"]
            if abs(mi - mj) < 1e-15:
                continue
            x_cross = (cj - ci) / (mi - mj)
            if (x_cross >= ai - 1e-12 and x_cross <= bi + 1e-12 and
                    x_cross >= aj - 1e-12 and x_cross <= bj + 1e-12):
                cand_x.add(float(x_cross))

    cand_list = sorted(cand_x)
    if not cand_list:
        return 0.0

    vals: List[float] = []
    for x in cand_list:
        y_max = -math.inf
        for s in segments:
            if x + eps_small < s["a"] or x - eps_small > s["b"]:
                continue
            yv = s["m"] * x + s["c"]
            if yv < 0:
                yv = max(yv, 0.0)
            if yv > y_max:
                y_max = yv
        if y_max < 0:
            y_max = 0.0
        vals.append(float(y_max))

    if not vals:
        return 0.0

    max_mu = max(vals)
    if max_mu <= 0.0:
        return 0.0
    tol = 1e-9
    for xi, yi in zip(cand_list, vals):
        if yi >= max_mu - tol:
            return float(xi)

    return float(cand_list[0])


temp_json_path = "функции-принадлежности-температуры.json"
heat_json_path = "функции-принадлежности-управление.json"
rules_json_path = "функция-отображения.json"

with open(temp_json_path, "r", encoding="utf-8") as f:
    json_temp = f.read()
with open(heat_json_path, "r", encoding="utf-8") as f:
    json_heat = f.read()
with open(rules_json_path, "r", encoding="utf-8") as f:
    json_rules = f.read()

if json_temp is None:
    json_temp = (
        '[{"id":"холодно","points":[[0,1],[5,0]]},'
        '{"id":"нормально","points":[[3,0],[10,1],[17,0]]},'
        '{"id":"жарко","points":[[15,0],[25,1]]}]'
    )
    print("Используется встроенный json_temp (пример).")
if json_heat is None:
    json_heat = (
        '{"температура": ['
        '{"id":"слабый","points":[[0,0],[0,1],[5,1],[8,0]]},'
        '{"id":"умеренный","points":[[5,0],[8,1],[13,1],[16,0]]},'
        '{"id":"интенсивный","points":[[13,0],[18,1],[23,1],[26,0]]}'
        ']}'
    )
    print("Используется встроенный json_heat (пример).")
if json_rules is None:
    json_rules = '[["холодно","интенсивный"],["нормально","умеренный"],["жарко","слабый"]]'
    print("Используется встроенный json_rules (пример).")

temps = [18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]

print("\nТестирование функции main для набора значений температуры")
print(f"{'temp (°C)':>10} -> {'heating (output)':>18}")
print("-" * 30)

results = []
for t in temps:
    try:
        out = main(json_temp, json_heat, json_rules, float(t))
    except Exception as exc:
        print(f"Ошибка при вызове main для temp={t}: {exc}")
        out = None
    results.append((t, out))
    print(f"{t:10.3f} -> {str(out):>18}")
