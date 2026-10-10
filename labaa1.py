import numpy as np
import itertools
import matplotlib.pyplot as plt

c = np.array([5, -2])

A_ub = np.array([
    [8, 2],
    [5, 6]
])
b_ub = np.array([12, 8])

num_vars = 2

def solve_by_vertex_enumeration(c, A_ub, b_ub):
    print("\nперебор вершин")
    num_constraints = len(b_ub)
    
    lines = []
    for i in range(num_constraints):
        lines.append((A_ub[i][0], A_ub[i][1], b_ub[i]))
    
    lines.append((1, 0, 0))
    lines.append((0, 1, 0))
    
    vertices = []
    
    for (l1, l2) in itertools.combinations(lines, 2):
        mat = np.array([[l1[0], l1[1]], [l2[0], l2[1]]])
        vec = np.array([l1[2], l2[2]])
        
        try:
            if np.linalg.det(mat) != 0:
                point = np.linalg.solve(mat, vec)
                vertices.append(point)
        except np.linalg.LinAlgError:
            continue

    feasible_vertices = []
    print("Найденные вершины и проверка допустимости:")
    for v in vertices:
        if v[0] >= -1e-9 and v[1] >= -1e-9:
            if np.all(np.dot(A_ub, v) <= b_ub + 1e-9):
                feasible_vertices.append(v)
                print(f"  Точка ({v[0]:.2f}, {v[1]:.2f}) - Допустима")
    
    best_val = -np.inf
    best_v = None
    
    print("\nВычисление целевой функции в допустимых вершинах:")
    for v in feasible_vertices:
        val = np.dot(c, v)
        print(f"  z({v[0]:.2f}, {v[1]:.2f}) = {val:.2f}")
        if val > best_val:
            best_val = val
            best_v = v
            
    print(f"\nОПТИМАЛЬНОЕ РЕШЕНИЕ (Перебор):")
    print(f"x1 = {best_v[0]:.4f}, x2 = {best_v[1]:.4f}")
    print(f"Max Z = {best_val:.4f}")
    
    plt.figure(figsize=(8, 6))
    
    x1_vals = np.linspace(0, 3, 400)
    
    y1 = (b_ub[0] - A_ub[0][0] * x1_vals) / A_ub[0][1]
    y2 = (b_ub[1] - A_ub[1][0] * x1_vals) / A_ub[1][1]
    
    plt.plot(x1_vals, y1, label=f'{A_ub[0][0]}x1 + {A_ub[0][1]}x2 <= {b_ub[0]}', color='blue')
    plt.plot(x1_vals, y2, label=f'{A_ub[1][0]}x1 + {A_ub[1][1]}x2 <= {b_ub[1]}', color='green')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    plt.fill_between(x1_vals, 0, np.minimum(y1, y2), where=(np.minimum(y1, y2) >= 0), color='gray', alpha=0.3, label='Допустимая область')
    
    for v in feasible_vertices:
        plt.plot(v[0], v[1], 'ro')
        plt.text(v[0] + 0.05, v[1] + 0.05, f'({v[0]:.2f}, {v[1]:.2f})', fontsize=9)
    
    plt.plot(best_v[0], best_v[1], 'r*', markersize=15, label='Оптимум')
    
    plt.xlim(0, 1.6)
    plt.ylim(0, 1.4)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Допустимая область и вершины (Перебор)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return best_v, best_val

def solve_simplex(c, A_ub, b_ub):
    print("\nсимплекс-метод")
    
    num_constraints = len(b_ub)
    num_vars = len(c)
    
    A = np.hstack([A_ub, np.eye(num_constraints)])
    c_full = np.concatenate([c, np.zeros(num_constraints)])
    
    basis_indices = list(range(num_vars, num_vars + num_constraints))
    
    num_total_vars = len(c_full)
    tableau = np.zeros((num_constraints + 1, num_total_vars + 1))
    
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b_ub
    tableau[-1, :num_total_vars] = -c_full
    
    print("Начальная симплекс-таблица:")
    print(tableau)
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\nИтерация {iteration}")
        
        last_row = tableau[-1, :-1]
        if np.all(last_row >= -1e-9):
            print("Оптимальное решение найдено.")
            break
            
        pivot_col = np.argmin(last_row)
        print(f"Ведущий столбец: {pivot_col}")
        
        ratios = []
        for i in range(num_constraints):
            if tableau[i, pivot_col] > 1e-9:
                ratios.append(tableau[i, -1] / tableau[i, pivot_col])
            else:
                ratios.append(np.inf)
        
        if all(r == np.inf for r in ratios):
            print("Задача не ограничена!")
            return None, None
            
        pivot_row = np.argmin(ratios)
        print(f"Ведущая строка: {pivot_row}")
        
        basis_indices[pivot_row] = pivot_col
        
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        
        for i in range(num_constraints + 1):
            if i != pivot_row:
                factor = tableau[i, pivot_col]
                tableau[i, :] -= factor * tableau[pivot_row, :]
                
        print("Текущая таблица:")
        print(np.round(tableau, 4))

    solution = np.zeros(num_total_vars)
    for i, var_idx in enumerate(basis_indices):
        solution[var_idx] = tableau[i, -1]
        
    real_z = np.dot(c, solution[:num_vars])
    
    print(f"\nОПТИМАЛЬНОЕ РЕШЕНИЕ (Симплекс):")
    print(f"x1 = {solution[0]:.4f}")
    print(f"x2 = {solution[1]:.4f}")
    print(f"Max Z = {real_z:.4f}")
    
    return solution[:num_vars], real_z

v_enum, z_enum = solve_by_vertex_enumeration(c, A_ub, b_ub)
v_simp, z_simp = solve_simplex(c, A_ub, b_ub)

print("\n--- сравнение результатов ---")
print(f"Перебор вершин: x={v_enum}, Z={z_enum}")
print(f"Симплекс-метод: x={v_simp}, Z={z_simp}")
if np.allclose(v_enum, v_simp) and np.isclose(z_enum, z_simp):
    print("Результаты совпадают.")
else:
    print("Результаты различаются.")