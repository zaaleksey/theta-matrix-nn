from typing import Callable

import numpy as np
from numpy import ndarray


def gradient_descent(omega: ndarray,
                     w: ndarray,
                     get_initial_theta: Callable,
                     eps: float = 10 ** (-10),
                     log_step: int = 0,
                     max_it: int = 2_000) -> tuple[ndarray, ndarray, list[float], int]:
    """
    Функция, реализующая метод формирования маршрутной матрицы СеМо
    с заданным вектором относительных интенсивностей потоков
    требований используя метод градиентного спуска.

    Алгоритм:
        Шаг 1. Определяем начальную маршрутную матрицу (инициализируем веса);
        Шаг 2. Определяем элементы маршрутной матрицы, которые не должны меняться (фиксированные элементы);
        Шаг 3. Вычисляем ошибку и определяем изменения (дельту) весовых коэффициентов;
        Шаг 4. Производим изменение коэффициентов маршрутной матрицы;
        Шаг 5. После предыдущего шага необходимо нормализовать вероятностные коэффициенты маршрутной матрицы;
        Шаг 6. Проверяем условие остановки: если входной и выходной вектора omega имеют незначительную разницу (eps)
        или достигнуто максимальное число итераций, то завершаем алгоритм, иначе переходим к шагу 3.

    Parameters:
        omega: вектор относительных интенсивностей потоков
        w: матрица смежности, определяющая топологию сети обслуживания
        get_initial_theta: функция для определения начальной маршрутной матрицы
        eps: точность определения вектора omega уравнением omega = omega * theta
        log_step: параметр, отвечающая за шаг логирование в процессе работы метода,
        если параметр равен 0 - логирование не производится, по умолчанию имеет значение 0
        max_it: максимальное число итераций, если решение не будет найдено, будет выведена приближенная матрица

    Returns:
        Полученная маршрутная матрица;
        Соответствующий вектор интенсивностей потоков;
        Массив, содержащий значения ошибок в процессе формирования матрицы;
        Число пройденных итераций.

    """

    # assert sum(omega) == 1, "The sum of the omega vector must be equal to one."

    # step 1.
    theta = get_initial_theta(w, omega)
    # print("Омега: ", omega)
    # print("Начальная маршрутная матрица theta имеет следующий вид:\n", theta, "\n")

    row_count, col_count = theta.shape[0], theta.shape[1]

    # step 2.
    fixed = set()
    for i in range(row_count):
        for j in range(col_count):
            # нулевые и единичные элементы фиксированы
            if theta[i][j] == 0 or theta[i][j] == 1:
                fixed |= {(i, j)}

    it = 0
    errors = []
    out_omega = omega.dot(theta)

    while any(list(abs(x) > eps for x in out_omega - omega)) and it < max_it:
        it += 1

        # step 3.
        out_omega = omega.dot(theta)
        delta = np.array(out_omega - omega)
        error = sum([float(d) ** 2 for d in delta]) / 2
        errors.append(error)
        weight_deltas = np.outer(omega, delta)

        for i in range(row_count):
            for j in range(col_count):
                if (i, j) in fixed:
                    weight_deltas[i][j] = 0

        # step 4.
        theta -= weight_deltas

        # step 5.
        if np.min(theta) < 0:
            for i in range(row_count):
                for j in range(col_count):
                    if (i, j) not in fixed:
                        theta[i][j] += (abs(np.min(theta)) * 2)

        for i in range(row_count):
            s = sum(theta[i])
            for j in range(col_count):
                theta[i][j] /= s

        if log_step and it % log_step == 0:
            print(f"Итерация {it}:")
            print("Полученная омега:", out_omega)
            print("Theta\n", theta, "\n")

    return theta, out_omega, errors, it
