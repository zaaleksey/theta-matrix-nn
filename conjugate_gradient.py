from typing import Callable

import numpy as np
from numpy import ndarray


def conjugate(omega: ndarray,
              w: ndarray,
              get_initial_theta: Callable,
              eps: float = 10 ** (-10),
              log_step: int = 0,
              max_it: int = 100_000) -> tuple[ndarray, ndarray, list[float], int]:
    """
    Функция, реализующая метод формирования маршрутной матрицы СеМо
    с заданным вектором относительных интенсивностей потоков
    требований используя метод сопряженных градиентов.

    Алгоритм:
        Шаг 1. Определяем начальную маршрутную матрицу (инициализируем веса);
        Шаг 2. Определяем элементы маршрутной матрицы, которые не должны меняться (фиксированные элементы);
        Шаг 3. Вычисляем ошибку и её изменение на входных данных;
        Шаг 4. Вычисляем значение градиента функции потери;
        Шаг 5. Определяем изменения (дельту) весовых коэффициентов;
        Шаг 6. Производим изменение коэффициентов маршрутной матрицы;
        Шаг 7. Корректируем скорость обучения;
        Шаг 8. Проверяем условие остановки: если входной и выходной вектора omega имеют незначительную разницу (eps)
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

    assert sum(omega) == 1, "Сумма вектора omega должна быть равна единицы"

    # шаг 1.
    theta = get_initial_theta(w, omega)
    row_count, col_count = theta.shape[0], theta.shape[1]

    # шаг 2.
    fixed = set()
    for i in range(row_count):
        for j in range(col_count):
            # нулевые и единичные элементы фиксированы
            if theta[i][j] == 0 or theta[i][j] == 1:
                fixed |= {(i, j)}

    it = 0
    errors = []
    out_omega = omega.dot(theta)
    n = row_count * col_count
    while any(list(abs(x) > eps for x in out_omega - omega)) and it < max_it:
        it += 1
        k = 0

        # шаг 3.
        delta_prev = out_omega - omega
        out_omega = omega.dot(theta)
        delta = out_omega - omega
        error = sum(map(lambda x: x ** 2, delta)) / 2
        errors.append(error)

        alpha = find_alpha(theta, omega, -error)

        if k == 0:
            weight_deltas = np.zeros(row_count, col_count)
            beta = 0
            p = delta

        if k + 1 > n:
            beta = 0
            p = delta
        else:
            delta_prev = np.array(delta_prev)
            delta = np.array(delta)
            beta = (np.transpose(delta).dot(delta)) / (np.transpose(delta_prev).dot(delta_prev))
            p = delta + beta * p

        weight = np.copy(theta)
        weight_deltas = alpha * (p + weight) + weight_deltas

        theta -= weight_deltas

        if np.min(theta) < 0:
            for i in range(row_count):
                for j in range(col_count):
                    if (i, j) not in fixed:
                        theta[i][j] += (abs(np.min(theta)) * 2)

        for i in range(row_count):
            s = sum(theta[i])
            for j in range(col_count):
                theta[i][j] /= s

    return theta, out_omega, errors, it


def find_alpha(theta, inp, p):
    alpha, min_f = float("inf"), float("inf")
    for a in np.arange(0, 2, 0.05):
        w = np.copy(theta)
        w += a * p
        out = inp.dot(w)
        f = sum([(o - i) ** 2 for o, i in zip(out, inp)]) / 2
        if f < min_f:
            min_f = f
            alpha = a

    return alpha
