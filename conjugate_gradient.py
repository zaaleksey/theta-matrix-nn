from typing import Callable

import numpy as np
from numpy import ndarray


def conjugate(omega: ndarray,
              w: ndarray,
              get_initial_theta: Callable,
              eps: float = 10 ** (-10),
              log_step: int = 0,
              max_it: int = 2_000) -> tuple[ndarray, ndarray, list[float], int]:
    """
    Функция, реализующая метод формирования маршрутной матрицы СеМо
    с заданным вектором относительных интенсивностей потоков
    требований используя метод сопряженных градиентов.

    Алгоритм:
        Шаг 1. Определяем начальную маршрутную матрицу (инициализируем веса);
        Шаг 2. Определяем элементы маршрутной матрицы, которые не должны меняться (фиксированные элементы);
        Шаг 3. Вычисляем ошибку и её изменение на входных данных;
        Шаг 4. Определяем коэффициент скорости обучения;
        Шаг 5. Определяем изменения (дельту) весовых коэффициентов;
        Шаг 6. Производим изменение коэффициентов маршрутной матрицы;
        Шаг 7. После предыдущего шага необходимо нормализовать вероятностные коэффициенты маршрутной матрицы;
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

    # assert sum(omega) == 1, "The sum of the omega vector must be equal to one."

    # step 1.
    theta = get_initial_theta(w, omega)
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
    n = row_count * col_count
    while any(list(abs(x) > eps for x in out_omega - omega)) and it < max_it:
        it += 1
        k = 0

        # step 3.
        delta_prev = np.array(out_omega - omega)
        out_omega = omega.dot(theta)
        delta = np.array(out_omega - omega)
        error = sum(map(lambda x: x ** 2, delta)) / 2
        errors.append(error)

        # step 4.
        alpha = find_alpha(theta, omega, error)
        p = delta

        weight_deltas = np.outer(omega, delta)
        beta = (np.transpose(delta).dot(delta)) / (np.transpose(delta_prev).dot(delta_prev))
        p = delta + beta * p

        # step 5.
        weight = np.copy(theta)
        weight_deltas = alpha * (p + weight) + weight_deltas

        for i in range(row_count):
            for j in range(col_count):
                if (i, j) in fixed:
                    weight_deltas[i][j] = 0

        # step 6.
        theta -= weight_deltas

        # step 7.
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
    for a in np.arange(0, 1, 0.01):
        w = np.copy(theta)
        w += a * p
        out = inp.dot(w)
        f = sum([(o - i) ** 2 for o, i in zip(out, inp)]) / 2
        if f < min_f:
            min_f = f
            alpha = a

    return alpha
