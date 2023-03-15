from typing import Callable

import numpy as np
from numpy import ndarray
from progress.bar import Bar


def find_routing_matrix(omega: ndarray,
                        w: ndarray,
                        get_initial_theta: Callable,
                        eps: float = 10 ** (-10),
                        log_step: int = 0,
                        max_it: int = 10_000_000) -> tuple[ndarray, ndarray]:
    """
    Функция реализующая метод формирования маршрутной матрицы СеМо
    с заданным вектором относительных интенсивностей потоков
    требований.

    Parameters:
        omega: вектор относительных интенсивностей потоков
        w: матрица смежности, определяющая топологию сети о
        бслуживания
        get_initial_theta: функция, которая должна вернуть начальную матрицу (имеет два параметра)
        eps: точность при определение вектора omega уравнением
        omega = omega * theta,
        по умолчанию имеет значение 10 ** (-10)
        log_step: параметр, отвечающая за шаг логирование в
        процессе работы метода,
        если параметр равен 0 - логирование не производится,
        по умолчанию имеет значение 0
        max_it: максимальное число воозмможных итераций, если решение не будет найдено, будет вывдена приближенныя матрица

    Returns:
        Маршрутную матрицу и соответствующий вектор интенсивностей
        потоков.

    """
    bar = Bar("Progress", max=max_it)

    # шаг 1: определяем начальную маршрутную матрицу
    theta = get_initial_theta(w, omega)
    print("Омега: ", omega)
    print("Начальная маршрутная матрица theta имеет следующиий вид:\n", theta, "\n")

    row_count, col_count = theta.shape[0], theta.shape[1]

    # шаг 2: определяем элементы маршрутной матрицы, которые не должны меняться при формировании
    fixed = set()
    for i in range(row_count):
        for j in range(col_count):
            if theta[i][j] == 0 or theta[i][j] == 1:  # нулевые и единичные элементы фиксированы
                fixed |= {(i, j)}

    it = 0
    out_omega = omega.dot(theta)
    # итерационно, пока не будет достигнута минимальная погрешность, производится изменение весовых коэффициентов
    while any(list(abs(x) > eps for x in out_omega - omega)) and it < max_it:
        it += 1
        bar.next()

        out_omega = omega.dot(theta)
        error = np.zeros(col_count)
        delta = np.zeros(col_count)
        for i in range(col_count):
            delta[i] = (out_omega[i] - omega[i])
            error[i] = delta[i] ** 2

        weight_deltas = np.outer(omega, delta)
        for i in range(row_count):
            for j in range(col_count):
                if (i, j) in fixed:
                    weight_deltas[i][j] = 0

        # шаг 3: корректируем маршрутную матрицу
        theta -= weight_deltas

        # шаг 4: нормализуем строки маршрутной матрицы
        for i in range(row_count):
            s = sum(theta[i])
            for j in range(col_count):
                theta[i][j] /= s

        if np.min(theta) < 0:
            for i in range(row_count):
                for j in range(col_count):
                    if (i, j) not in fixed:
                        theta[i][j] += (abs(np.min(theta)) * 2)

        # шаг 4: нормализуем строки маршрутной матрицы
        for i in range(row_count):
            s = sum(theta[i])
            for j in range(col_count):
                theta[i][j] /= s

        if log_step and it % log_step == 0:
            print(f"Итерация {it}:")
            print("Полученная омега:", out_omega)
            print("Delta:", delta)
            print("Error:", error)
            print("Theta\n", theta, "\n")

    bar.finish()

    print(f"Итерация {it} (последняя):")
    print("Полученная омега:", out_omega)
    print("Разница омег:", delta)
    print("Ошибка:", error)
    print("Theta\n", theta, "\n")
    return theta, out_omega


def get_uniform_initial_theta(w: ndarray, omega: ndarray) -> ndarray:
    row_count, col_count = w.shape[0], w.shape[1]
    theta = np.zeros((row_count, col_count))

    non_zero_elements = [0] * row_count
    for i, row in enumerate(w):
        non_zero_elements[i] = sum(row)

    for i in range(row_count):
        for j in range(col_count):
            if w[i][j] == 1:
                theta[i][j] = 1 / non_zero_elements[i]  # равномерное распределние элементов матрицы

    return theta
