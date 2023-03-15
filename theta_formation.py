import numpy as np


def find_routing_matrix(omega, W, eps=10 ** (-10), log_step=0):
    """
    Функция реализующая метод формирования маршрутной матрицы СеМо
    с заданным вектором относительных интенсивностей потоков
    требований.

    Parameters:
        omega: вектор относительных интенсивностей потоков
        W: матрица смежности, определяющая топологию сети о
        бслуживания
        eps: точность при определение вектора omega уравнением
        omega = omega * Theta,
        по умолчанию имеет значение 10 ** (-10)
        log_step: параметр, отвечающая за шаг логирование в
        процессе работы метода,
        если параметр равен 0 - логирование не производится,
        по умолчанию имеет значение 0

    Returns:
        Маршрутную матрицу и соответствующий вектор интенсивностей
        потоков.

    """
    # определяем начальную маршрутную матрицу1
    row_count, col_count = W.shape[0], W.shape[1]
    Theta = np.zeros((row_count, col_count))

    non_zero_elements_count = [0] * row_count
    for i, row in enumerate(W):
        non_zero_elements_count[i] = sum(row)

    for i in range(row_count):
        for j in range(col_count):
            if W[i][j] == 1:
                Theta[i][j] = 1 / non_zero_elements_count[i]

    # определяем компоненты маршрутной матрицы,
    # которые не должны изменяться
    fixed = set()
    for i in range(Theta.shape[0]):
        for j in range(Theta.shape[1]):
            # нулевые и единичные компоненты должны быть фиксированы
            if Theta[i][j] == 0 or Theta[i][j] == 1:
                fixed |= {(i, j)}

    it = 0
    out_omega = omega.dot(Theta)
    while any(list(abs(x) > eps for x in out_omega - omega)):
        it += 1
        out_omega = omega.dot(Theta)
        error = np.zeros(len(omega))
        delta = np.zeros(len(omega))
        for i in range(len(omega)):
            delta[i] = out_omega[i] - omega[i]
            error[i] = delta[i] ** 2

        weight_deltas = np.outer(omega, delta)
        for i in range(len(weight_deltas)):
            for j in range(len(weight_deltas)):
                if (i, j) in fixed:
                    weight_deltas[i][j] = 0

        # корректируем маршрутную матрицу
        Theta -= weight_deltas

        # нормализуем строки маршрутной матрицы
        for i in range(Theta.shape[0]):
            s = sum(Theta[i])
            for j in range(Theta.shape[1]):
                Theta[i][j] /= s

        if log_step and it % log_step == 0:
            print(f"Итерация {it}:")
            print("Полученная омега:", out_omega)
            print("Delta:", delta)
            print("Error:", error)
            print("Theta\n", Theta, "\n")

    print(f"Итерация {it} (последняя):")
    print("Полученная омега:", out_omega)
    print("Delta:", delta)
    print("Error:", error)
    print("Theta\n", Theta, "\n")
    return Theta, out_omega
