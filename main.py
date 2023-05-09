import time

from file_utils import save_list_in_file
from generators import *
from gradient_descent import gradient_descent
from initial_theta import *


def simple_case_1():
    inp_omega = np.array([.35, .27, .15, .23])
    w = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ])

    theta1, omega1, errors1, it1 = gradient_descent(inp_omega, w, get_uniform_initial_theta)
    print("Равномерное распределение начальной маршрутной матрицы")
    print(f"Количество итераций: {it1}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega1}")
    print("Полученная маршрутная матрица:")
    print(theta1)

    print("#" * 150)

    theta2, omega2, errors2, it2 = gradient_descent(inp_omega, w, get_smart_initial_theta)
    print("Распределение начальной маршрутной матрицы с использованием вектора omega")
    print(f"Количество итераций: {it2}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega2}")
    print("Полученная маршрутная матрица:")
    print(theta2)

    save_list_in_file(errors1, "data/errors1.txt")
    save_list_in_file(errors2, "data/errors2.txt")


def simple_case_2():
    inp_omega = np.array([.35, .27, .15, .23])
    w = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ])

    theta1, omega1, errors1, it1 = conjugate(inp_omega, w, get_uniform_initial_theta)
    print("Равномерное распределение начальной маршрутной матрицы")
    print(f"Количество итераций: {it1}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega1}")
    print("Полученная маршрутная матрица:")
    print(theta1)

    print("#" * 150)

    theta2, omega2, errors2, it2 = conjugate(inp_omega, w, get_smart_initial_theta)
    print("Распределение начальной маршрутной матрицы с использованием вектора omega")
    print(f"Количество итераций: {it2}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega2}")
    print("Полученная маршрутная матрица:")
    print(theta2)


def case_1():
    max_it = 10_000
    max_step = 1000
    inp_omega = np.array([.35, .27, .15, .23])

    difference = []
    its = []
    opt_its = []
    for step in range(max_step):
        w = get_random_w(len(inp_omega))
        _, _, _, it = gradient_descent(inp_omega, w, get_uniform_initial_theta, max_it=max_it)
        _, _, _, opt_it = gradient_descent(inp_omega, w, get_smart_initial_theta, max_it=max_it)

        if it != max_it or opt_it != max_it:
            its.append(it)
            opt_its.append(opt_it)
            if it - opt_it > 0:
                difference.append(it - opt_it)

        print(f"{step}/{max_step}")

    save_list_in_file(its, "data/its.txt")
    save_list_in_file(opt_its, "data/opt_its.txt")
    save_list_in_file(difference, "data/diff.txt")


def case_2():
    omegas = get_omegas(4, 8, 10)
    for omega in omegas:
        w = get_random_w(len(omega))
        theta1, omega1, errors1, it1 = gradient_descent(omega, w, get_uniform_initial_theta)
        theta2, omega2, errors2, it2 = gradient_descent(omega, w, get_smart_initial_theta)

        if it1 != 100_000 or it2 != 100_000:
            print("Равномерное распределение начальной маршрутной матрицы")
            print(f"Количество итераций: {it1}")
            print(f"Сравнение омег:\nВход  {omega}\nВыход {omega1}")
            print("Полученная маршрутная матрица:")
            print(theta1)

            print()

            print("Распределение начальной маршрутной матрицы с использованием вектора omega")
            print(f"Количество итераций: {it2}")
            print(f"Сравнение омег:\nВход  {omega}\nВыход {omega2}")
            print("Полученная маршрутная матрица:")
            print(theta2)

        print("\n", "#" * 150)


def case_3():
    max_step = 100
    omega = np.array([.35, .27, .15, .23])
    for step in range(max_step):
        w = get_random_w(len(omega))
        theta1, omega1, errors1, it1 = conjugate(omega, w, get_uniform_initial_theta)
        theta2, omega2, errors2, it2 = conjugate(omega, w, get_smart_initial_theta)

        if it1 != 10_000 or it2 != 10_000:
            print("Равномерное распределение начальной маршрутной матрицы")
            print(f"Количество итераций: {it1}")
            print(f"Сравнение омег:\nВход  {omega}\nВыход {omega1}")
            print("Полученная маршрутная матрица:")
            print(theta1)

            print()

            print("Распределение начальной маршрутной матрицы с использованием вектора omega")
            print(f"Количество итераций: {it2}")
            print(f"Сравнение омег:\nВход  {omega}\nВыход {omega2}")
            print("Полученная маршрутная матрица:")
            print(theta2)

        print("\r" + f"{step + 1}/{count}", end="")
        print("\n", "#" * 150)


def case_4():
    inp_omega = np.array([0.3, 0.43, 0.27])
    w = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])

    theta1, omega1, errors1, it1 = gradient_descent(inp_omega, w, get_uniform_initial_theta)
    print("Равномерное распределение начальной маршрутной матрицы")
    print(f"Количество итераций: {it1}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega1}")
    print("Полученная маршрутная матрица:")
    print(theta1)
    theta1, omega1, errors1, it1 = conjugate(inp_omega, w, get_uniform_initial_theta)
    print("Равномерное распределение начальной маршрутной матрицы")
    print(f"Количество итераций: {it1}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega1}")
    print("Полученная маршрутная матрица:")
    print(theta1)

    print("#" * 150)

    theta2, omega2, errors2, it2 = gradient_descent(inp_omega, w, get_smart_initial_theta)
    print("Распределение начальной маршрутной матрицы с использованием вектора omega")
    print(f"Количество итераций: {it2}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega2}")
    print("Полученная маршрутная матрица:")
    print(theta2)
    theta2, omega2, errors2, it2 = conjugate(inp_omega, w, get_smart_initial_theta)
    print("Распределение начальной маршрутной матрицы с использованием вектора omega")
    print(f"Количество итераций: {it2}")
    print(f"Сравнение омег:\nВход  {inp_omega}\nВыход {omega2}")
    print("Полученная маршрутная матрица:")
    print(theta2)


def case_5():
    max_it = 5000
    max_step = 500
    inp_omega = np.array([.35, .27, .15, .23])

    its = []
    opt_its = []
    for step in range(max_step):
        w = get_random_w(len(inp_omega))
        _, _, _, it = conjugate(inp_omega, w, get_uniform_initial_theta, max_it=max_it)
        _, _, _, opt_it = conjugate(inp_omega, w, get_smart_initial_theta, max_it=max_it)

        if it != max_it or opt_it != max_it:
            its.append(it)
            opt_its.append(opt_it)

        save_list_in_file(its, "data/its.txt")
        save_list_in_file(opt_its, "data/opt_its.txt")

        print("\r" + f"{step + 1}/{count}", end="")


def general(omegas, count):
    max_iter = 2_000
    gradient_its = []
    gradient_opt_its = []
    conjugate_its = []
    conjugate_opt_its = []

    for index, omega in enumerate(omegas):
        print(f"{index + 1}/{len(omegas)}", omega)
        for step in range(count):
            w = get_random_w(len(omega))
            _, _, _, it = gradient_descent(omega, w, get_uniform_initial_theta)
            _, _, _, opt_it = gradient_descent(omega, w, get_smart_initial_theta)
            if it != max_iter and opt_it != max_iter:
                gradient_its.append(it)
                gradient_opt_its.append(opt_it)

            _, _, _, it = conjugate(omega, w, get_uniform_initial_theta)
            _, _, _, opt_it = conjugate(omega, w, get_smart_initial_theta)
            if it != max_iter and opt_it != max_iter:
                conjugate_its.append(it)
                conjugate_opt_its.append(opt_it)

            print("\r" + f"\t{step + 1}/{count}", end="")
        print()

    save_list_in_file(gradient_its, f"data/gradient/its.txt")
    save_list_in_file(gradient_opt_its, f"data/gradient/opt_its.txt")

    save_list_in_file(conjugate_its, f"data/conjugate/its.txt")
    save_list_in_file(conjugate_opt_its, f"data/conjugate/opt_its.txt")


if __name__ == '__main__':
    # simple_case_1()
    # print("\n", "-" * 150, "\n")
    # simple_case_2()

    count = 1_000

    omegas = [
        np.array([.35, .27, .15, .23]),
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.31, 0.18, 0.27, 0.24]),
        np.array([0.36, 0.25, 0.18, 0.21]),
        np.array([0.3, 0.22, 0.22, 0.26]),
    ]
    # general(omegas, count)
