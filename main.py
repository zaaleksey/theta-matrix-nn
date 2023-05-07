from conjugate_gradient import conjugate
from file_utils import save_list_in_file
from generators import *
from initial_theta import *
from gradient_descent import gradient_descent


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
        # theta, out_omega, _ = find_routing_matrix(inp_omega, w, get_uniform_initial_theta)
        # print("Начальная омега:", inp_omega)
        # print("Полученная омега:", out_omega, sum(out_omega))
        # print("Theta\n", theta, "\n")
        #
        # theta, out_omega, _ = find_routing_matrix(inp_omega, w, get_initial_theta)
        # print("Начальная омега:", inp_omega)
        # print("Полученная омега:", out_omega, sum(out_omega))
        # print("Theta\n", theta, "\n")
        # print("#" * 150)
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


if __name__ == '__main__':
    # simple_case_1()
    simple_case_2()
    # case_1()
    # case_2()
