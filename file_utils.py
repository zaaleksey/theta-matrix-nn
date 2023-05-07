def save_list_in_file(data, filename):
    data = map(str, data)
    with open(filename, "w") as file:
        file.write(",".join(data))


def load_list_from_file(filename, map_func):
    with open(filename, "r") as file:
        data = map(map_func, file.readline().split(","))

    return list(data)
