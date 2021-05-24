import itertools
import numpy as np


def distance(i, j):
    return abs(i - j)


def generate_random_nodes(n, lim, lines):
    City = complex
    cities = set()
    nodes = np.random.randint(1, lim, n, dtype=int)
    for i in nodes:
        v, x, y = lines[i - 1].split()
        cities.add(City(float(x), float(y)))
    return list(cities)


def get_distance_btw(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def get_distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def brute_force_tsp(cities):
    start = cities[0]
    all_permutations = [[start] + list(p) for p in itertools.permutations(set(cities) - {start})]
    sp = min(all_permutations, key=lambda path: sum([distance(path[i], path[i - 1]) for i in range(len(path))]))
    c = sum([distance(sp[i], sp[i - 1]) for i in range(len(sp))])
    return sp, c


def solution(cities):
    all_permutations = [list(p) for p in itertools.permutations(cities)]
    for i in range(len(all_permutations)):
        all_permutations[i].append(all_permutations[i][0])
    sp = min(all_permutations, key=lambda path: sum([get_distance_btw(path[i], path[i - 1]) for i in range(len(path))]))
    c = sum([get_distance_btw(sp[i], sp[i - 1]) for i in range(len(sp))])
    return sp, c


class City:
    def __init__(self, name, x, y):
        self.id = name
        self.x = x
        self.y = y


def generate_output(x):
    lst = x
    y = []
    for i in lst:
        l = []
        for a, b in i:
            l.append(complex(a, b))
        all_permutations = [list(p) for p in itertools.permutations(l)]
        # print(all_permutations)
        sp = min(all_permutations, key=lambda path: sum([distance(path[i], path[i - 1]) for i in range(len(path))]))
        c = sum([distance(sp[i], sp[i - 1]) for i in range(len(sp))])
        y_row = []
        for e in sp:
            y_row.append([e.real, e.imag])
        y.append(y_row)
    return np.asarray(y)


def generate_input(batch_size, seq_length, vector_dim):
    return np.random.randint(0, 100, (batch_size, seq_length, 2)).astype(np.float32) / 100


# Used to generate new city data input and output
def generate_test_data_tsp(batch_size=10, seq=10):
    inp = np.random.randint(0, 100, (batch_size, seq, seq)).astype(np.float32)
    out = np.random.randint(0, 100, (batch_size, seq, seq)).astype(np.float32)
    for ind in range(batch_size):
        lst = np.random.randint(0, 10, (seq, 2)).astype(np.float32)
        cities = []
        for i in range(seq):
            city = City(i, lst[i][0], lst[i][1])
            cities.append(city)

        g, t = np.zeros((seq, seq)), np.zeros((seq, seq))
        for i in range(seq):
            for j in range(seq):
                g[i][j] = get_distance(lst[i], lst[j])
        res, p = solution(cities)
        ans = 0
        for i in range(1, seq + 1):
            t[res[i - 1].id][res[i].id] = 1
            ans += get_distance_btw(res[i - 1], res[i])
        # print(g,t)
        inp[ind] = g
        out[ind] = t
    return inp, out
