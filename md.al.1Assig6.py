from queue import Queue

import numpy as np
import matplotlib.pyplot as plt
import math

file_name = 'wiki_vote.txt'
positive_infinity = math.inf


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def read_graph_from_file():
    edge_list = np.loadtxt(file_name, dtype=int, delimiter="\t")
    max_number_of_node = max(np.max(edge_list[:, 0]), np.max(edge_list[:, 1]))
    graph_matrix = np.zeros((max_number_of_node, max_number_of_node))

    for edge in edge_list:
        graph_matrix[edge[0] - 1][edge[1] - 1] = 1

    return graph_matrix


def degree_crentrality(graph):
    if len(graph) <= 0:
        return "This is an empty graph\n"
    max_degree_in = max_degree_out = -1
    in_index = out_index = -1
    for i in range(len(graph[0])):
        if np.count_nonzero(graph[i, :]) > max_degree_in:
            max_degree_in = np.count_nonzero(graph[i, :])
            in_index = i

    print(str(in_index + 1) + "th vertex has maximum out-degree: " + str(max_degree_in) + "\n")

    transpose_matrix = np.transpose(graph)
    for i in range(len(transpose_matrix[0])):
        if np.count_nonzero(graph[:, i]) > max_degree_out:
            max_degree_out = np.count_nonzero(graph[:, i])
            out_index = i
    print(str(out_index + 1) + "th vertex has maximum in-degree: " + str(max_degree_out) + "\n")


def lesat_distance(graph, node):
    q = Queue()
    distance = {k: np.inf for k in range(0, len(graph))}
    visit_vertices = set()
    q.put(node - 1)
    visit_vertices.update({node - 1})
    while not q.empty():
        vertex = q.get()
        if vertex == node - 1:
            distance[vertex] = 0
        for u in np.nonzero(graph[vertex - 1])[0]:
            if u not in visit_vertices:
                if distance[u] > distance[vertex] + 1:
                    distance[u] = distance[vertex] + 1
            q.put(u)
            visit_vertices.update({u})

    print(distance)


def prestige(graph, p, v):
    k = 0
    p = np.transpose(p)
    g = np.transpose(graph)
    lemda = []
    while True:
        k += 1
        p1 = np.matmul(g, p)
        i = np.argmax(p1)
        lemda.append(p1[i] / p[i])
        p1 = p1 / p1[i]
        s = 0
        for i in range(len(p1)):
            s += (p1[i] - p[i]) ** 2
        s = math.sqrt(s)
        p = p1
        if s <= v:
            break
    s = 0
    for i in range(len(p)):
        s += p[i] ** 2
    s = math.sqrt(s)
    p = p / s
    return [p, lemda]


def page_rank(graph, alpha, v):
    temp_graph = np.zeros((len(graph),len(graph)))
    for i in range(len(graph)):
        out_edge = np.count_nonzero(graph[i])
        if out_edge != 0:
            temp_graph[i] = graph[i] / out_edge
    normalized_matrix = np.full((len(graph), len(graph)), (1 / len(graph)))

    temp_graph = np.transpose(temp_graph)
    normalized_matrix = np.transpose(normalized_matrix)
    p = np.ones(len(temp_graph))
    p = np.transpose(p)
    k = 0
    lemda = []
    while True:
        k += 1
        p1 = (1 - alpha) * np.matmul(temp_graph, p) + alpha * np.matmul(normalized_matrix, p)
        i = np.argmax(p1)
        lemda.append(p1[i] / p[i])
        s = 0
        for i in range(len(p1)):
            s += (p1[i] - p[i]) ** 2
        s = math.sqrt(s)
        p = p1
        if s <= v:
            break

    s = 0
    for i in range(len(p)):
        s += p[i] ** 2
    s = math.sqrt(s)
    p = p / s
    return [p, lemda]


def hub_authority(graph, v, v1):
    hubs = np.full(len(graph),(1/len(graph)))
    k = 0
    a = []
    while k<=v:
        k+=1
        last_hubs = hubs
        a = np.matmul(np.transpose(graph),np.transpose(hubs))
        hubs = np.matmul(graph,a)
        i = np.argmax(a)
        a = a/a[i]
        i = np.argmax(hubs)
        hubs = hubs/hubs[i]
        err = sum([abs(hubs[i] - last_hubs[i]) for i in range(len(hubs))])
        print(err)
        if err <= v1:
            break
    return [hubs,a]
    # a = np.ones(len(graph))
    # a = np.transpose(a)
    # print(a)
    # print(graph)
    # k = 0
    # graph_transpose = np.transpose(graph)
    # h = 0
    # while k <= v:
    #     k += 1
    #     h = np.matmul(graph, a)
    #     i = np.argmax(h)
    #     h = h / h[i]
    #     a = np.matmul(graph_transpose, h)
    #     i = np.argmax(a)
    #     a1 = a / a[i]
    #     s = 0
    #     for i in range(len(h)):
    #         s += (a[i] - a1[i]) ** 2
    #     s = math.sqrt(s)
    #     a = a1
    #     print(h)
    #     if s <= v1:
    #         break
    # return [h, a]


if __name__ == '__main__':
    graph = read_graph_from_file()
    degree_crentrality(graph)
    p = np.ones(int(len(graph)))

    get_prestige = prestige(graph, p, 0.000001)
    m = np.max(get_prestige[0])
    max_prestige = list(get_prestige[0]).index(m)

    plt.xlim(0, len(get_prestige[1]), 0.5)
    plt.ylim(1, 2.5, 0.5)
    x = [i for i in range(0, len(get_prestige[1]))]
    plt.plot(x, get_prestige[1])
    plt.show()

    for i in np.arange(0, 1.2, 0.2):
        get_page_rank = page_rank(graph, i, 0.001)
        plt.title(r'Histogram of PR: $\alpha=$' + str(i))
        plt.xlabel("Page rank distribution")
        plt.ylabel("Number of nodes")
        plt.hist(get_page_rank[0])
        plt.savefig("photo_" + str(i) + ".png")
        plt.show()

    get_hub = hub_authority(graph, 20,0.001)
    plt.xlabel("Hub distribution")
    plt.ylabel("Number of nodes")
    plt.hist(get_hub[0])
    plt.savefig("photo_hub.png")
    plt.show()
