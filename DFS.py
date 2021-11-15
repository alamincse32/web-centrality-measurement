import numpy as np

file_name = 'small_graph.txt'
adjacency_list = {}
graph_node_number = 0
graph_edge_number = 0
color = []
start_time = []
finish_time = []
parents = []
time = 0


def get_graph():
    graph_edge_number = np.loadtxt(file_name, dtype=int, delimiter=' ')
    print(graph_edge_number)
    graph_node_number = max(np.max(graph_edge_number[0, :]), np.max(graph_edge_number[:, 1]))
    for i in range(1, graph_node_number+1):
        adjacency_list[i] = []
    for edge in graph_edge_number:
        temp = []
        if edge[0] not in adjacency_list and len(edge) == 2:
            temp.append(edge[1])
            adjacency_list[edge[0]] = temp
        else:
            temp.extend(adjacency_list[edge[0]])
            temp.append(edge[1])
            adjacency_list[edge[0]] = temp


def print_graph():
    for node in adjacency_list:
        print(node, "---->", [i for i in adjacency_list[node]])


def dfs_visit(vertex):
    color[vertex-1] = 1
    if bool(adjacency_list[vertex]):
        for v in adjacency_list[vertex]:
            if color[v-1] == 0:
                parents[v-1] = vertex
                dfs_visit(v)
    color[vertex-1] = 2


def dfs():
    if len(adjacency_list) <= 0:
        print("This graph is empty\n")
        return

    for key in adjacency_list.keys():
        color.append(0)
        parents.append(-1)
        start_time.append(0)
        finish_time.append(0)
        count = 1
    for key in adjacency_list.keys():
        if color[key - 1] == 0:
            print("Component of the graph:" + str(count))
            dfs_visit(key)
            count+=1


if '__main__' == __name__:
    get_graph()
    print_graph()
    #dfs()
