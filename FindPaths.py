import networkx as nx
import pickle


#the worst-case end-to-end delay (WCD)
def calculate_wcd(graph, path):
    sigma = 1500  # Burst size in Byte/ms
    L = 1500  # Packet size in Byte
    propagation_delay = 40  # Propagation delay in microseconds

    wcd = sigma / min(sigma for i, j in zip(path[:-1], path[1:]))
    for i, j in zip(path[:-1], path[1:]):
        w_ij = graph[i][j]['bandwidth']  # Available bandwidth for the edge from node i to node j
        r_ij = sigma
        theta_ij = (L / w_ij) + (L / r_ij)
        l_ij = propagation_delay / 1000  # Propagation delay converted to ms
        n_i = 40 / 1000  # Node delay converted to ms
        wcd += theta_ij + l_ij + n_i # according to the equation (3) in the selected paper ' QoS routing with worst-case delay constraints: Models, algorithms and performance analysis'

    return wcd


def find_paths(graph, start_node):
    target_node = None

    # Find the target node that meets the specified conditions
    for node in graph.nodes:
        if graph.nodes[node]['type'] == 'S' and graph.nodes[start_node]['service_type'] in graph.nodes[node]['services_processed']:
            target_node = node
            break

    if target_node is None:
        # If no target node is found, return an empty list
        return []

    # Find all paths from start_node to the target_node
    paths = list(nx.all_simple_paths(graph, start_node, target_node))#[[0, 1, 3], [0, 2, 3], [0, 3]]
    # Add wcd=0 to each path
    paths= [(path, {'wcd': 0}) for path in paths]#[([0, 1, 3], {'wcd': 0}), ([0, 2, 3], {'wcd': 0}), ([0, 3], {'wcd': 0})]
    return paths


def sorted_paths():
    filename = input("Enter the filename of the graph to load (without extension): ")
    try:
        with open(filename + '.gpickle', 'rb') as f:
            graph = pickle.load(f)

        field_devices = [node for node in graph.nodes if graph.nodes[node]['type'] == 'F']
        paths = {}
        for device in field_devices:
            Device_paths = find_paths(graph, device)#[([0, 1, 3], {'wcd': 0}), ([0, 2, 3], {'wcd': 0}), ([0, 3], {'wcd': 0})]
            paths.update({device: tuple(Device_paths)})#{'f1': ([0, 1, 3], {'wcd': 0}), ([0, 2, 3], {'wcd': 0}), ([0, 3], {'wcd': 0})}
            #print(paths)
            #paths[device] = Device_paths

            # Sort paths for each field device based on WCD (shortest to longest)
            #paths_with_wcd[device] = sorted(paths_with_wcd[device], key=lambda x: x[1])

            # Print sorted paths for each field device
            #print(f"\nAvailable paths for Field Device '{device}' and its related server processing '{graph.nodes[device]['service_type']}':")

    except FileNotFoundError:
        print("File not found.")
    for key, path in paths.items():
        print(f"Feild Device: {key:}, Paths: {str(path)[1:-1]}")
    return graph, paths


#'''''''''
if __name__ == "__main__":
#sorted_paths()
 '''''''''
 G = nx.complete_graph(4)
 paths = list(nx.all_simple_paths(G, source=0, target=3))
 paths= [(path, {'wcd': 0}) for path in paths]#[([0, 1, 3], {'wcd': 0}), ([0, 2, 3], {'wcd': 0}), ([0, 3], {'wcd': 0})]
 print(paths)
 #paths = {}
 device = 'f1'

 paths = {device: tuple(paths)}
 print(paths)#{'f1': (([0, 1, 2, 3], {'wcd': 0}), ([0, 1, 3], {'wcd': 0}), ([0, 2, 1, 3], {'wcd': 0}), ([0, 2, 3], {'wcd': 0}), ([0, 3], {'wcd': 0}))}
 print(paths['f1'][2][1]['wcd'])#0

 paths['f2']=[0, 1, 2, 3], [0, 1, 3], [0, 2, 1, 3], [0, 2, 3], [0, 3]
 print(paths)
 for u, path in paths.items():
  print(path)
  for i in path:
    print(i)
 '''''''''
