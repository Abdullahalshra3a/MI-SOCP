import networkx as nx
import matplotlib.pyplot as plt
import pickle

service_types = {
    1: {"name": "AR", "bandwidth": 2.5, "latency": 100, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 4},
    2: {"name": "CVC", "bandwidth": 3, "latency": 500, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 4},
    3: {"name": "LM", "bandwidth": 1, "latency": 1000, "processor_cores": 0.5, "Memory_GB": 0.2, "Storage_GB": 2.5},
    4: {"name": "SE", "bandwidth": 0.5, "latency": 1000, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 0.5},
    5: {"name": "VPP", "bandwidth": 2, "latency": 800, "processor_cores": 4, "Memory_GB": 8, "Storage_GB": 4}
}

def create_field_devices():
    # Sample input format in JSON
    input_data = [
        {"name": "F11", "position": "4.0,18.0", "service_choice": 4},
        {"name": "F12", "position": "4.0,14.0", "service_choice": 3},
        {"name": "F13", "position": "4.0,10.0", "service_choice": 5},
        {"name": "F14", "position": "6.0,8.0", "service_choice": 5},
        {"name": "F31", "position": "10.0,8.0", "service_choice": 4},
        {"name": "F32", "position": "10.0,4.0", "service_choice": 2},
        {"name": "F35", "position": "23.0,4.0", "service_choice": 5},
        {"name": "F34", "position": "26.0,6.0", "service_choice": 3},
        {"name": "F33", "position": "23.0,8.0", "service_choice": 4},
        {"name": "F26", "position": "23.0,10.0", "service_choice": 1},
        {"name": "F25", "position": "28.0,10.0", "service_choice": 2},
        {"name": "F24", "position": "30.0,12.0", "service_choice": 4},
        {"name": "F23", "position": "30.0,16.0", "service_choice": 4},
        {"name": "F22", "position": "28.0,18.0", "service_choice": 3},
        {"name": "F21", "position": "30.0,20.0", "service_choice": 5},
        {"name": "F16", "position": "18.0,14.0", "service_choice": 2},
        {"name": "F15", "position": "15.0,17.0", "service_choice": 2},
    ]

    # Parse the input data
    devices_info = input_data

    field_devices = {}

    for device in devices_info:
        name = device['name'].upper()
        position = tuple(map(float, device['position'].split(',')))
        service_choice = device['service_choice']
        selected_service = service_types[service_choice]["name"]
        field_devices[name] = {'type': 'F', 'position': position, 'service_type': selected_service}

    print(field_devices)

    return field_devices


def create_servers():
    # Sample input format in JSON
    input_data = [
        {
            "name": "S1",
            "position": "8.0,20.0",
            "cpu": 6.0,
            "memory": 16.0,
            "storage": 50.0,
            "service_choices": [2, 4, 5]
        },
        {
            "name": "S2",
            "position": "23.0,18.0",
            "cpu": 4.0,
            "memory": 8.0,
            "storage": 200.0,
            "service_choices": [1, 3]
        }
    ]

    # Parse the input data
    servers_info = input_data

    servers = {}

    for server in servers_info:
        name = server['name'].upper()
        position = tuple(map(float, server['position'].split(',')))
        cpu = server['cpu']
        memory = server['memory']
        storage = server['storage']
        service_choices = server['service_choices']
        services_processed = [service_types[choice]["name"] for choice in service_choices]

        servers[name] = {
            'type': 'S',
            'position': position,
            'services_processed': services_processed,
            'CPU': cpu,
            'Memory': memory,
            'Storage': storage
        }

    print(servers)

    return servers

def create_forwarding_nodes():
    scheduling_algorithms = {
        1: "Strictly Rate-Proportional (SRP) latency",
        2: "Group-Based (GB) approximations of WFQ",
        3: "Schedulers with Weakly Rate-Proportional (WRP) latency",
        4: "Frame-Based (FB) schedulers’ latency"
    }

    # Sample input format in JSON
    input_data = [
        {
            "name": "R12",
            "position": "8.0,16.0",
            "algorithm_choice": 1
        },
        {
            "name": "R13",
            "position": "8.0,12.0",
            "algorithm_choice": 1
        },
        {
            "name": "R11",
            "position": "12.0,14.0",
            "algorithm_choice": 1
        },
        {
            "name": "R10",
            "position": "15.0,12.0",
            "algorithm_choice": 1
        },
        {
            "name": "R32",
            "position": "15.0,6.0",
            "algorithm_choice": 1
        },
        {
            "name": "R30",
            "position": "18.0,8.0",
            "algorithm_choice": 1
        },
        {
            "name": "R20",
            "position": "20.0,12.0",
            "algorithm_choice": 1
        },
        {
            "name": "R31",
            "position": "20.0,6.0",
            "algorithm_choice": 1
        },
        {
            "name": "R21",
            "position": "23.0,14.0",
            "algorithm_choice": 1
        },
        {
            "name": "R22",
            "position": "26.0,16.0",
            "algorithm_choice": 1
        },
        {
            "name": "R24",
            "position": "28.0,14.0",
            "algorithm_choice": 1
        },
        {
            "name": "R23",
            "position": "26.0,12.0",
            "algorithm_choice": 1
        }
    ]

    # Parse the input data
    nodes_info = input_data

    forwarding_nodes = {}

    for node in nodes_info:
        name = node['name']
        position = tuple(map(float, node['position'].split(',')))
        algorithm_choice = node['algorithm_choice']
        scheduling_algorithm = scheduling_algorithms[algorithm_choice]

        forwarding_nodes[name] = {
            'type': 'N',
            'position': position,
            'scheduling_algorithm': scheduling_algorithm
        }

    print(forwarding_nodes)
    return forwarding_nodes


def connect_vertices(graph):
    link_types = {
        1: "edge link with bandwidth = 20 and latency = 30",
        2: "network link with bandwidth = 100 and latency = 25",
        3: "core link with bandwidth = 200 and latency = 20"
    }
    input_data = [
        {
            "source": "F11",
            "target": "R12",
            "link_type": 1
        },
        {
            "source": "R12",
            "target": "S1",
            "link_type": 2
        },
        {
            "source": "R12",
            "target": "R13",
            "link_type": 2
        },
        {
            "source": "R13",
            "target": "F12",
            "link_type": 1
        },
        {
            "source": "R13",
            "target": "F13",
            "link_type": 1
        },
        {
            "source": "R13",
            "target": "F14",
            "link_type": 1
        },
        {
            "source": "R13",
            "target": "R11",
            "link_type": 2
        },{
            "source": "R12",
            "target": "R11",
            "link_type": 2
        },
        {
            "source": "R11",
            "target": "F15",
            "link_type": 1
        },{
            "source": "R11",
            "target": "R10",
            "link_type": 2
        },
        {
            "source": "R11",
            "target": "F16",
            "link_type": 1
        },{
            "source": "R10",
            "target": "R20",
            "link_type": 3
        },
        {
            "source": "R10",
            "target": "R30",
            "link_type": 3
        },
        {
            "source": "R30",
            "target": "R20",
            "link_type": 3
        },
        {
            "source": "R30",
            "target": "R32",
            "link_type": 2
        },
        {
            "source": "R30",
            "target": "R31",
            "link_type": 1
        },
        {
            "source": "R32",
            "target": "F31",
            "link_type": 1
        },
        {
            "source": "R32",
            "target": "F32",
            "link_type": 1
        },
        {
            "source": "R32",
            "target": "R31",
            "link_type": 2
        },
        {
            "source": "R31",
            "target": "F33",
            "link_type": 1
        },
        {
            "source": "R31",
            "target": "F34",
            "link_type": 1
        },

        {
            "source": "R20",
            "target": "R21",
            "link_type": 2
        },
        {
            "source": "R21",
            "target": "R23",
            "link_type": 2
        },
        {
            "source": "R21",
            "target": "S2",
            "link_type": 1
        },
        {
            "source": "R23",
            "target": "F26",
            "link_type": 1
        },
        {
            "source": "R23",
            "target": "R24",
            "link_type": 2
        },
        {
            "source": "F23",
            "target": "R24",
            "link_type": 2
        },
        {
            "source": "R21",
            "target": "R22",
            "link_type": 2
        },
        {
            "source": "R31",
            "target": "F33",
            "link_type": 1
        },
        {
            "source": "R31",
            "target": "F34",
            "link_type": 1
        },
        {
            "source": "R22",
            "target": "R24",
            "link_type": 2
        },
        {
            "source": "R22",
            "target": "F21",
            "link_type": 1
        },
        {
            "source": "R31",
            "target": "F33",
            "link_type": 1
        },
        {
            "source": "R31",
            "target": "F34",
            "link_type": 1
        },
        {
            "source": "R31",
            "target": "F35",
            "link_type": 1
        },
        {
            "source": "R20",
            "target": "R21",
            "link_type": 2
        },{
            "source": "R22",
            "target": "F22",
            "link_type": 1
        },
        {
            "source": "R22",
            "target": "F21",
            "link_type": 1
        },

        {
            "source": "R24",
            "target": "F24",
            "link_type": 1
        },
        {
            "source": "R24",
            "target": "F25",
            "link_type": 2
        },
    ]

    # Parse the input data
    connections = input_data

    for connection in connections:
        source = connection['source'].upper()
        target = connection['target'].upper()
        link_type = connection['link_type']

        if link_type in link_types:
            if link_type == 1:
                bandwidth, latency = 25, 30
            elif link_type == 2:
                bandwidth, latency = 100, 25
            elif link_type == 3:
                bandwidth, latency = 200, 20

            graph.add_edge(source, target, bandwidth=bandwidth, latency=latency, link_description=link_types[link_type])
        else:
            print(f"Invalid link type: {link_type}")


def save_graph(graph):
    #graph = nx.path_graph(4)
    file_name = input("Enter filename to save the graph (without extension): ")
    with open(file_name+'.gpickle', 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

    #nx.write_gpickle(graph, f"{file_name}.gpickle")
    print(f"Graph saved as {file_name}.gpickle")


def display_graph(graph):
    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph)
    print("Number of positions:", len(pos))
    print("Nodes in the graph:", graph.nodes())

    node_colors = []
    for node in graph.nodes():
        if 'type' in graph.nodes[node]:
            if graph.nodes[node]['type'] == 'F':
                node_colors.append('skyblue')
            elif graph.nodes[node]['type'] == 'S':
                node_colors.append('red')
            else:
                node_colors.append('green')
        else:
            node_colors.append('gray')  # Default color for nodes without 'type' attribute

    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=500, node_color=node_colors)
    edge_labels = nx.get_edge_attributes(graph,'x,y')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.title('Generated Graph')
    plt.show()

def upload_graph():
    file_name = input("Enter the filename to upload the graph (without extension): ")
    try:
        #graph = nx.read_gpickle(f"{file_name}.gpickle")
        with open('test.gpickle', 'rb') as f:
            graph = pickle.load(f)

        display_graph(graph)
        return graph
    except FileNotFoundError:
        print("File not found.")
        return None

def BuildGraph():
    graph = nx.Graph()
    scheduling_algorithms_info = {
        1: "Strictly Rate-Proportional (SRP) scheduling algorithm assigns priorities to packets based on their rates.",
        2: "Group-Based (GB) scheduling approximates Weighted Fair Queuing (WFQ) using groups for packet scheduling.",
        3: "Schedulers with Weakly Rate-Proportional (WRP) latency assign priorities based on weak rate proportionality.",
        4: "Frame-Based (FB) schedulers use frame-based scheduling to manage latency."
    }

    field_devices = create_field_devices()
    servers = create_servers()
    forwarding_nodes = create_forwarding_nodes()

    # Add nodes to the graph with attributes
    for name, data in field_devices.items():
        graph.add_node(name, **data)

    for name, data in servers.items():
        graph.add_node(name, **data)

    for name, data in forwarding_nodes.items():
        graph.add_node(name, **data)

    connect_vertices(graph)
    display_graph(graph)
    save_graph(graph)







if __name__ == "__main__":
    print("Do you have a pre-saved graph to upload?")
    option = input("Enter 'yes' or 'no': ").lower()

    if option == 'yes':
        graph = upload_graph()
        display_graph(graph) if graph else (print("The graph has a None value. Creating a new graph..."), BuildGraph())
    else:
        print("Creating a new graph...")
        BuildGraph()


