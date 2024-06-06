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

def create_field_devices(num, service_types):# num is the number of field devices in our network.

    field_devices = {}
    for i in range(num):
        name = input(f"Enter name for Field Device {i + 1}: ").upper()
        position = tuple(map(float, input(f"Enter position (x, y) for {name}: ").split(',')))

        print("Select a service type offered by this Field Device:")
        for key, value in service_types.items():
            print(f"{key} - {value}")

        service_choice = int(input("Enter the service type (1/2/3/4/5): "))
        selected_service = service_types[service_choice]["name"]

        #selected_service = service_types.get(service_choice)

        field_devices[name] = {'type': 'F', 'position': position, 'service_type': selected_service}# We might add more info regarding the flow such as packet size.
        # field_devices = {'F1': {'type': 'F', 'position': (1.0, 1.0), 'service_type': 'CVC'}, 'F2': {'type': 'F', 'position': (2.0, 2.0), 'service_type': 'LM'}}

    return field_devices


def create_servers(num, service_types):

    servers = {}
    for i in range(num):
        name = input(f"Enter name for Server {i + 1}: ").upper()
        position = tuple(map(float, input(f"Enter position (x, y) for {name}: ").split(',')))
        cpu = float(input(f"Enter CPU for {name}: "))
        memory = float(input(f"Enter Memory for {name}: "))
        storage = float(input(f"Enter Storage for {name}: "))
        print(f"Select service types processed by {name} (Enter one or multiple numbers, e.g., 1 3 5):")
        for key, value in service_types.items():
            print(key, '     ', value["name"])

        selected_services = input("Enter the service types separated by space: ").split()
        services_processed = [service_types[s]["name"] for s in selected_services]#service_types[int(s)]["name"] = [1,3,5]
        servers[name] = {'type': 'S', 'position': position, 'services_processed': services_processed, 'CPU': cpu, 'Memory': memory, 'Storage': storage}# services_processed = {'name': 'SE', 'bandwidth': 0.5, 'latency': 1000, 'processor_cores': 1, 'Memory_GB': 2, 'Storage_GB': 0.5}
    return servers
"""
if 'S1' in servers:
    if len(servers['S1']['services_processed']) > 0:
        bandwidth_of_first_service = servers['S1']['services_processed'][0]['bandwidth']
        print(f"Bandwidth of the first service for server 'S1': {bandwidth_of_first_service}")
    else:
        print("No services selected for server 'S1'")
else:
    print("Server 'S1' not found")
"""
def create_forwarding_nodes(num, scheduling_algorithm):
    forwarding_nodes = {}
    scheduling_algorithms = {
        1: "Strictly Rate-Proportional (SRP) latency",
        2: "Group-Based (GB) approximations of WFQ",
        3: "Schedulers with Weakly Rate-Proportional (WRP) latency",
        4: "Frame-Based (FB) schedulers’ latency"
    }
    for i in range(num):
        name = input(f"Enter name for Forwarding Node {i + 1}: ")
        position = tuple(map(float, input(f"Enter position (x, y) for {name}: ").split(',')))
        #algorithm = int(input(f"Select scheduling algorithm for {name} (1/2/3/4): "))
        forwarding_nodes[name] = {'type': 'N', 'position': position,
                                  'scheduling_algorithm': scheduling_algorithm}
    return forwarding_nodes


def connect_vertices(graph):
    link_types = {
        1: "edge link with bandwidth = 20 and latency = 30",
        2: "network link with bandwidth = 100 and latency = 25",
        3: "core link with bandwidth = 200 and latency = 20"
    }
    while True:
        source = input("Enter source vertex name (or 'done' to exit): ").upper()
        if source == 'DONE':
            break
        target = input("Enter target vertex name: ").upper()
        link_type = int(input("Enter link type (1/2/3): "))
        if link_type in link_types:
            bandwidth, latency = 20, 30
            if link_type == 2:
                bandwidth, latency = 100, 25
            elif link_type == 3:
                bandwidth, latency = 200, 20
            #link_description = link_types.get(link_type)
            graph.add_edge(source, target, bandwidth=bandwidth, latency=latency, link_description=(bandwidth,latency))
        else:
            print("Invalid link type.")


def save_graph(graph):
    #graph = nx.path_graph(4)
    file_name = input("Enter filename to save the graph (without extension): ")
    with open(file_name+'.gpickle', 'wb') as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)

    #nx.write_gpickle(graph, f"{file_name}.gpickle")
    print(f"Graph saved as {file_name}.gpickle")


def display_graph(graph):
    #plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots()
    pos = nx.spring_layout(graph)
    color_map = []
    for i in graph:
        if graph.nodes[i]['type']== 'F':
            color_map.append('blue')
        elif graph.nodes[i]['type']== 'S':
            color_map.append('red')
        else:
            color_map.append('skyblue')
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=500, node_color=color_map)
    edge_labels = nx.get_edge_attributes(graph, 'link_description')
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

def BulidGraph():
    graph = nx.Graph()
    scheduling_algorithms_info = {
        1: "Strictly Rate-Proportional (SRP) scheduling algorithm assigns priorities to packets based on their rates.",
        2: "Group-Based (GB) scheduling approximates Weighted Fair Queuing (WFQ) using groups for packet scheduling.",
        3: "Schedulers with Weakly Rate-Proportional (WRP) latency assign priorities based on weak rate proportionality.",
        4: "Frame-Based (FB) schedulers use frame-based scheduling to manage latency."
    }

    print("Select a scheduling algorithm for forwarding nodes:")
    for key, value in scheduling_algorithms_info.items():
        print(f"{key} - {value}")

    scheduling_algorithm = int(input("Enter the scheduling algorithm (1/2/3/4): "))

    num_field_devices = int(input("Enter number of Field Devices: "))
    field_devices = create_field_devices(num_field_devices)

    num_servers = int(input("Enter number of Servers: "))
    servers = create_servers(num_servers)

    num_forwarding_nodes = int(input("Enter number of Forwarding Nodes: "))
    forwarding_nodes = create_forwarding_nodes(num_forwarding_nodes, scheduling_algorithm)

    # Add nodes to the graph
    graph.add_nodes_from([(name, data) for name, data in field_devices.items()])
    for name, data in field_devices.items(): graph.nodes[name]['pos'] = data['position']
    graph.add_nodes_from([(name, data) for name, data in servers.items()])
    for name, data in servers.items(): graph.nodes[name]['pos'] = data['position']
    graph.add_nodes_from([(name, data) for name, data in forwarding_nodes.items()])
    for name, data in forwarding_nodes.items(): graph.nodes[name]['pos'] = data['position']
    connect_vertices(graph)
    display_graph(graph)
    save_graph(graph)
    # print(nx.get_node_attributes(graph, 'pos'))







if __name__ == "__main__":
    print("Do you have a pre-saved graph to upload?")
    option = input("Enter 'yes' or 'no': ").lower()

    if option == 'yes':
        graph = upload_graph()
        display_graph(graph) if graph else (print("The graph has a None value. Creating a new graph..."), BuildGraph())
    else:
        print("Creating a new graph...")
        BuildGraph()


