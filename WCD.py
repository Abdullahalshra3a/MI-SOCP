import cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from math import prod
import re




def info(prob):
    # Get the number of variables
    num_variables = prob.variables.get_num()
    print(f"Number of variables: {num_variables}")

    # Get the number of constraints
    num_constraints = prob.linear_constraints.get_num()
    print(f"Number of linear constraints: {num_constraints}")

    # Get the number of quadratic constraints (if any)
    num_quadratic_constraints = prob.quadratic_constraints.get_num()
    print(f"Number of quadratic constraints: {num_quadratic_constraints}")



# Function to generate consistent variable names for edges
def generate_edge_variable_name(edge):
    return 'x_{}_{}'.format(*sorted(edge))

service_types = {
    1: {"name": "AR", "bandwidth": 2.5, "latency": 100, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 4},
    2: {"name": "CVC", "bandwidth": 3, "latency": 500, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 4},
    3: {"name": "LM", "bandwidth": 1, "latency": 1000, "processor_cores": 0.5, "Memory_GB": 0.2, "Storage_GB": 2.5},
    4: {"name": "SE", "bandwidth": 0.5, "latency": 1000, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 0.5},
    5: {"name": "VPP", "bandwidth": 2, "latency": 800, "processor_cores": 4, "Memory_GB": 8, "Storage_GB": 4}
}

def define_scheduling_algorithm():
    scheduling_algorithms_types = {
        1: "Strictly Rate-Proportional (SRP) latency",
        2: "Group-Based (GB) approximations of WFQ",
        3: "Schedulers with Weakly Rate-Proportional (WRP) latency",
        4: "Frame-Based (FB) schedulers’ latency"
    }
    for i in range(4):
        print(i + 1, scheduling_algorithms_types[i + 1])
    scheduling_algorithm = input(f"Enter number of the used node scheduling_algorithm for WCD calculation : ")
    return int(scheduling_algorithm)
# Define global P
P = {}
scheduling_algorithm = define_scheduling_algorithm()
resource_graph, paths = sorted_paths()  # WCD = 0 for all available pathes ===> {'f1': ([0, 1, 3], {'wcd': 0}), ([0, 2, 3], {'wcd': 0}), ([0, 3], {'wcd': 0})}
# scheduling_algorithm = next((resource_graph.nodes[node]['scheduling_algorithm'] for node in resource_graph.nodes if resource_graph.nodes[node]['type'] == 'N'), 1)
# field_devices = [node for node in resource_graph.nodes if resource_graph.nodes[node]['type'] == 'F']#field_devices[name] = {'type': 'F', 'position': position, 'service_type': selected_service}
field_devices = {node: resource_graph.nodes[node] for node in resource_graph.nodes if resource_graph.nodes[node]['type'] == 'F'}
field_devices_delta = {}  # the latency deadline accepted to realize the QoS requirements for the related flow


for device, attributes in field_devices.items():
    # Extracting service_type attribute value from the device
    service_type = attributes['service_type']

    # Finding the key in service_types dictionary based on 'name' attribute
    try:
        service_type_key = next(key for key, value in service_types.items() if value['name'] == service_type)
    except StopIteration:
        print(f"Service type {service_type} not found in service_types dictionary.")
        continue

    # Extracting δ (latency) from the service_types dictionary using the key
    delta = service_types[service_type_key]['latency']
    field_devices_delta[device] = delta#The path should achieve an end-to-end deadline (δ), where the worst-case end-to-end delay (WCD) does not exceed δ.

def flow_number(i, j, field_devices, flow):
    if i not in P:
        P[i] = {}
    if j not in P[i]:
        P[i][j] = 0

    for device in field_devices:
        for path in flow[device]['paths']:
            if Check_edge_inpath(path, i, j):
                P[i][j] += 1
                break

    return P[i][j]

def Check_edge_inpath(path, i, j):

    for x in range(len(path) - 1):
        if [path[x], path[x + 1]] == [i, j] or [path[x], path[x + 1]] == [j, i]:
           return True
    return False

for i, j in resource_graph.edges:
    resource_graph[i][j]['bandwidth'] = resource_graph[i][j]['bandwidth'] * 1000


def calculate_total_combinations(flow):
    # Calculate the number of paths for each device
    path_counts = [len(flow[device]['paths']) for device in flow]
    print(path_counts)
    # Total combinations is the product of all path counts
    return prod(path_counts)


def generate_path_combinations(flow):
    device_paths = {device: [tuple(path[0]) for path in flow[device]['paths']] for device in flow}
    devices = list(device_paths.keys())

    def combinations_generator():
        for combination in itertools.product(*device_paths.values()):
            yield dict(zip(devices, combination))

    return combinations_generator()


def create_subgraph(combination, resource_graph):
    subgraph = nx.DiGraph()
    for device, path in combination.items():
        for i in range(len(path) - 1):
            subgraph.add_edge(path[i], path[i + 1])
    return subgraph


def plot_subgraphs(combinations, resource_graph, num_plots=10):
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("First 10 Path Combinations Subgraphs", fontsize=16)

    for i, (ax, combo) in enumerate(zip(axs.flatten(), combinations)):
        if i >= num_plots:
            break

        subgraph = create_subgraph(combo, resource_graph)
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, ax=ax, with_labels=True, node_color='lightblue',
                node_size=300, font_size=8, font_weight='bold')
        ax.set_title(f"Combination {i + 1}", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm=1, sigma=1500, rho=1500):
    prob = None
    try:
        if not isinstance(resource_graph, nx.DiGraph):
            resource_graph = nx.DiGraph(resource_graph)

        def add_variable_if_new(name, var_type):
            if name not in added_variables:
                prob.variables.add(names=[name], types=[var_type])
                added_variables.add(name)

        def add_constraint_if_new(constraint_name, lin_expr, sense, rhs):
            if constraint_name not in added_constraints:
                prob.linear_constraints.add(
                    lin_expr=[lin_expr],
                    senses=[sense],
                    rhs=[rhs],
                    names=[constraint_name]
                )
                added_constraints.add(constraint_name)

        tau = 1
        L = 1500
        l_ij = 0.0005
        n_i = 40 / 1000

        flow = {}
        for device in field_devices_delta:
            flow[device] = {
                'paths': paths[device],
                'deadline': field_devices_delta[device],
                'burst': sigma,
                'rate': rho,
                'reserved_rates': {rho * tau + sigma}
            }

        total_combinations = calculate_total_combinations(flow)
        print(f"Total number of combinations: {total_combinations}")

        combinations_generator = generate_path_combinations(flow)

        first_100_combinations = list(itertools.islice(combinations_generator, 100))

        valid_solutions = []
        valid_solution_count = 0

        for combination in first_100_combinations:
            prob = cplex.Cplex()
            prob.set_problem_type(prob.problem_type.MIQCP)
            added_variables = set()
            added_constraints = set()

            remaining_capacity = {}
            reservable_capacity = {}
            for device, path in combination.items():
                for i, j in zip(path[:-1], path[1:]):
                    edge = (i, j)
                    if edge not in reservable_capacity:
                        remaining_capacity[edge] = resource_graph[i][j]['bandwidth']
                        reservable_capacity[edge] = 0

            for device, path in combination.items():
                for i, j in zip(path[:-1], path[1:]):
                    edge = (i, j)
                    var_name = generate_edge_variable_name(edge)
                    add_variable_if_new(var_name, 'B')

                    flow_rate_var = f"r_{device}_{i}_{j}"
                    add_variable_if_new(flow_rate_var, 'C')

                    constraint_name = f"flow_rate_constraint_{device}_{i}_{j}"
                    add_constraint_if_new(
                        constraint_name,
                        [[var_name, flow_rate_var], [reservable_capacity[edge], -1.0]],
                        'L', 0.0
                    )

                    r_min = f"r_min_{device}_{i}_{j}"
                    add_variable_if_new(r_min, 'C')
                    constraint_name_min = f"min_flow_rate_constraint_{device}_{i}_{j}"
                    add_constraint_if_new(
                        constraint_name_min,
                        [[var_name, flow_rate_var, r_min], [reservable_capacity[edge], -1.0, -rho]],
                        'L', 0.0
                    )

                    reservable_capacity[edge] += rho * tau + sigma
                    remaining_capacity[edge] -= reservable_capacity[edge]
            '''''''''
            for node in resource_graph.nodes:
                incoming_edges = [(j, node) for j in resource_graph.predecessors(node)]
                outgoing_edges = [(node, j) for j in resource_graph.successors(node)]
                conservation_constraint = []

                constraint_name = f"flow_conservation_{node}"
                if constraint_name not in added_constraints:
                    if node in field_devices:
                        conservation_constraint = [[generate_edge_variable_name(edge) for edge in outgoing_edges],
                                                   [1.0] * len(outgoing_edges)]
                        rhs = 1.0
                    elif any(node in path for path in combination.values()):
                        conservation_constraint = [
                            [generate_edge_variable_name(edge) for edge in incoming_edges + outgoing_edges],
                            [1.0] * len(incoming_edges) + [-1.0] * len(outgoing_edges)
                        ]
                        rhs = 0.0
                    else:
                        conservation_constraint = [[generate_edge_variable_name(edge) for edge in incoming_edges],
                                                   [1.0] * len(incoming_edges)]
                        rhs = -1.0

                    if conservation_constraint:
                        prob.linear_constraints.add(
                            lin_expr=[conservation_constraint],
                            senses=['E'],
                            rhs=[rhs],
                            names=[constraint_name]
                        )
                        added_constraints.add(constraint_name)
            '''''''''

            for edge in reservable_capacity:
                constraint_name = f"capacity_constraint_{edge[0]}_{edge[1]}"
                add_constraint_if_new(
                    constraint_name,
                    [[generate_edge_variable_name(edge)], [reservable_capacity[edge]]],
                    'L', resource_graph[edge[0]][edge[1]]['bandwidth']
                )

            c_max_var = 'c_max'
            add_variable_if_new(c_max_var, 'C')
            for edge in reservable_capacity:
                constraint_name = f"c_max_constraint_{edge[0]}_{edge[1]}"
                add_constraint_if_new(
                    constraint_name,
                    [[c_max_var], [-1.0]],
                    'L', reservable_capacity[edge]
                )
            for device, path in combination.items():
                path_edges = list(zip(path[:-1], path[1:]))

                if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP)
                    wcd_vars = []
                    wcd_coeffs = []

                    for i, j in path_edges:
                        s_ij_var = f's_{device}_{i}_{j}'
                        x_ij_var = generate_edge_variable_name((i, j))
                        add_variable_if_new(s_ij_var, prob.variables.type.continuous)

                        # s_ij * r_ij >= 1.0
                        add_constraint_if_new(
                            f'srp_rate_constraint_{device}_{i}_{j}',
                            [[s_ij_var, f'r_{device}_{i}_{j}'], [1.0, -flow[device]['rate']]],
                            'G',
                            1.0
                        )

                        # s_ij >= 0
                        add_constraint_if_new(
                            f'srp_non_neg_s_{device}_{i}_{j}',
                            [[s_ij_var], [1.0]],
                            'G',
                            0
                        )

                        wcd_vars.extend([s_ij_var, x_ij_var])
                        w_ij = resource_graph[i][j]['bandwidth']
                        wcd_coeffs.extend([L, L / w_ij + l_ij + n_i])

                    # Worst-case delay constraint
                    t = sigma / min(flow[device]['rate'] for _ in path_edges)
                    add_constraint_if_new(
                        f'srp_wcd_{device}',
                        [wcd_vars, wcd_coeffs],
                        'L',
                        flow[device]['deadline'] - t
                    )

                elif scheduling_algorithm == 2:  # Group-Based (GB)

                        wcd_vars = []
                        wcd_coeffs = []

                        for i, j in path_edges:
                            s_ij_var = f's_{i}_{j}'
                            x_ij_var = f'x_{i}_{j}'
                            add_variable_if_new(s_ij_var, prob.variables.type.continuous)

                            # s_ij * r_ij >= 1.0
                            add_constraint_if_new(
                                f'gb_rate_constraint_{device}_{i}_{j}',
                                [[s_ij_var, f'r_{i}_{j}'], [1.0, -flow[device]['rate']]],
                                'G',
                                1.0
                            )

                            # s_ij >= 0
                            add_constraint_if_new(
                                f'gb_non_neg_s_{device}_{i}_{j}',
                                [[s_ij_var], [1.0]],
                                'G',
                                0
                            )

                            wcd_vars.extend([s_ij_var, x_ij_var])
                            w_ij = resource_graph[i][j]['bandwidth']
                            wcd_coeffs.extend([6 * L, 2 * L / w_ij + l_ij + n_i])

                        # Worst-case delay constraint
                        t = sigma / min(flow[device]['rate'] for _ in path_edges)
                        add_constraint_if_new(
                            f'gb_wcd_{device}',
                            [wcd_vars, wcd_coeffs],
                            'L',
                            flow[device]['deadline'] - t
                        )

                elif scheduling_algorithm == 3:  # Weakly Rate-Proportional (WRP)
                        wcd_vars = []
                        wcd_coeffs = []
                        delay_slack_vars = []
                        delay_slack_coeffs = []

                        for i, j in path_edges:
                            s_ij_var = f's_{i}_{j}'
                            x_ij_var = f'x_{i}_{j}'
                            add_variable_if_new(s_ij_var, prob.variables.type.continuous)

                            # s_ij * r_ij >= 1.0
                            add_constraint_if_new(
                                f'wrp_rate_constraint_{device}_{i}_{j}',
                                [[s_ij_var, f'r_{i}_{j}'], [1.0, -flow[device]['rate']]],
                                'G',
                                1.0
                            )

                            # s_ij >= 0
                            add_constraint_if_new(
                                f'wrp_non_neg_s_{device}_{i}_{j}',
                                [[s_ij_var], [1.0]],
                                'G',
                                0
                            )

                            w_ij = resource_graph[i][j]['bandwidth']
                            P = sum(1 for d in flow if any(Check_edge_inpath(p[0], i, j) for p in flow[d]['paths']))

                            wcd_vars.extend([s_ij_var, x_ij_var])
                            wcd_coeffs.extend([6 * L, 2 * L / w_ij + l_ij + n_i])

                            delay_slack_vars.append(x_ij_var)
                            delay_slack_coeffs.append(L / flow[device]['rate'] + (P - 1) * (L / w_ij) + l_ij + n_i)

                        # Worst-case delay constraint
                        t = sigma / min(flow[device]['rate'] for _ in path_edges)
                        add_constraint_if_new(
                            f'wrp_wcd_{device}',
                            [wcd_vars, wcd_coeffs],
                            'L',
                            flow[device]['deadline']
                        )

                        # Delay slack constraint
                        add_constraint_if_new(
                            f'wrp_delay_slack_{device}',
                            [delay_slack_vars, delay_slack_coeffs],
                            'L',
                            flow[device]['deadline'] - t
                        )

                elif scheduling_algorithm == 4:  # Frame-Based (FB)
                        wcd_vars = []
                        wcd_coeffs = []

                        for i, j in path_edges:
                            v_ij_var = f'v_{i}_{j}'
                            z_ij_var = f'z_{i}_{j}'
                            s_ij_var = f's_{i}_{j}'
                            x_ij_var = f'x_{i}_{j}'
                            add_variable_if_new(v_ij_var, prob.variables.type.continuous)
                            add_variable_if_new(z_ij_var, prob.variables.type.continuous)
                            add_variable_if_new(s_ij_var, prob.variables.type.continuous)

                            w_ij = resource_graph[i][j]['bandwidth']
                            r_ij = flow[device]['rate']
                            min_path_rate = min(flow[device]['rate'] for _ in path_edges)

                            # FB scheduler algorithm constraints
                            add_constraint_if_new(
                                f'fb_v_constraint1_{device}_{i}_{j}',
                                [[v_ij_var, s_ij_var], [1.0, -L]],
                                'G',
                                -L / w_ij
                            )
                            add_constraint_if_new(
                                f'fb_v_constraint2_{device}_{i}_{j}',
                                [[v_ij_var, x_ij_var], [1.0, -L / min_path_rate]],
                                'G',
                                -(L * r_ij) / (w_ij * min_path_rate)
                            )
                            add_constraint_if_new(
                                f'fb_v_non_neg_{device}_{i}_{j}',
                                [[v_ij_var], [1.0]],
                                'G',
                                0
                            )
                            add_constraint_if_new(
                                f'fb_s_non_neg_{device}_{i}_{j}',
                                [[s_ij_var], [1.0]],
                                'G',
                                0
                            )
                            add_constraint_if_new(
                                f'fb_s_constraint_{device}_{i}_{j}',
                                [[s_ij_var, x_ij_var], [r_ij, -1.0]],
                                'G',
                                0
                            )
                            add_constraint_if_new(
                                f'fb_z_constraint1_{device}_{i}_{j}',
                                [[z_ij_var], [1.0]],
                                'G',
                                1 / min_path_rate
                            )
                            add_constraint_if_new(
                                f'fb_z_constraint2_{device}_{i}_{j}',
                                [[z_ij_var, s_ij_var], [1.0, -1.0]],
                                'G',
                                0
                            )

                            P = sum(1 for d in flow if any(Check_edge_inpath(p[0], i, j) for p in flow[d]['paths']))

                            wcd_vars.extend([s_ij_var, x_ij_var, v_ij_var])
                            wcd_coeffs.extend([L, (L / w_ij) * P, 1])

                        # Worst-case delay constraint for FB
                        t = sigma / min(flow[device]['rate'] for _ in path_edges)
                        add_constraint_if_new(
                            f'fb_wcd_{device}',
                            [wcd_vars, wcd_coeffs],
                            'L',
                            flow[device]['deadline'] - t - len(path_edges) * (l_ij + n_i)
                        )
            # ... (FB constraints implementation
            prob.objective.set_linear([(c_max_var, 1.0)])
            prob.objective.set_sense(prob.objective.sense.minimize)

            prob.solve()
            #info(prob)

            if prob.solution.get_status() in [1, 101, 102]:
                valid_solutions.append((combination, prob.solution.get_objective_value()))
                valid_solution_count += 1

        print(f"Total valid solutions found: {valid_solution_count}")
        #for solution in valid_solutions:
            #print(f"Combination: {solution[0]} - Objective Value: {solution[1]}")

    except CplexError as e:
        print(f"CplexError: {e}")


solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm)
