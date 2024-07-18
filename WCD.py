
import cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import itertools
from math import prod

# Define service types with their characteristics
service_types = {
    1: {"name": "AR", "bandwidth": 2.5, "latency": 100, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 4},
    2: {"name": "CVC", "bandwidth": 3, "latency": 500, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 4},
    3: {"name": "LM", "bandwidth": 1, "latency": 1000, "processor_cores": 0.5, "Memory_GB": 0.2, "Storage_GB": 2.5},
    4: {"name": "SE", "bandwidth": 0.5, "latency": 1000, "processor_cores": 1, "Memory_GB": 2, "Storage_GB": 0.5},
    5: {"name": "VPP", "bandwidth": 2, "latency": 800, "processor_cores": 4, "Memory_GB": 8, "Storage_GB": 4}
}

# Define scheduling algorithms
scheduling_algorithms_types = {
    1: "Strictly Rate-Proportional (SRP) latency",
    2: "Group-Based (GB) approximations of WFQ",
    3: "Schedulers with Weakly Rate-Proportional (WRP) latency",
    4: "Frame-Based (FB) schedulers' latency"
}

# Global variables
P = {}  # Dictionary to store flow numbers

def define_scheduling_algorithm():
    """Prompt user to select a scheduling algorithm"""
    for i, algorithm in scheduling_algorithms_types.items():
        print(f"{i}: {algorithm}")
    return int(input("Enter number of the used node scheduling algorithm for WCD calculation: "))

def generate_edge_variable_name(edge):
    """Generate consistent variable names for edges"""
    return 'x_{}_{}'.format(*sorted(edge))

def flow_number(i, j, field_devices, flow):
    """Calculate the flow number for an edge"""
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
    """Check if an edge is in a path"""
    for x in range(len(path) - 1):
        if [path[x], path[x + 1]] == [i, j] or [path[x], path[x + 1]] == [j, i]:
           return True
    return False

def calculate_total_combinations(flow):
    """Calculate the total number of path combinations"""
    path_counts = [len(flow[device]['paths']) for device in flow]
    return prod(path_counts)

def generate_path_combinations(flow):
    """Generate all possible path combinations"""
    device_paths = {device: [tuple(path[0]) for path in flow[device]['paths']] for device in flow}
    devices = list(device_paths.keys())

    def combinations_generator():
        for combination in itertools.product(*device_paths.values()):
            yield dict(zip(devices, combination))

    return combinations_generator()

def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm=1, sigma=1500, rho=1500):
    """Main function to solve the optimal path problem"""
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

        # Constants
        tau = 1
        L = 1500  # Maximum packet size
        l_ij = 0.0005  # Link propagation delay
        n_i = 40 / 1000  # Node processing delay

        # Initialize flow dictionary
        flow = {device: {
            'paths': paths[device],
            'deadline': field_devices_delta[device],
            'burst': sigma,
            'rate': rho,
            'reserved_rates': {rho * tau + sigma}
        } for device in field_devices_delta}

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

            # Initialize capacity variables
            remaining_capacity = {}
            reservable_capacity = {}
            for device, path in combination.items():
                for i, j in zip(path[:-1], path[1:]):
                    edge = (i, j)
                    if edge not in reservable_capacity:
                        remaining_capacity[edge] = resource_graph[i][j]['bandwidth']
                        reservable_capacity[edge] = 0

            # Add variables and constraints for each device and path
            for device, path in combination.items():
                for i, j in zip(path[:-1], path[1:]):
                    edge = (i, j)
                    var_name = generate_edge_variable_name(edge)
                    add_variable_if_new(var_name, 'B')

                    flow_rate_var = f"r_{device}_{i}_{j}"
                    add_variable_if_new(flow_rate_var, 'C')

                    # Constraint: flow rate <= reservable capacity
                    add_constraint_if_new(
                        f"flow_rate_constraint_{device}_{i}_{j}",
                        [[var_name, flow_rate_var], [reservable_capacity[edge], -1.0]],
                        'L', 0.0
                    )

                    # Constraint: minimum flow rate
                    r_min = f"r_min_{device}_{i}_{j}"
                    add_variable_if_new(r_min, 'C')
                    add_constraint_if_new(
                        f"min_flow_rate_constraint_{device}_{i}_{j}",
                        [[var_name, flow_rate_var, r_min], [reservable_capacity[edge], -1.0, -rho]],
                        'L', 0.0
                    )

                    reservable_capacity[edge] += rho * tau + sigma
                    remaining_capacity[edge] -= reservable_capacity[edge]

            # Capacity constraints
            for edge in reservable_capacity:
                add_constraint_if_new(
                    f"capacity_constraint_{edge[0]}_{edge[1]}",
                    [[generate_edge_variable_name(edge)], [reservable_capacity[edge]]],
                    'L', resource_graph[edge[0]][edge[1]]['bandwidth']
                )

            # Maximum capacity variable
            c_max_var = 'c_max'
            add_variable_if_new(c_max_var, 'C')
            for edge in reservable_capacity:
                add_constraint_if_new(
                    f"c_max_constraint_{edge[0]}_{edge[1]}",
                    [[c_max_var], [-1.0]],
                    'L', reservable_capacity[edge]
                )
            '''''''''
            # Add flow-specific constraints
            for device in flow:
                for path in flow[device]['paths']:
                    path_nodes = path[0]
                    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

                    # Flow conservation constraints
                    for node in path_nodes:
                        incoming = sum(1 for prev_node in path_nodes if (prev_node, node) in resource_graph.edges)
                        outgoing = sum(1 for next_node in path_nodes if (node, next_node) in resource_graph.edges)

                        if node == path_nodes[0]:  # Source node
                            rhs = 1
                        elif node == path_nodes[-1]:  # Destination node
                            rhs = -1
                        else:  # Intermediate node
                            rhs = 0

                        add_constraint_if_new(
                            f'flow_conservation_{device}_{node}',
                            [[f'x_{prev}_{node}' for prev in path_nodes if (prev, node) in resource_graph.edges] +
                             [f'x_{node}_{next}' for next in path_nodes if (node, next) in resource_graph.edges],
                             [1.0] * incoming + [-1.0] * outgoing],
                            'E',
                            rhs
                        )

             '''''''''
            # Add scheduling algorithm specific constraints
            if scheduling_algorithm == 1:
                # Strictly Rate-Proportional (SRP) scheduling algorithm
                for device, path in combination.items():
                    wcd_expr = []
                    t_var = f"t_{device}"
                    add_variable_if_new(t_var, 'C')

                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        s_var = f"s_{device}_{i}_{j}"
                        r_var = f"r_{device}_{i}_{j}"
                        add_variable_if_new(s_var, 'C')

                        # WCD expression
                        wcd_expr.extend([(s_var, L), (x_var, L / reservable_capacity[edge] + l_ij + n_i)])

                        # Linearized constraint for: sij rij ≥ xij^2
                        M = 1e6  # A large constant
                        y_var = f"y_{device}_{i}_{j}"
                        add_variable_if_new(y_var, 'B')

                        # sij ≥ xij^2 - M(1-y)
                        add_constraint_if_new(
                            f"srp_constraint1_{device}_{i}_{j}",
                            [[s_var, x_var, y_var], [1.0, -2.0, M]],
                            'G',
                            -M
                        )

                        # rij ≥ xij^2 - My
                        add_constraint_if_new(
                            f"srp_constraint2_{device}_{i}_{j}",
                            [[r_var, x_var, y_var], [1.0, -2.0, -M]],
                            'G',
                            0
                        )

                    # WCD constraint
                    add_constraint_if_new(
                        f"wcd_constraint_{device}",
                        [[t_var] + [var for var, _ in wcd_expr], [1.0] + [-1.0] * len(wcd_expr)],
                        'G',
                        -field_devices_delta[device]
                    )

                    # Constraint: t rmin ≥ σ
                    add_constraint_if_new(
                        f"t_rmin_constraint_{device}",
                        [[t_var, f"r_min_{device}_{path[0]}_{path[1]}"], [1.0, -sigma]],
                        'G',
                        0.0
                    )

            elif scheduling_algorithm == 2:
                # Group-Based (GB) scheduling algorithm
                for device, path in combination.items():
                    wcd_expr = []
                    t_var = f"t_{device}"
                    add_variable_if_new(t_var, 'C')

                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        s_var = f"s_{device}_{i}_{j}"
                        r_var = f"r_{device}_{i}_{j}"
                        add_variable_if_new(s_var, 'C')

                        # WCD expression
                        wcd_expr.extend([(s_var, L), (x_var, L / reservable_capacity[edge] + l_ij + n_i)])

                        # Linearized constraint for: sij rij ≥ xij^2
                        M = 1e6  # A large constant
                        y_var = f"y_{device}_{i}_{j}"
                        add_variable_if_new(y_var, 'B')

                        # sij ≥ xij^2 - M(1-y)
                        add_constraint_if_new(
                            f"gb_constraint1_{device}_{i}_{j}",
                            [[s_var, x_var, y_var], [1.0, -2.0, M]],
                            'G',
                            -M
                        )

                        # rij ≥ xij^2 - My
                        add_constraint_if_new(
                            f"gb_constraint2_{device}_{i}_{j}",
                            [[r_var, x_var, y_var], [1.0, -2.0, -M]],
                            'G',
                            0
                        )

                    # WCD constraint
                    add_constraint_if_new(
                        f"wcd_constraint_{device}",
                        [[t_var] + [var for var, _ in wcd_expr], [1.0] + [-1.0] * len(wcd_expr)],
                        'G',
                        -field_devices_delta[device]
                    )

            elif scheduling_algorithm == 3:
                # Weakly Rate-Proportional (WRP) algorithm
                for device, path in combination.items():
                    wcd_expr = []
                    t_var = f"t_{device}"
                    add_variable_if_new(t_var, 'C')

                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        s_var = f"s_{device}_{i}_{j}"
                        r_var = f"r_{device}_{i}_{j}"
                        add_variable_if_new(s_var, 'C')

                        # WCD expression
                        wcd_expr.extend([(s_var, L), (x_var, L / reservable_capacity[edge] * flow_number(i, j, field_devices_delta, flow) + l_ij + n_i)])

                        # Linearized constraint for: sij rij ≥ xij^2
                        M = 1e6  # A large constant
                        y_var = f"y_{device}_{i}_{j}"
                        add_variable_if_new(y_var, 'B')

                        # sij ≥ xij^2 - M(1-y)
                        add_constraint_if_new(
                            f"wrp_constraint1_{device}_{i}_{j}",
                            [[s_var, x_var, y_var], [1.0, -2.0, M]],
                            'G',
                            -M
                        )

                        # rij ≥ xij^2 - My
                        add_constraint_if_new(
                            f"wrp_constraint2_{device}_{i}_{j}",
                            [[r_var, x_var, y_var], [1.0, -2.0, -M]],
                            'G',
                            0
                        )

                    # WCD constraint
                    add_constraint_if_new(
                        f"wcd_constraint_{device}",
                        [[t_var] + [var for var, _ in wcd_expr], [1.0] + [-1.0] * len(wcd_expr)],
                        'G',
                        -field_devices_delta[device]
                    )

                    # Constraint: t = σ / min{rij: (i,j) ∈ p}
                    for i, j in zip(path[:-1], path[1:]):
                        r_var = f"r_{device}_{i}_{j}"
                        add_constraint_if_new(
                            f"t_constraint_{device}_{i}_{j}",
                            [[t_var, r_var], [1.0, -sigma]],
                            'G',
                            0.0
                        )

            elif scheduling_algorithm == 4:
                # Frame-Based (FB) scheduler algorithm
                for device, path in combination.items():
                    wcd_expr = []
                    t_var = f"t_{device}"
                    add_variable_if_new(t_var, 'C')

                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        s_var = f"s_{device}_{i}_{j}"
                        r_var = f"r_{device}_{i}_{j}"
                        v_var = f"v_{device}_{i}_{j}"
                        z_var = f"z_{device}_{i}_{j}"
                        add_variable_if_new(s_var, 'C')
                        add_variable_if_new(v_var, 'C')
                        add_variable_if_new(z_var, 'C')

                        # WCD expression
                        wcd_expr.extend(
                            [(s_var, L * L / reservable_capacity[edge] * flow_number(i, j, field_devices_delta, flow)),
                             (x_var, 0), (v_var, 1)])

                        # Constraints for vij
                        add_constraint_if_new(
                            f"v_constraint1_{device}_{i}_{j}",
                            [[v_var, s_var], [1.0, -1.0]],
                            'G',
                            -L / reservable_capacity[edge]
                        )
                        add_constraint_if_new(
                            f"v_constraint2_{device}_{i}_{j}",
                            [[v_var, x_var, r_var], [1.0, -L / rho, L / (reservable_capacity[edge] * rho)]],
                            'G',
                            0.0
                        )

                        # Linearized constraint for: sij rij ≥ xij^2
                        M = 1e6  # A large constant
                        y_var = f"y_{device}_{i}_{j}"
                        add_variable_if_new(y_var, 'B')

                        # sij ≥ xij^2 - M(1-y)
                        add_constraint_if_new(
                            f"fb_constraint1_{device}_{i}_{j}",
                            [[s_var, x_var, y_var], [1.0, -2.0, M]],
                            'G',
                            -M
                        )

                        # rij ≥ xij^2 - My
                        add_constraint_if_new(
                            f"fb_constraint2_{device}_{i}_{j}",
                            [[r_var, x_var, y_var], [1.0, -2.0, -M]],
                            'G',
                            0
                        )

                        # Constraints for zij
                        add_constraint_if_new(
                            f"z_constraint1_{device}_{i}_{j}",
                            [[z_var], [1.0]],
                            'G',
                            1 / rho
                        )
                        add_constraint_if_new(
                            f"z_constraint2_{device}_{i}_{j}",
                            [[z_var, s_var], [1.0, -1.0]],
                            'G',
                            0.0
                        )

                    # WCD constraint
                    add_constraint_if_new(
                        f"wcd_constraint_{device}",
                        [[t_var] + [var for var, _ in wcd_expr], [1.0] + [-1.0] * len(wcd_expr)],
                        'G',
                        -field_devices_delta[device]
                    )

                    # Constraint: t = σ / min{rij: (i,j) ∈ p}
                    for i, j in zip(path[:-1], path[1:]):
                        r_var = f"r_{device}_{i}_{j}"
                        add_constraint_if_new(
                            f"t_constraint_{device}_{i}_{j}",
                            [[t_var, r_var], [1.0, -sigma]],
                            'G',
                            0.0
                        )

                    # Admission control constraint
                    admission_expr = []
                    for i, j in zip(path[:-1], path[1:]):
                        x_var = generate_edge_variable_name((i, j))
                        z_var = f"z_{device}_{i}_{j}"
                        admission_expr.extend([x_var, z_var])

                    add_constraint_if_new(
                        f"admission_constraint_{device}",
                        [admission_expr, [L / reservable_capacity[(i, j)] for i, j in zip(path[:-1], path[1:])] +
                         [L - L * rho / reservable_capacity[(i, j)] for i, j in zip(path[:-1], path[1:])]],
                        'L',
                        field_devices_delta[device] - (
                                sigma / min([reservable_capacity[edge] for edge in zip(path[:-1], path[1:])]))
                    )

            # Set objective function
            prob.objective.set_linear([(c_max_var, 1.0)])
            prob.objective.set_sense(prob.objective.sense.minimize)

            # Solve the problem
            prob.solve()

            # Check if a valid solution was found
            if prob.solution.get_status() in [1, 101, 102]:
                valid_solutions.append((combination, prob.solution.get_objective_value()))
                valid_solution_count += 1

        print(f"Total valid solutions found: {valid_solution_count}")
        #for solution in valid_solutions:
            #print(f"Combination: {solution[0]} - Objective Value: {solution[1]}")

    except CplexError as e:
        print(f"CplexError: {e}")


# Main execution
if __name__ == "__main__":
    # Load resource graph and paths
    resource_graph, paths = sorted_paths()

    # Convert bandwidth to proper units
    for i, j in resource_graph.edges:
        resource_graph[i][j]['bandwidth'] = resource_graph[i][j]['bandwidth'] * 1000

    # Get field devices and their deltas
    field_devices = {node: resource_graph.nodes[node] for node in resource_graph.nodes if
                     resource_graph.nodes[node]['type'] == 'F'}
    field_devices_delta = {}
    for device, attributes in field_devices.items():
        service_type = attributes['service_type']
        try:
            service_type_key = next(key for key, value in service_types.items() if value['name'] == service_type)
        except StopIteration:
            print(f"Service type {service_type} not found in service_types dictionary.")
            continue
        delta = service_types[service_type_key]['latency']
        field_devices_delta[device] = delta

    # Get scheduling algorithm from user
    scheduling_algorithm = define_scheduling_algorithm()

    # Solve the optimal path problem
    solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm)

'''''''''

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

'''''''''
