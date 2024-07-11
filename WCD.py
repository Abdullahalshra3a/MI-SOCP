import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice, product
import itertools
from math import prod
import re

def is_valid_variable_name(name):
    # Variable names must start with a letter or underscore and contain only alphanumeric characters and underscores
    return re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) is not None

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

    # Get more detailed information about variables
    # for i in range(num_variables):
    # var_name = prob.variables.get_names(i)
    # var_type = prob.variables.get_types(i)
    # print(f"Variable {i}: Name = {var_name}, Type = {var_type}")

    # Get more detailed information about constraints
    # for i in range(num_constraints):
    # con_name = prob.linear_constraints.get_names(i)
    # con_sense = prob.linear_constraints.get_senses(i)
    # con_rhs = prob.linear_constraints.get_rhs(i)
    # print(f"Constraint {i}: Name = {con_name}, Sense = {con_sense}, RHS = {con_rhs}")

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

def log_infeasible_constraints(prob):
    infeasible_constraints = []
    try:
        # Refine conflicts and get the conflicting indices
        prob.conflict.refine(prob.conflict.all_constraints())
        conflicts = prob.conflict.get()

        # Iterate over the conflict indices
        for conflict_type, index in zip(conflicts[0], conflicts[1]):
            if conflict_type == prob.conflict.constraint_type.linear:
                constraint_name = prob.linear_constraints.get_names(index)
                row = prob.linear_constraints.get_rows(index)
                lhs = sum(prob.solution.get_values(ind) * val for ind, val in zip(row.ind, row.val))
                rhs = prob.linear_constraints.get_rhs(index)
                sense = prob.linear_constraints.get_senses(index)
                infeasible_constraints.append((constraint_name, lhs, sense, rhs))
            elif conflict_type == prob.conflict.constraint_type.lower_bound:
                var_name = prob.variables.get_names(index)
                lb = prob.variables.get_lower_bounds(index)
                infeasible_constraints.append((f'Lower bound on {var_name}', lb, 'L', lb))
            elif conflict_type == prob.conflict.constraint_type.upper_bound:
                var_name = prob.variables.get_names(index)
                ub = prob.variables.get_upper_bounds(index)
                infeasible_constraints.append((f'Upper bound on {var_name}', ub, 'U', ub))

    except cplex.exceptions.CplexError as e:
        print(f"CPLEX Error during conflict refinement: {e}")
    except IndexError as e:
        print(f"IndexError during conflict refinement: {e}")
    except Exception as e:
        print(f"Unexpected error during conflict refinement: {e}")
    return infeasible_constraints


def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm=1, sigma=1500, rho=1500):
    try:
        # Ensure the graph is directed
        if not isinstance(resource_graph, nx.DiGraph):
            resource_graph = nx.DiGraph(resource_graph)

        # Create a CPLEX problem
        prob = cplex.Cplex()

        # Set the problem type to MIQCP
        prob.set_problem_type(prob.problem_type.MIQCP)

        # Initialize sets to keep track of added variables and constraints
        added_variables = set()
        added_constraints = set()

        # Helper function to add a variable if it hasn't been added before
        def add_variable_if_new(name, var_type):
            if name not in added_variables:
                prob.variables.add(names=[name], types=[var_type])
                added_variables.add(name)

        # Helper function to add a constraint if it hasn't been added before
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
        tau = 1  # Time needed to receive 1 packet in the queue (ms)
        L = 1500  # MTU fixed to 1500 bytes
        l_ij = 0.0005  # Propagation delay in ms
        n_i = 40 / 1000  # Node delay in ms

        # Create flow dictionary
        flow = {}
        for device in field_devices_delta:
            flow[device] = {
                'paths': paths[device],
                'deadline': field_devices_delta[device],
                'burst': sigma,
                'rate': rho,
                'reserved_rates': {}
            }

        # Calculate total number of combinations
        total_combinations = calculate_total_combinations(flow)
        print(f"Total number of combinations: {total_combinations}")

        # Generate combinations generator
        combinations_generator = generate_path_combinations(flow)

        # Process the first 100 combinations
        first_100_combinations = list(itertools.islice(combinations_generator, 100))

        # Initialize storage for results
        valid_solutions = []
        valid_solution_count = 0

        for combination in first_100_combinations:
            prob = cplex.Cplex()
            prob.set_problem_type(prob.problem_type.MIQCP)
            added_variables = set()
            added_constraints = set()

            # Add variables and basic constraints for all edges
            for i, j in resource_graph.edges:
                add_variable_if_new(f'x_{i}_{j}', prob.variables.type.binary)
                add_variable_if_new(f'r_{i}_{j}', prob.variables.type.continuous)
                add_variable_if_new(f'r_min', prob.variables.type.continuous)
                add_variable_if_new(f'c_ij_{i}_{j}', prob.variables.type.continuous)

                # Basic constraints
                add_constraint_if_new(f'non_neg_r_{i}_{j}', [[f'r_{i}_{j}'], [1.0]], 'G', 0)
                add_constraint_if_new(f'non_neg_r_min', [[f'r_min'], [1.0]], 'G', rho)
                add_constraint_if_new(f'non_neg_c_ij_{i}_{j}', [[f'c_ij_{i}_{j}'], [1.0]], 'G', 0)

                # Edge bandwidth constraint
                w_ij = resource_graph[i][j]['bandwidth']
                r_bar_ij = sum(flow[k]['reserved_rates'].get((i, j), 0) for k in flow)
                add_constraint_if_new(f'edge_bandwidth_{i}_{j}',
                                      [[f'r_{i}_{j}', f'c_ij_{i}_{j}'], [1.0, 1.0]],
                                      'L',
                                      w_ij - r_bar_ij)

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

                    # Scheduling algorithm specific constraints
                    if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP)
                        wcd_vars = []
                        wcd_coeffs = []

                        for i, j in path_edges:
                            s_ij_var = f's_{i}_{j}'
                            x_ij_var = f'x_{i}_{j}'
                            add_variable_if_new(s_ij_var, prob.variables.type.continuous)

                            # s_ij * r_ij >= 1.0
                            add_constraint_if_new(
                                f'srp_rate_constraint_{device}_{i}_{j}',
                                [[s_ij_var, f'r_{i}_{j}'], [1.0, -flow[device]['rate']]],
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

            # Set the objective function
            objective_dict = {}

            for i, j in resource_graph.edges:
                r_ij_var = f'r_{i}_{j}'
                x_ij_var = f'x_{i}_{j}'

                # Calculate the total potential reserved rate for this edge
                total_potential_rate = sum(
                    flow[device]['rate']
                    for device in field_devices_delta
                    if any(Check_edge_inpath(path[0], i, j) for path in flow[device]['paths'])
                )

                # Add the rate variable to the objective, weighted by the potential reserved rate
                if r_ij_var not in objective_dict:
                    objective_dict[r_ij_var] = 0
                objective_dict[r_ij_var] += total_potential_rate

                # Add a small penalty for using the edge to prefer shorter paths
                if x_ij_var not in objective_dict:
                    objective_dict[x_ij_var] = 0
                objective_dict[x_ij_var] += 0.1  # Small penalty for using an edge

            # Convert the dictionary to the format required by CPLEX
            objective = [(var, cost) for var, cost in objective_dict.items()]

            # Set the objective function to minimize the weighted amount of allocated rate
            prob.objective.set_sense(prob.objective.sense.minimize)
            prob.objective.set_linear(objective)

            # Solve the problem with exception handling
            try:
                prob.solve()
                status = prob.solution.get_status()
                print('526', status)

                if status == 103:  # MIP Infeasible or Unbounded
                    # Check if the problem is infeasible or unbounded
                    if "infeasible" in prob.solution.get_status_string():
                        print("The problem is infeasible.")
                    elif "unbounded" in prob.solution.get_status_string():
                        print("The problem is unbounded.")
                    else:
                        print("The problem is infeasible or unbounded.")

                    # Attempt conflict refinement to identify the infeasibility
                    infeasible_constraints = log_infeasible_constraints(prob)
                    print("Conflict set:")
                    for name, lhs, sense, rhs in infeasible_constraints:
                        print(f"Constraint {name}: LHS = {lhs}, Sense = {sense}, RHS = {rhs}")
                    continue  # Skip this combination and move to the next one

                # Extract and store the solution if it's valid
                if status == prob.solution.status.optimal:
                    print('538')
                    solution = {
                        'objective_value': prob.solution.get_objective_value(),
                        'paths': {},
                        'rates': {}
                    }

                    for device in flow:
                        solution['paths'][device] = []
                        solution['rates'][device] = {}
                        for path in flow[device]['paths']:
                            path_nodes = path[0]
                            path_links = list(zip(path_nodes[:-1], path_nodes[1:]))
                            if all(prob.solution.get_values(f'x_{i}_{j}') > 0.5 for i, j in path_links):
                                solution['paths'][device].append(path)
                                for i, j in path_links:
                                    solution['rates'][device][(i, j)] = prob.solution.get_values(f'r_{i}_{j}')

                    valid_solutions.append(solution)
                    valid_solution_count += 1

            except cplex.exceptions.CplexError as exc:
                print(f"An exception occurred: {exc}")
            except IndexError as e:
                print(f"IndexError occurred: {e}")

        return valid_solutions

    except cplex.exceptions.CplexError as exc:
        print(f"An exception occurred: {exc}")
        return None


# if __name__ == "__main__":
result = solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm, 1500, 1500)
print(result)
