

import cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import numpy as np
import itertools
from math import prod
import time
import random
from collections import Counter
from py4j_adapter.DiscoDNC_adapter import javNC_validateNetwork

prio_HIGH = 0
prio_MED = 1
prio_LOW = 2
prio_dict = {'AR': prio_HIGH, 'AR2': prio_HIGH, 'CVC': prio_HIGH, 'CVC2': prio_HIGH,
             'LM': prio_LOW, 'LM2': prio_LOW, 'SE': prio_MED, 'SE2': prio_MED,
             'VPP': prio_LOW, 'VPP2': prio_MED}  # Has to be gathered from real program later

pickle_qos_dict = {'AR': 100.0, 'AR2': 100.0, 'CVC': 500.0, 'LM': 1000.0, 'SE': 1000.0,
                   'VPP': 800.0, 'CVC2': 500.0, 'LM2': 1000.0, 'SE2': 1000.0,
                   'VPP2': 800.0}  # Has to be gathered from real program later


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


def flow_number(node1, node2, combination):
    """Calculate the flow number for an edge"""
    count = 0

    for device, path in combination.items():
        for i in range(len(path) - 1):
            if path[i] == node1 and path[i + 1] == node2:
                count += 1
                break  # Move to the next path once we find a match

    return count


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


def generate_path_combinations(flow, max_combinations=100000, max_paths_per_device=5):
    """Generate a sample of path combinations prioritizing variety and shorter paths"""
    device_paths = {}
    for device in flow:
        # Sort paths by length (shortest first) and take the top max_paths_per_device
        paths = sorted([tuple(path[0]) for path in flow[device]['paths']], key=len)[:max_paths_per_device]
        device_paths[device] = paths

    devices = list(device_paths.keys())

    def sample_combinations():
        seen_combinations = set()
        attempts = 0
        max_attempts = max_combinations * 10  # Limit total attempts to avoid infinite loop

        while len(seen_combinations) < max_combinations and attempts < max_attempts:
            # Generate a random combination
            combination = tuple(random.choice(paths) for paths in device_paths.values())

            if combination not in seen_combinations:
                seen_combinations.add(combination)
                yield dict(zip(devices, combination))

            attempts += 1

        print(f"Generated {len(seen_combinations)} unique combinations")

    # Sort the sampled combinations by total path length
    sampled_combinations = sorted(sample_combinations(), key=lambda x: sum(len(path) for path in x.values()))

    return sampled_combinations


def analyze_edge_usage(combinations):
    """
    Analyze the usage of edges across all paths in the given combinations.

    Args:
    combinations (list): A list of dictionaries, where each dictionary represents a combination
                         of paths for different devices.

    Returns:
    list: A list of tuples (edge, count), sorted by count in descending order.
    """
    edge_counter = Counter()


    for path in combinations.values():
            # Create edges from consecutive nodes in the path
            edges = list(zip(path[:-1], path[1:]))
            edge_counter.update(edges)

    # Sort edges by usage count in descending order
    sorted_edges = sorted(edge_counter.items(), key=lambda x: x[1], reverse=True)

    return sorted_edges


def pretty_print_constraint(name, sparse_pair:cplex.SparsePair, sense:str, rhs:float, sparse_triple:cplex.SparseTriple = None)-> str:
    multiplications = []
    if sparse_pair is not None:
        multiplications.extend([f'({val}*{ind})' for (ind, val) in zip(sparse_pair.ind, sparse_pair.val)])

    if sparse_triple is not None:
        multiplications.extend([f'({val}*{ind1}*{ind2})' for (ind1, ind2, val) in zip(sparse_triple.ind1, sparse_triple.ind2, sparse_triple.val)])

    lhs = ' + '.join(multiplications)
    senses = {
        'L': '<=', 'G': '>=', 'E': '=='
    }
    return f'{name:20}: {lhs} {senses[sense]} {rhs}'



def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm=1, sigma=255, rho=255):
    """Main function to solve the optimal path problem"""
    prob = None

    if not isinstance(resource_graph, nx.DiGraph):
        resource_graph = nx.DiGraph(resource_graph)

    def add_variable_if_new(name, var_type):
        if name not in added_variables:
            prob.variables.add(names=[name], types=[var_type])
            added_variables[name] = var_type
        else:
            #print(f'The variable {name}, {var_type} has already been added!')
            assert added_variables[name] == var_type, 'The variable type did not match the previous definition!'

    def add_constraint_if_new(name, sense, rhs, lin_expr=None, quad_expr=None):

        if name not in added_constraints:
            #print(f'Adding constraint {name}:')
            if quad_expr is None:
                prob.linear_constraints.add(
                    lin_expr=[lin_expr],
                    senses=[sense],
                    rhs=[rhs],
                    names=[name]
                )
                print(pretty_print_constraint(name, lin_expr, sense, rhs))
            else:
                prob.quadratic_constraints.add(
                    lin_expr=lin_expr,
                    quad_expr=quad_expr,
                    sense=sense,
                    rhs=rhs,
                    name=name
                )
                print(pretty_print_constraint(name, lin_expr, sense, rhs, quad_expr))

            added_constraints[name] = [lin_expr, quad_expr, sense, rhs]
        else:
            pass
            print(f'The constraint {name} has already been added!')
            #print(added_constraints[name])

    # Constants
    tau = 1
    L = 255  # Maximum packet size in Byte
    l_ij = 0.0005  # Link propagation delay in ms
    n_i = 0.04  # Node processing delay in ms

    # Initialize flows dictionary
    flows = {device: {
        'paths': paths[device],
        'deadline': field_devices_delta[device],
        'burst': sigma,
        'rate': rho,#arrival
        'reserved_rates': rho * tau + sigma   # Changed from set to direct value
    } for device in field_devices_delta}

    total_combinations = calculate_total_combinations(flows)
    print(f"Total number of combinations: {total_combinations}")

    # Usage
    combinations_generator = generate_path_combinations(flows)

    # Get all combinations
    #all_combinations = list(combinations_generator)

    all_combinations = list(itertools.islice(combinations_generator, 1000))
    random.shuffle(all_combinations)
    Random_1000_combinations = all_combinations[:1000]

    valid_solutions = []
    valid_solution_count = 0
    wcd_values = {}
    # Initialize the reservation_costs dictionary
    reservation_costs = {}

    # Initialize the variables dictionary
    variables = {}

    # Iterate through the first 100 combinations
    for combination in Random_1000_combinations:

        prob = cplex.Cplex()
        prob.set_problem_type(prob.problem_type.MIQCP)
        added_variables = {}

        added_constraints = {}

        # Initialize capacity variables
        remaining_capacity = {}
        reservable_capacity = {}

        # Initialize variables for minimum reserved flows and maximum flows
        rmin = {}
        cmax = float('-inf')
        flows_over_edge = {}

        # Process all devices and paths in a single loop
        for flow_name, path in combination.items():
            current_reserved_rate = flows[flow_name]['reserved_rates']

            # Update cmax for the entire network
            if current_reserved_rate > cmax:
                cmax = current_reserved_rate

            path_min_value = float('inf')  # Initialize path minimum value

            # Process each edge in the path
            for i, j in zip(path[:-1], path[1:]):
                edge = (i, j)
                var_x = f'x_{flow_name}_{i}_{j}'  # JF: I think the edge variables have to be defined per flows. Otherwise it is not possible to construct the flows conservation constraint
                if (i, j) not in flows_over_edge:
                    flows_over_edge[(i, j)] = []

                flows_over_edge[(i, j)].append(flow_name)

                var_flow_rate = f"r_{flow_name}_{i}_{j}"
                var_edge_capacity = f"c_{i}_{j}"

                # Add variables
                add_variable_if_new(var_x, 'B')
                add_variable_if_new(var_flow_rate, 'C')
                add_variable_if_new(var_edge_capacity, 'C')

                # Initialize capacities and rmin if not already done
                if edge not in reservable_capacity:
                    remaining_capacity[edge] = resource_graph[i][j]['bandwidth']
                    reservable_capacity[edge] = 0
                    rmin[edge] = float('inf')

                # Update rijmin for the current edge
                if current_reserved_rate < rmin[edge]:
                    rmin[edge] = current_reserved_rate

                # Update path_rmin_value
                path_min_value = min(path_min_value, rmin[edge])
                # Update reservable capacity and remaining capacity
                reservable_capacity[edge] += current_reserved_rate
                remaining_capacity[edge] = resource_graph[i][j]['bandwidth'] - reservable_capacity[edge]

                reservation_costs[edge] = reservable_capacity[edge]
                variables[edge] =  var_edge_capacity


            # Store the minimum value for the entire path
            flows[flow_name]['path_min_value'] = path_min_value

            # JF: This part is not yet clear? Why is the edge_capacity a variable?
            # The edge capacity = the edge bandwidth (wij)
            add_constraint_if_new(
                name=f"cap_{i}_{j}",
                lin_expr=cplex.SparsePair([var_edge_capacity], [1.0]),
                sense='E',
                rhs=remaining_capacity[edge]
            )

            # Constraint: flows rate(rij) <= remaining capacity
            add_constraint_if_new(
                name=f"flow_cap_{flow_name}_{i}_{j}",
                lin_expr=cplex.SparsePair([var_flow_rate], [1.0]),
                sense='L',
                rhs=remaining_capacity[edge]
            )

            # Equ. (2)
            # Constraint: rij >= ρ
            add_constraint_if_new(
                name=f"rho_{flow_name}_{i}_{j}",
                lin_expr=cplex.SparsePair([var_flow_rate], [1.0]),
                sense='G',
                rhs=rho
            )

            # Constraint: flows rate = flows[flow_name]['rates']
            add_constraint_if_new(
                name=f"flow_eq_{flow_name}_{i}_{j}",
                lin_expr=cplex.SparsePair([var_flow_rate], [1.0]),
                sense='E',
                rhs=flows[device]['reserved_rates']
            )


        # Output rmin for debugging
        #for edge, min_value in rmin.items():
            #print(f"Minimum reserved flows for edge {edge}: {min_value}")

        # Output cmax for debugging
        #print(f"Maximum flows value in the network: {cmax}")

        # Equ. (11) c_max
        # Maximum capacity variable
        print('\nAdding constraint for c_max')
        var_c_max = 'c_max'
        add_variable_if_new(var_c_max, 'C')
        add_constraint_if_new(
            name=f"c_max",
            lin_expr=cplex.SparsePair([var_c_max], [1.0]),
            sense='E',
            rhs=cmax
        )

        def add_constraint_9(flow_paths, flows_over_edge):
            print('\nAdding constraints for flow conservation:')
            used_nodes = []
            for path in flow_paths:
                used_nodes.extend(path)
            used_nodes = np.unique(used_nodes)

            # Equ. (9)
            # Add flows-specific constraints
            for node in used_nodes:
                outgoing = []
                incoming = []
                #print(f'Setting up conservation constraints for the node {node}')
                for edge, flow_names in flows_over_edge.items():
                    if node in edge:
                        var_x_list = [f'x_{flow_name}_{edge[0]}_{edge[1]}' for flow_name in flow_names]
                        if node == edge[0]:
                            outgoing.extend(var_x_list)
                        if node == edge[1]:
                            incoming.extend(var_x_list)

                #print(f'{outgoing=}')
                #print(f'{incoming=}')
                # Determine the right-hand side value
                rhs = len(incoming) - len(outgoing)
                #print(f'balance = {rhs}')

                add_constraint_if_new(
                    name=f'flow_conserve_{node}',
                    lin_expr=cplex.SparsePair([*incoming, *outgoing],
                                              [*[1.0] * len(incoming), *[-1.0] * len(outgoing)]),
                    sense='E',
                    rhs=rhs
                )

        add_constraint_9(combination.values(), flows_over_edge)

        def add_constraint_10_and_11(flow_name, path, reservable_capacity):
            print(f'\nAdding constraints 10 and 11 for flow {flow_name}:')
            for i,j in reservable_capacity:
                if (i, j) in zip(path[:-1], path[1:]):
                    var_flow_rate = f"r_{flow_name}_{i}_{j}"
                    add_variable_if_new(var_flow_rate, 'C')
                    var_x = f'x_{flow_name}_{i}_{j}'

                    # Left part of Equ. (10): rij >= 0
                    add_constraint_if_new(
                        name=f"flow_lb_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_flow_rate], [1.0]),
                        sense='G',
                        rhs=0.0
                    )
                    # Right part of Equ. (10): rij <= cij * xij
                    # The constraint you've created is: var_flow_rate−cmax×var_x≤0. This can be rearranged to: var_flow_rate≤cmax×var_x
                    #   -> rij - cij * xij <= 0
                    add_constraint_if_new(
                        name=f"flow_ub_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_flow_rate, var_x], [1.0, -cmax]),
                        sense='L',
                        rhs=0.0
                    )

                    # Constraint: ρ <= rmin <= rij + cmax(1 - xij)
                    # Defining var_r_min variable
                    var_r_min = f"rmin_{flow_name}_{i}_{j}"
                    add_variable_if_new(var_r_min, 'C')

                    # Left part of Equ. (11): ρ <= r_min
                    add_constraint_if_new(
                        name=f"rmin_lb_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_r_min], [1.0]),
                        sense='G',
                        rhs=rho
                    )

                    # Right part of Equ. (11): r_min <= r_ij + c_max * (1 - x_ij)
                    #  -> r_min <= r_ij + c_max - c_max*x_ij
                    #  -> r_min + c_max*x_ij - r_ij <= c_max
                    add_constraint_if_new(
                        name=f"rmin_ub_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_r_min, var_x, var_flow_rate], [1.0, cmax, -1.0]),
                        sense='L',
                        rhs=cmax
                    )

        # Additional constraints
        for flow_name, path in combination.items():
            add_constraint_10_and_11(flow_name, path, reservable_capacity)


        # Add scheduling algorithm specific constraints
        if scheduling_algorithm == 1:
              for flow_name, path in combination.items():
                #add_constraint_13(flow_name, path, sigma)
                #add_constraint_15(flow_name, path)
                #add_constraint_16(flow_name, path)

                var_t = f"t_{flow_name}"
                add_variable_if_new( var_t, 'C')
                var_r_min = f'rmin_{flow_name}_{i}_{j}'
                add_variable_if_new(var_r_min, 'C')

                # Equ. (13.2): t >= 0 is already implicitly ensured by cplex
                add_variable_if_new(var_r_min, 'C')
                r_min_value = min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))
                # Constraint: var_r_min = r_min_value
                add_constraint_if_new(
                    name=f"r_min_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_r_min], [1.0]),
                    sense='E',
                    rhs=r_min_value
                )

                # Calculate t = σ / min{rij: (i,j) ∈ p}
                t_value = sigma / min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))

                # Constraint: var_t = t_value
                add_constraint_if_new(
                    name=f"t_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_t], [1.0]),
                    sense='E',
                    rhs=t_value
                )
                # Equ. (13)
                # Equ. (13.2): t >= 0 is already implicitly ensured by cplex
                # add_variable_if_new(var_r_min, 'C')
                # Equ. (13.1): t*r_min >= sigma
                print(f'\nAdding constraint 13 for flow {flow_name} for edge ({i}, {j}):')
                add_constraint_if_new(
                    name=f"t_rmin_sigma_{flow_name}_{i}_{j}",
                    quad_expr=cplex.SparseTriple(ind1=[var_t], ind2=[var_r_min], val=[1.0]),
                    sense='G',
                    rhs=sigma
                )

                # Initialize WCD Equ. (16)
                wcd_sparse_pair = cplex.SparsePair(ind=[var_t], val=[1.0])
                for i, j in zip(path[:-1], path[1:]):
                    var_x = f'x_{flow_name}_{i}_{j}'
                    var_flow_rate = f"r_{flow_name}_{i}_{j}"
                    var_s = f"s_{flow_name}_{i}_{j}"

                    add_variable_if_new(var_s, 'C')
                    add_variable_if_new(var_x, 'B')
                    add_variable_if_new(var_flow_rate, 'C')

                    var_x = f'x_{flow_name}_{i}_{j}'
                    var_flow_rate = f"r_{flow_name}_{i}_{j}"
                    var_s = f"s_{flow_name}_{i}_{j}"

                    add_variable_if_new(var_s, 'C')
                    add_variable_if_new(var_x, 'B')
                    add_variable_if_new(var_flow_rate, 'C')

                    #print(f'\nAdding constraint 15 for flow {flow_name} for edge ({i}, {j}):')
                    # Equ. (15)
                    # sij * rij ≥ xij^2
                    #  -> sij * rij - xij^2 >= 0
                    # Constraint: sij ≥ 0

                    add_constraint_if_new(
                        name=f"srp_nonlinear_constraint_{flow_name}_{i}_{j}",
                        quad_expr=cplex.SparseTriple(ind1=[var_s, var_x], ind2=[var_flow_rate, var_x], val=[1.0, -1.0]),
                        sense='G',
                        rhs=0
                    )
                    # Constraint: sij ≥ 0
                    add_constraint_if_new(
                        name=f"srp_s_nonnegative_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_s], [1.0]),
                        sense='G',
                        rhs=0
                    )

                    # JF: This part is not necesarry, since it is already ensured by the flows conservation constraint
                    # Ensure var_x is 1 for edges in the path
                    add_constraint_if_new(
                        name=f"x_var_path_{flow_name}_{i}_{j}",
                       lin_expr=cplex.SparsePair([var_x], [1.0]),
                       sense='E',
                       rhs=1.0
                     )

                    # Equ. (16) summand for current edge
                    # L * s_ij term
                    wcd_sparse_pair.ind.append(var_s)
                    wcd_sparse_pair.val.append(L)

                    # (L / w_ij + l_ij + n_i) * x_ij term
                    term_value = (L / resource_graph[i][j]['bandwidth']) + resource_graph[i][j]['latency'] + n_i
                    wcd_sparse_pair.ind.append(var_x)
                    wcd_sparse_pair.val.append(term_value)

                # Equ. (16) finalize WCD constraint: WCD ≤ δ
                add_constraint_if_new(
                    name=f"wcd_deadline_{flow_name}",
                    lin_expr=wcd_sparse_pair,
                    sense='L',
                    rhs=field_devices_delta[flow_name]
                )

        # Group-Based (GB) scheduling algorithm
        elif scheduling_algorithm == 2:
            for flow_name, path in combination.items():
                # add_constraint_13(flow_name, path, sigma)
                # add_constraint_15(flow_name, path)
                # add_constraint_16(flow_name, path)
                var_t = f"t_{flow_name}"
                add_variable_if_new(var_t, 'C')
                var_r_min = f'rmin_{flow_name}_{i}_{j}'
                add_variable_if_new(var_r_min, 'C')

                # Equ. (13.2): t >= 0 is already implicitly ensured by cplex
                add_variable_if_new(var_r_min, 'C')
                r_min_value = min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))
                # Constraint: var_r_min = r_min_value
                add_constraint_if_new(
                    name=f"r_min_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_r_min], [1.0]),
                    sense='E',
                    rhs=r_min_value
                )

                # Calculate t = σ / min{rij: (i,j) ∈ p}
                t_value = sigma / min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))

                # Constraint: var_t = t_value
                add_constraint_if_new(
                    name=f"t_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_t], [1.0]),
                    sense='E',
                    rhs=t_value
                )
                # Equ. (13)
                # Equ. (13.2): t >= 0 is already implicitly ensured by cplex
                # add_variable_if_new(var_r_min, 'C')
                # Equ. (13.1): t*r_min >= sigma
                print(f'\nAdding constraint 13 for flow {flow_name} for edge ({i}, {j}):')
                add_constraint_if_new(
                    name=f"t_rmin_sigma_{flow_name}_{i}_{j}",
                    quad_expr=cplex.SparseTriple(ind1=[var_t], ind2=[var_r_min], val=[1.0]),
                    sense='G',
                    rhs=sigma
                )

                # Initialize WCD Equ. (16)
                wcd_sparse_pair = cplex.SparsePair(ind=[var_t], val=[1.0])
                for i, j in zip(path[:-1], path[1:]):
                    var_x = f'x_{flow_name}_{i}_{j}'
                    var_flow_rate = f"r_{flow_name}_{i}_{j}"
                    var_s = f"s_{flow_name}_{i}_{j}"

                    add_variable_if_new(var_s, 'C')
                    add_variable_if_new(var_x, 'B')
                    add_variable_if_new(var_flow_rate, 'C')


                    # print(f'\nAdding constraint 15 for flow {flow_name} for edge ({i}, {j}):')
                    # Equ. (15)
                    # sij * rij ≥ xij^2
                    #  -> sij * rij - xij^2 >= 0
                    # Constraint: sij ≥ 0

                    add_constraint_if_new(
                        name=f"srp_nonlinear_constraint_{flow_name}_{i}_{j}",
                        quad_expr=cplex.SparseTriple(ind1=[var_s, var_x], ind2=[var_flow_rate, var_x], val=[1.0, -1.0]),
                        sense='G',
                        rhs=0
                    )
                    # Constraint: sij ≥ 0
                    add_constraint_if_new(
                        name=f"srp_s_nonnegative_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_s], [1.0]),
                        sense='G',
                        rhs=0
                    )

                    # JF: This part is not necesarry, since it is already ensured by the flows conservation constraint
                    # Ensure var_x is 1 for edges in the path
                    add_constraint_if_new(
                        name=f"x_var_path_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_x], [1.0]),
                        sense='E',
                        rhs=1.0
                    )

                    # Equ. (16) summand for current edge
                    # L * s_ij term
                    wcd_sparse_pair.ind.append(var_s)
                    wcd_sparse_pair.val.append(6*L)

                    # (2 * L / w_ij + l_ij + n_i) * x_ij term
                    term_value = (2 * L / resource_graph[i][j]['bandwidth']) + resource_graph[i][j]['latency'] + n_i
                    wcd_sparse_pair.ind.append(var_x)
                    wcd_sparse_pair.val.append(term_value)

                # Equ. (16) finalize WCD constraint: WCD ≤ δ
                add_constraint_if_new(
                    name=f"wcd_deadline_{flow_name}",
                    lin_expr=wcd_sparse_pair,
                    sense='L',
                    rhs=field_devices_delta[flow_name]
                )


        elif scheduling_algorithm == 3:

            for flow_name, path in combination.items():
                # add_constraint_13(flow_name, path, sigma)
                # add_constraint_15(flow_name, path)
                # add_constraint_16(flow_name, path)

                var_t = f"t_{flow_name}"
                add_variable_if_new(var_t, 'C')
                var_r_min = f'rmin_{flow_name}_{i}_{j}'
                add_variable_if_new(var_r_min, 'C')

                # Equ. (13.2): t >= 0 is already implicitly ensured by cplex
                add_variable_if_new(var_r_min, 'C')
                r_min_value = min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))
                # Constraint: var_r_min = r_min_value
                add_constraint_if_new(
                    name=f"r_min_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_r_min], [1.0]),
                    sense='E',
                    rhs=r_min_value
                )

                # Calculate t = σ / min{rij: (i,j) ∈ p}
                t_value = sigma / min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))

                # Constraint: var_t = t_value
                add_constraint_if_new(
                    name=f"t_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_t], [1.0]),
                    sense='E',
                    rhs=t_value
                )
                # Equ. (13)
                # Equ. (13.2): t >= 0 is already implicitly ensured by cplex
                # add_variable_if_new(var_r_min, 'C')
                # Equ. (13.1): t*r_min >= sigma
                print(f'\nAdding constraint 13 for flow {flow_name} for edge ({i}, {j}):')
                add_constraint_if_new(
                    name=f"t_rmin_sigma_{flow_name}_{i}_{j}",
                    quad_expr=cplex.SparseTriple(ind1=[var_t], ind2=[var_r_min], val=[1.0]),
                    sense='G',
                    rhs=sigma
                )
                delay_slack = field_devices_delta[device] - t_value
                # Initialize WCD Equ. (16)
                wcd_sparse_pair = cplex.SparsePair(ind=[var_t], val=[1.0])
                for i, j in zip(path[:-1], path[1:]):
                    var_x = f'x_{flow_name}_{i}_{j}'
                    var_flow_rate = f"r_{flow_name}_{i}_{j}"
                    var_s = f"s_{flow_name}_{i}_{j}"

                    add_variable_if_new(var_s, 'C')
                    add_variable_if_new(var_x, 'B')
                    add_variable_if_new(var_flow_rate, 'C')



                    # print(f'\nAdding constraint 15 for flow {flow_name} for edge ({i}, {j}):')
                    # Equ. (15)
                    # sij * rij ≥ xij^2
                    #  -> sij * rij - xij^2 >= 0
                    # Constraint: sij ≥ 0

                    add_constraint_if_new(
                        name=f"srp_nonlinear_constraint_{flow_name}_{i}_{j}",
                        quad_expr=cplex.SparseTriple(ind1=[var_s, var_x], ind2=[var_flow_rate, var_x], val=[1.0, -1.0]),
                        sense='G',
                        rhs=0
                    )
                    # Constraint: sij ≥ 0
                    add_constraint_if_new(
                        name=f"srp_s_nonnegative_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_s], [1.0]),
                        sense='G',
                        rhs=0
                    )

                    # JF: This part is not necesarry, since it is already ensured by the flows conservation constraint
                    # Ensure var_x is 1 for edges in the path
                    add_constraint_if_new(
                        name=f"x_var_path_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_x], [1.0]),
                        sense='E',
                        rhs=1.0
                    )

                    # Equ. (16) summand for current edge
                    # L * s_ij term
                    wcd_sparse_pair.ind.append(var_s)
                    wcd_sparse_pair.val.append(L)

                    # (L / w_ij + l_ij + n_i) * x_ij term
                    term_value = (L / resource_graph[i][j]['bandwidth']) * len(flows_over_edge[(i,j)]) + resource_graph[i][j]['latency'] + n_i
                    wcd_sparse_pair.ind.append(var_x)
                    wcd_sparse_pair.val.append(term_value)

                    # Update delay slack (19)
                    delay_slack -= resource_graph[i][j]['latency'] + (len(flows_over_edge[(i,j)]) - 1) * L / resource_graph[i][j]['bandwidth'] + l_ij + n_i


                # Equ. (16) finalize WCD constraint: WCD ≤ δ
                add_constraint_if_new(
                    name=f"wcd_deadline_{flow_name}",
                    lin_expr=wcd_sparse_pair,
                    sense='L',
                    rhs=field_devices_delta[flow_name]
                )


                # Calculate the delay slack for the flow k
                # ¯δk = δk - (σk / min{rij: (i, j) ∈ p(k)}) - Σ(L/rij + (|P(i,j)|−1) L/wij + lij + ni)(i,j)∈p(k)
                for i, j in zip(path[:-1], path[1:]):
                   # Admission control constraint
                   admission_control_expr = cplex.SparsePair(
                    ind=[f'x_{flow_name}_{i}_{j}' ],
                    val=[L / resource_graph[i][j]['bandwidth']]
                )

                add_constraint_if_new(
                    name=f"admission_control_{flow_name}",
                    lin_expr=admission_control_expr,
                    sense='L',
                    rhs=delay_slack
                )

        # Frame-Based (FB) scheduling algorithm
        elif scheduling_algorithm == 4:
            for flow_name, path in combination.items():

                var_t = f"t_{flow_name}"
                add_variable_if_new(var_t, 'C')
                var_r_min = f'rmin_{flow_name}'
                add_variable_if_new(var_r_min, 'C')

                # Calculate r_min
                r_min_value = min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))
                add_constraint_if_new(
                    name=f"r_min_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_r_min], [1.0]),
                    sense='E',
                    rhs=r_min_value
                )

                # Calculate t = σ / min{rij: (i,j) ∈ p}
                t_value = sigma / r_min_value
                add_constraint_if_new(
                    name=f"t_value_{flow_name}",
                    lin_expr=cplex.SparsePair([var_t], [1.0]),
                    sense='E',
                    rhs=t_value
                )

                # Initialize WCD calculation
                wcd_sparse_pair = cplex.SparsePair(ind=[var_t], val=[1.0])
                delay_slack = field_devices_delta[flow_name] - t_value
                for i, j in zip(path[:-1], path[1:]):
                    var_x = f'x_{flow_name}_{i}_{j}'
                    var_s = f"s_{flow_name}_{i}_{j}"
                    var_v = f"v_{flow_name}_{i}_{j}"
                    var_z = f"z_{flow_name}_{i}_{j}"
                    var_flow_rate = f"r_{flow_name}_{i}_{j}"
                    var_theta = f"theta_{flow_name}_{i}_{j}"

                    add_variable_if_new(var_x, 'B')
                    add_variable_if_new(var_s, 'C')
                    add_variable_if_new(var_v, 'C')
                    add_variable_if_new(var_z, 'C')
                    add_variable_if_new(var_flow_rate, 'C')
                    add_variable_if_new(var_theta, 'C')

                    w_ij = resource_graph[i][j]['bandwidth']
                    l_ij = resource_graph[i][j]['latency']
                    p_ij = len(flows_over_edge[(i, j)])

                    # θij = L*sij*L/wij *|P(i, j)|* xij + vij
                    # Equ.
                    wcd_sparse_pair.ind.append(var_s)
                    wcd_sparse_pair.val.append((L * L)/ w_ij * p_ij)# we ignoreed x here, beceuse it is considred 1 when we calculate its value on a slected path
                    wcd_sparse_pair.ind.append(var_v)
                    wcd_sparse_pair.val.append(1.0)



                    # vij ≥ L*sij –L/wij
                    add_constraint_if_new(
                        name=f"v_constraint1_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_v, var_s], [1.0, -L]),
                        sense='G',
                        rhs=-L / w_ij
                    )

                    # vij ≥ (L/ rijmin) xij –L*rij /(wij ∗rijmin)
                    add_constraint_if_new(
                        name=f"v_constraint2_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_v, var_x, var_flow_rate],
                                                  [1.0, -L / r_min_value, L / (w_ij * r_min_value)]),
                        sense='G',
                        rhs=0
                    )

                    # vij ≥ 0
                    add_constraint_if_new(
                        name=f"v_nonnegative_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_v], [1.0]),
                        sense='G',
                        rhs=0
                    )

                    # sij rij ≥ xij^2
                    add_constraint_if_new(
                        name=f"srp_nonlinear_constraint_{flow_name}_{i}_{j}",
                        quad_expr=cplex.SparseTriple(ind1=[var_s, var_x], ind2=[var_flow_rate, var_x], val=[1.0, -1.0]),
                        sense='G',
                        rhs=0
                    )

                    # sij ≥ 0
                    add_constraint_if_new(
                        name=f"s_nonnegative_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_s], [1.0]),
                        sense='G',
                        rhs=0
                    )
                    # Update delay slack (19)
                    delay_slack -= resource_graph[i][j]['latency'] + (len(flows_over_edge[(i, j)]) - 1) * L / resource_graph[i][j]['bandwidth'] + l_ij + n_i


                # WCD constraint: WCD ≤ δ
                add_constraint_if_new(
                    name=f"wcd_deadline_{flow_name}",
                    lin_expr=wcd_sparse_pair,
                    sense='L',
                    rhs=field_devices_delta[flow_name]
                )

                # Admission control constraint

                # Calculate the delay slack for the flow k
                # ¯δk = δk - (σk / min{rij: (i, j) ∈ p(k)}) - Σ(L/rij + (|P(i,j)|−1) L/wij + lij + ni)(i,j)∈p(k)
                for i, j in zip(path[:-1], path[1:]):
                    var_x = f'x_{flow_name}_{i}_{j}'
                    var_z = f"z_{flow_name}_{i}_{j}"
                    var_flow_rate = f"r_{flow_name}_{i}_{j}"
                    w_ij = resource_graph[i][j]['bandwidth']

                    # zij ≥ 1 /rijmin (24.1)
                    add_constraint_if_new(
                        name=f"z_constraint1_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_z], [1.0]),
                        sense='G',
                        rhs=1 / rmin[(i, j)]
                    )

                    # zij ≥ sij (24.2)
                    add_constraint_if_new(
                        name=f"z_constraint2_{flow_name}_{i}_{j}",
                        lin_expr=cplex.SparsePair([var_z, f"s_{flow_name}_{i}_{j}"], [1.0, -1.0]),
                        sense='G',
                        rhs=0
                    )

                    # Admission control constraint
                    admission_control_expr = cplex.SparsePair(
                            ind=[f'x_{flow_name}_{i}_{j}'],
                            val=[L / resource_graph[i][j]['bandwidth'] + L]
                        )

                    admission_control_expr.ind.append(var_z),
                    admission_control_expr.val.append(L - flows[flow_name]['reserved_rates'])


                #Equ (23)
                add_constraint_if_new(
                    name=f"admission_control_{flow_name}",
                    lin_expr=admission_control_expr,
                    sense='L',
                    rhs=delay_slack
                )


        # construct the objective_terms list using the reservation_costs and variables
        objective_terms = [(variables[edge], reservation_costs[edge]) for edge in variables]
        #print(objective_terms)
        # Set the linear objective function in CPLEX using the constructed objective_terms
        prob.objective.set_linear(objective_terms)
        prob.objective.set_sense(prob.objective.sense.minimize)

        # Solve the problem
        prob.solve()

        # Check if a valid solution was found
        if prob.solution.get_status() in [1, 101, 102]:
            valid_solutions.append((combination, prob.solution.get_objective_value()))
            valid_solution_count += 1

            for flow_name, path in combination.items():
                deadline = field_devices_delta[flow_name]

                print(f"Device: {flow_name}")
                #print(f"Calculated WCD: {calculated_wcd}")
                print(f"Deadline (field_devices_delta): {deadline}")
                print(f"Path: {path}")

                # Calculate edge delay based on scheduling algorithm
                if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP)
                    # Dictionary to store WCD values and deadlines for each flow
                    wcd_results = {}

                    # Iterate over each flow and compute the WCD value after solving
                    for flow_name, path in combination.items():
                        # Retrieve the value of t
                        var_t = f"t_{flow_name}"
                        t_value = prob.solution.get_values(var_t)

                        # Initialize wcd_value with the value of t
                        wcd_value = t_value

                        # Iterate over the edges and retrieve corresponding variable values
                        for i, j in zip(path[:-1], path[1:]):
                            var_x = f'x_{flow_name}_{i}_{j}'
                            var_s = f"s_{flow_name}_{i}_{j}"

                            # Get the values of s_ij and x_ij
                            s_value = prob.solution.get_values(var_s)
                            x_value = prob.solution.get_values(var_x)

                            # Calculate L * s_ij
                            wcd_value += L * s_value

                            # Calculate (L / w_ij + l_ij + n_i) * x_ij
                            term_value = (L / resource_graph[i][j]['bandwidth']) + resource_graph[i][j]['latency'] + n_i
                            wcd_value += term_value * x_value

                        # Get the deadline (delta) for the current flow
                        deadline = field_devices_delta[flow_name]

                        # Store the WCD value and deadline in the dictionary
                        wcd_results[flow_name] = {
                            'service_type': field_devices[flow_name]['service_type'],
                            'Path': path,
                            'bitrate_factor': flows[flow_name]['reserved_rates'],
                            'wcd_value': wcd_value,
                            'deadline': deadline
                        }

                        # Print the results for debugging
                        print(f"\nComputed WCD value for flow {flow_name}: {wcd_value}")
                        print(f"Field device deadline (delta): {deadline}")
                elif scheduling_algorithm == 2:  # Group-Based (GB)
                    # Dictionary to store WCD values and deadlines for each flow
                    wcd_results = {}

                    # Iterate over each flow and compute the WCD value after solving
                    for flow_name, path in combination.items():
                        # Retrieve the value of t
                        var_t = f"t_{flow_name}"
                        t_value = prob.solution.get_values(var_t)

                        # Initialize wcd_value with the value of t
                        wcd_value = t_value

                        # Iterate over the edges and retrieve corresponding variable values
                        for i, j in zip(path[:-1], path[1:]):
                            var_x = f'x_{flow_name}_{i}_{j}'
                            var_s = f"s_{flow_name}_{i}_{j}"

                            # Get the values of s_ij and x_ij
                            s_value = prob.solution.get_values(var_s)
                            x_value = prob.solution.get_values(var_x)

                            # Calculate L * s_ij
                            wcd_value += 6 * L * s_value

                            # Calculate (L / w_ij + l_ij + n_i) * x_ij
                            term_value = (2 * L / resource_graph[i][j]['bandwidth']) + resource_graph[i][j]['latency'] + n_i
                            wcd_value += term_value * x_value

                        # Get the deadline (delta) for the current flow
                        deadline = field_devices_delta[flow_name]

                        # Store the WCD value and deadline in the dictionary
                        wcd_results[flow_name] = {
                            'service_type': field_devices[flow_name]['service_type'],
                            'Path': path,
                            'bitrate_factor': flow[flow_name]['reserved_rates'],
                            'wcd_value': wcd_value,
                            'deadline': deadline
                        }

                        # Print the results for debugging
                        print(f"\nComputed WCD value for flow {flow_name}: {wcd_value}")
                        print(f"Field device deadline (delta): {deadline}")
                elif scheduling_algorithm == 3:  # Weakly Rate-Proportional (WRP)
                    # Dictionary to store WCD values and deadlines for each flow
                    wcd_results = {}

                    # Iterate over each flow and compute the WCD value after solving
                    for flow_name, path in combination.items():
                        # Retrieve the value of t
                        var_t = f"t_{flow_name}"
                        t_value = prob.solution.get_values(var_t)

                        # Initialize wcd_value with the value of t
                        wcd_value = t_value

                        # Iterate over the edges and retrieve corresponding variable values
                        for i, j in zip(path[:-1], path[1:]):
                            var_x = f'x_{flow_name}_{i}_{j}'
                            var_s = f"s_{flow_name}_{i}_{j}"

                            # Get the values of s_ij and x_ij
                            s_value = prob.solution.get_values(var_s)
                            x_value = prob.solution.get_values(var_x)

                            # Calculate L * s_ij
                            wcd_value += L * s_value

                            # Calculate (L / w_ij + l_ij + n_i) * x_ij
                            term_value = ((L / resource_graph[i][j]['bandwidth']) * len(flows_over_edge[(i,j)])) + resource_graph[i][j]['latency'] + n_i
                            wcd_value += term_value * x_value

                        # Get the deadline (delta) for the current flow
                        deadline = field_devices_delta[flow_name]

                        # Store the WCD value and deadline in the dictionary
                        wcd_results[flow_name] = {
                            'service_type': field_devices[flow_name]['service_type'],
                            'Path': path,
                            'bitrate_factor': flow[flow_name]['reserved_rates'],
                            'wcd_value': wcd_value,
                            'deadline': deadline
                        }
                        # Print the results for debugging
                        print(f"\nComputed WCD value for flow {flow_name}: {wcd_value}")
                        print(f"Field device deadline (delta): {deadline}")
                elif scheduling_algorithm == 4:  # Frame-Based (FB)
                    # Dictionary to store WCD values and deadlines for each flow
                    wcd_results = {}

                    # Iterate over each flow and compute the WCD value after solving
                    for flow_name, path in combination.items():
                        # Retrieve the value of t (σ / min{rij: (i,j) ∈ p})
                        var_t = f"t_{flow_name}"
                        t_value = prob.solution.get_values(var_t)

                        # Initialize wcd_value with the value of t
                        wcd_value = t_value

                        # Iterate over the edges and retrieve corresponding variable values
                        for i, j in zip(path[:-1], path[1:]):
                            var_x = f'x_{flow_name}_{i}_{j}'
                            var_theta = f"theta_{flow_name}_{i}_{j}"

                            # Get the values of theta_ij and x_ij
                            theta_value = prob.solution.get_values(var_theta)
                            x_value = prob.solution.get_values(var_x)

                            # Add theta_ij to the WCD value
                            wcd_value += theta_value

                            # Add (l_ij + n_i) * x_ij to the WCD value
                            l_ij = resource_graph[i][j]['latency']
                            #n_i = n_i_values[i]  # Assuming n_i_values is a dictionary with node processing times
                            wcd_value += (l_ij + n_i) * x_value

                        # Get the deadline (delta) for the current flow
                        deadline = field_devices_delta[flow_name]

                        # Store the WCD value and deadline in the dictionary
                        wcd_results[flow_name] = {
                            'service_type': field_devices[flow_name]['service_type'],
                            'Path': path,
                            'bitrate_factor': flow[flow_name]['reserved_rates'],
                            'wcd_value': wcd_value,
                            'deadline': deadline
                        }


                        # Print the results for debugging
                        print(f"\nComputed WCD value for flow {flow_name}: {wcd_value}")
                        print(f"Field device deadline (delta): {deadline}")

                else:
                        raise ValueError("Invalid scheduling algorithm")


                Run = input('Enter 1 or True if you would like to run the second Phase using Network Calculas')
                if Run == 1 or True:
                    eval_result = javNC_validateNetwork(resource_graph, wcd_results, pickle_qos_dict, prio_dict)
                    dumpNetAndResult(result)
                    print("Network calculus evaluation finished. Qos is met: " + str(not eval_result))
                return

    print(f"Total valid solutions found: {valid_solution_count}")
    print(wcd_values)
    #for solution in valid_solutions:
        #print(f"Combination: {solution[0]} - Objective Value: {solution[1]}")


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

    # get the start time
    st = time.time()
    # Solve the optimal path problem
    solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm)
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
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
