
import cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import itertools
from math import prod
import time
import random
from collections import Counter
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


def calculate_wcd(combination, resource_graph, scheduling_algorithm):
    for device, path in combination.items():
        wcd_var = f"wcd_{device}"
        calculated_wcd = prob.solution.get_values(wcd_var)
        deadline = field_devices_delta[device]

        print(f"Device: {device}")
        print(f"Calculated WCD: {calculated_wcd}")
        print(f"Deadline (field_devices_delta): {deadline}")
        print(f"Path: {path}")

        total_wcd = 0
        for i, j in zip(path[:-1], path[1:]):
            edge = (i, j)
            bandwidth = resource_graph[i][j]['bandwidth']
            latency = resource_graph[i][j]['latency']
            L = 1500  # Make sure this matches your packet size
            n_i = 40 / 1000  # Node processing delay, ensure this matches your definition

            # Get the flow rate for this edge and device
            flow_rate_var = f"r_{device}_{i}_{j}"
            flow_rate = prob.solution.get_values(flow_rate_var)

            # Get the s variable for this edge and device
            s_var = f"s_{device}_{i}_{j}"
            s_value = prob.solution.get_values(s_var)

            # Calculate edge delay based on scheduling algorithm
            if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP)
                edge_delay = L * s_value + (L / bandwidth + latency + n_i)
            elif scheduling_algorithm == 2:  # Group-Based (GB)
                edge_delay = 6 * L * s_value + (2 * L / bandwidth + latency + n_i)
            elif scheduling_algorithm == 3:  # Weakly Rate-Proportional (WRP)
                paths_using_edge = flow_number(i, j, combination)
                edge_delay = L * s_value + (L / flow_rate + L / bandwidth + latency + n_i) * paths_using_edge
            elif scheduling_algorithm == 4:  # Frame-Based (FB)
                v_var = f"v_{device}_{i}_{j}"
                v_value = prob.solution.get_values(v_var)
                paths_using_edge = flow_number(i, j, combination)
                edge_delay = L * s_value * (L / latency + L / bandwidth) * paths_using_edge + v_value + latency + n_i
            else:
                raise ValueError("Invalid scheduling algorithm")

            total_wcd += edge_delay

            print(f"Edge {i}-{j}:")
            print(f"  Flow rate: {flow_rate}")
            print(f"  s value: {s_value}")
            print(f"  Edge delay: {edge_delay}")

        print(f"Total calculated WCD: {total_wcd}")
        print(f"Difference from CPLEX WCD: {abs(total_wcd - calculated_wcd)}")
        print("--------------------")
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
            'reserved_rates': rho * tau + sigma  # Changed from set to direct value
        } for device in field_devices_delta}

        total_combinations = calculate_total_combinations(flow)
        print(f"Total number of combinations: {total_combinations}")

        # Usage
        combinations_generator = generate_path_combinations(flow)

        # Get all combinations
        #all_combinations = list(combinations_generator)

        first_100_combinations = list(itertools.islice(combinations_generator, 100))

        valid_solutions = []
        valid_solution_count = 0
        wcd_values = {}

        # Iterate through the first 100 combinations
        for combination in first_100_combinations:

            prob = cplex.Cplex()
            prob.set_problem_type(prob.problem_type.MIQCP)
            added_variables = set()
            added_constraints = set()

            # Initialize capacity variables
            remaining_capacity = {}
            reservable_capacity = {}

            # Initialize variables for minimum reserved flow and maximum flow
            rmin = {}
            cmax = float('-inf')

            # Process all devices and paths in a single loop
            for device, path in combination.items():
                current_reserved_rate = flow[device]['reserved_rates']

                # Update cmax for the entire network
                if current_reserved_rate > cmax:
                    cmax = current_reserved_rate

                path_min_value = float('inf')  # Initialize path minimum value

                # Process each edge in the path
                for i, j in zip(path[:-1], path[1:]):
                    edge = (i, j)
                    var_name = generate_edge_variable_name(edge)
                    flow_rate_var = f"r_{device}_{i}_{j}"
                    edge_capacity = f"c__{i}_{j}"

                    # Add variables
                    add_variable_if_new(var_name, 'B')
                    add_variable_if_new(flow_rate_var, 'C')
                    add_variable_if_new(edge_capacity, 'C')

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

                # Store the minimum value for the entire path
                flow[device]['path_min_value'] = path_min_value


                    # Add constraints
                    # The edge capacity = the edge bandwidth (wij)
                add_constraint_if_new(
                        f"cap_{i}_{j}",
                        cplex.SparsePair([edge_capacity], [1.0]),
                        'E', remaining_capacity[edge]
                    )

                    # Constraint: flow rate(rij) <= remaining capacity
                add_constraint_if_new(
                        f"flow_cap_{device}_{i}_{j}",
                        cplex.SparsePair([flow_rate_var], [1.0]),
                        'L', remaining_capacity[edge]
                    )

                    # Constraint: rij >= ρ
                add_constraint_if_new(
                        f"rho_{device}_{i}_{j}",
                        cplex.SparsePair([flow_rate_var], [1.0]),
                        'G', rho
                    )

                    # Constraint: flow rate = flow[device]['reserved_rates']
                add_constraint_if_new(
                        f"flow_eq_{device}_{i}_{j}",
                        cplex.SparsePair([flow_rate_var], [1.0]),
                        'E', current_reserved_rate
                    )



            # Output rmin for debugging
            #for edge, min_value in rmin.items():
                #print(f"Minimum reserved flow for edge {edge}: {min_value}")

            # Output cmax for debugging
            #print(f"Maximum flow value in the network: {cmax}")

            # Maximum capacity variable
            c_max_var = 'c_max'
            add_variable_if_new(c_max_var, 'C')
            for edge in reservable_capacity:
                i, j = edge
                add_constraint_if_new(
                    f"c_max_{i}_{j}",
                    cplex.SparsePair([c_max_var], [-1.0]),
                    'L', cmax
                )

            # Additional constraints
            for edge in reservable_capacity:
                i, j = edge
                for device in combination:
                    if edge in zip(combination[device][:-1], combination[device][1:]):
                        flow_rate_var = f"r_{device}_{i}_{j}"
                        add_variable_if_new(flow_rate_var, 'C')
                        edge_var = generate_edge_variable_name(edge)

                        # Constraint: 0 <= rij <= cij * xij
                        add_constraint_if_new(
                            f"flow_ub_{device}_{i}_{j}",
                            cplex.SparsePair([flow_rate_var, edge_var], [1.0, -cmax]),
                            'L', 0.0
                        )

                        # Constraint: ρ <= rmin <= rij + cmax(1 - xij)
                        # Defining r_min variable
                        r_min = f"rmin_{i}_{j}"
                        add_variable_if_new(r_min, 'C')

                        # Adding the constraint ρ <= r_min
                        add_constraint_if_new(
                            f"rmin_lb_{device}_{i}_{j}",
                            cplex.SparsePair([r_min], [1.0]),
                            'G', rho
                        )

                        # Adding the constraint r_min <= r_ij + c_max * (1 - x_ij)
                        add_constraint_if_new(
                            f"rmin_ub_{device}_{i}_{j}",
                            cplex.SparsePair([r_min, flow_rate_var, edge_var, c_max_var],
                                             [1.0, -1.0, reservable_capacity[edge], -1.0]),
                            'L', 0.0
                        )

            # Add flow-specific constraints
            for device, path_nodes in combination.items():
                    # Flow conservation constraints
                    for node in path_nodes:
                        incoming = []
                        outgoing = []

                        # Collect incoming edges
                        for prev_node in path_nodes:
                            if (prev_node, node) in resource_graph.edges:
                                incoming.append(generate_edge_variable_name((prev_node, node)))

                        # Collect outgoing edges
                        for next_node in path_nodes:
                            if (node, next_node) in resource_graph.edges:
                                outgoing.append(generate_edge_variable_name((node, next_node)))

                        # Determine the right-hand side value
                        if node == path_nodes[0]:  # Source node
                            rhs = 1
                        elif node == path_nodes[-1]:  # Destination node
                            rhs = -1
                        else:  # Intermediate node
                            rhs = 0

                        # Create the constraint name
                        constraint_name = f'flow_conserve_{device}_{node}'
                        # Update the set of added constraints
                        added_constraints.add(constraint_name)
                        # Add the constraint if it hasn't been added already
                        if constraint_name not in added_constraints:
                            add_constraint_if_new(
                                constraint_name=constraint_name,
                                lin_expr=cplex.SparsePair(incoming + outgoing,
                                                          [1.0] * len(incoming) + [-1.0] * len(outgoing)),
                                sense='E',
                                rhs=rhs
                            )

            # Add scheduling algorithm specific constraints
            if scheduling_algorithm == 1:
                for device, path in combination.items():
                    # Calculate t = σ / min{rij: (i,j) ∈ p}
                    t_value = sigma / min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))

                    wcd_var = f"wcd_{device}"
                    add_variable_if_new(wcd_var, 'C')

                    # Initialize WCD with t_value
                    add_constraint_if_new(
                        f"wcd_init_{device}",
                        cplex.SparsePair([wcd_var], [1.0]),
                        'E',
                        t_value
                    )

                    # Constraint: t * rmin ≥ σ
                    add_constraint_if_new(
                        f"t_rmin_sigma_{device}",
                        cplex.SparsePair([f"r_{device}_{path[0]}_{path[1]}"], [t_value]),
                        'G',
                        sigma
                    )

                    # Constraint: t ≥ 0 (implicitly handled by CPLEX for continuous variables)

                    # Initialize WCD as a SparsePair
                    wcd = cplex.SparsePair(ind=[wcd_var], val=[1.0])
                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        flow_rate_var = f"r_{device}_{i}_{j}"
                        s_var = f"s_{device}_{i}_{j}"

                        add_variable_if_new(s_var, 'C')  # Changed to Continuous
                        add_variable_if_new(x_var, 'B')
                        add_variable_if_new(flow_rate_var, 'C')

                        # Ensure x_var is 1 for edges in the path
                        add_constraint_if_new(
                            f"x_var_path_{device}_{i}_{j}",
                            cplex.SparsePair([x_var], [1.0]),
                            'E',
                            1.0
                        )

                        # Constraint: sij * rij ≥ xij^2
                        add_constraint_if_new(
                            f"srp_nonlinear_constraint_{device}_{i}_{j}",
                            cplex.SparsePair([s_var, flow_rate_var, x_var], [1.0, 1.0, -2.0]),
                            'G',
                            0
                        )

                        # Constraint: sij ≥ 0
                        add_constraint_if_new(
                            f"srp_s_nonnegative_{device}_{i}_{j}",
                            cplex.SparsePair([s_var], [1.0]),
                            'G',
                            0
                        )

                        # Calculate θij = L * sij + (L/wij + lij + ni) * xij
                        bandwidth = resource_graph[i][j]['bandwidth']
                        latency = resource_graph[i][j]['latency']
                        #theta_ij = (L * s_var) + ((L / bandwidth + latency + n_i) * x_var)
                        # Create linear expression for θij
                        theta_ij = cplex.SparsePair(
                            ind=[s_var, x_var],
                            val=[L, (L / bandwidth + latency + n_i)]
                        )

                        # Update WCD by combining SparsePairs
                        wcd = cplex.SparsePair(
                            ind=wcd.ind + theta_ij.ind,
                            val=wcd.val + theta_ij.val
                        )

                    # WCD constraint: WCD ≤ δ
                    add_constraint_if_new(
                        f"wcd_deadline_{device}",
                        cplex.SparsePair([wcd_var], [1.0]),
                        'L',
                        field_devices_delta[device]
                    )


            # Group-Based (GB) scheduling algorithm
            elif scheduling_algorithm == 2:
                for device, path in combination.items():
                    # Calculate t = σ / min{rij: (i,j) ∈ p}
                    t_value = sigma / min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))

                    wcd_var = f"wcd_{device}"
                    add_variable_if_new(wcd_var, 'C')

                    # Initialize WCD with t_value
                    add_constraint_if_new(
                        f"wcd_init_{device}",
                        cplex.SparsePair([wcd_var], [1.0]),
                        'E',
                        t_value
                    )

                    # Constraint: t * rmin ≥ σ
                    add_constraint_if_new(
                        f"t_rmin_sigma_{device}",
                        cplex.SparsePair([f"r_{device}_{path[0]}_{path[1]}"], [t_value]),
                        'G',
                        sigma
                    )

                    # Initialize WCD as a SparsePair
                    wcd = cplex.SparsePair(ind=[wcd_var], val=[1.0])
                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        flow_rate_var = f"r_{device}_{i}_{j}"
                        s_var = f"s_{device}_{i}_{j}"

                        add_variable_if_new(s_var, 'C')
                        add_variable_if_new(x_var, 'B')
                        add_variable_if_new(flow_rate_var, 'C')

                        # Ensure x_var is 1 for edges in the path
                        add_constraint_if_new(
                            f"x_var_path_{device}_{i}_{j}",
                            cplex.SparsePair([x_var], [1.0]),
                            'E',
                            1.0
                        )

                        # Constraint: sij * rij ≥ xij^2
                        add_constraint_if_new(
                            f"srp_nonlinear_constraint_{device}_{i}_{j}",
                            cplex.SparsePair([s_var, flow_rate_var, x_var], [1.0, 1.0, -2.0]),
                            'G',
                            0
                        )

                        # Constraint: sij ≥ 0
                        add_constraint_if_new(
                            f"srp_s_nonnegative_{device}_{i}_{j}",
                            cplex.SparsePair([s_var], [1.0]),
                            'G',
                            0
                        )

                        # Calculate θij = 6Lsij + (2L/wij + lij + ni) * xij
                        bandwidth = resource_graph[i][j]['bandwidth']
                        latency = resource_graph[i][j]['latency']#L/rij

                        # Create linear expression for θij
                        theta_ij = cplex.SparsePair(
                            ind=[s_var, x_var],
                            val=[6 * L, (latency + 2 * L / bandwidth + l_ij + n_i)]
                        )

                        # Update WCD by combining SparsePairs
                        wcd = cplex.SparsePair(
                            ind=wcd.ind + theta_ij.ind,
                            val=wcd.val + theta_ij.val
                        )

                    # WCD constraint: WCD ≤ δ
                    add_constraint_if_new(
                        f"wcd_deadline_{device}",
                        cplex.SparsePair(wcd.ind, wcd.val),
                        'L',
                        field_devices_delta[device]
                    )

            elif scheduling_algorithm == 3:
                for device, path in combination.items():
                    # Calculate t = σ / min{rij: (i,j) ∈ p}
                    t_value = sigma / min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))
                    # intilize delay slack
                    delay_slack = field_devices_delta[device] - t_value
                    wcd_var = f"wcd_{device}"
                    add_variable_if_new(wcd_var, 'C')

                    # Initialize WCD with t_value
                    add_constraint_if_new(
                        f"wcd_init_{device}",
                        cplex.SparsePair([wcd_var], [1.0]),
                        'E',
                        t_value
                    )

                    # Constraint: t * rmin ≥ σ
                    add_constraint_if_new(
                        f"t_rmin_sigma_{device}",
                        cplex.SparsePair([f"r_{device}_{path[0]}_{path[1]}"], [t_value]),
                        'G',
                        sigma
                    )

                    # Constraint: t ≥ 0 (implicitly handled by CPLEX for continuous variables)

                    # Initialize WCD as a SparsePair
                    wcd = cplex.SparsePair(ind=[wcd_var], val=[1.0])
                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        flow_rate_var = f"r_{device}_{i}_{j}"
                        s_var = f"s_{device}_{i}_{j}"

                        add_variable_if_new(s_var, 'C')  # Changed to Continuous
                        add_variable_if_new(x_var, 'B')
                        add_variable_if_new(flow_rate_var, 'C')

                        # Ensure x_var is 1 for edges in the path
                        add_constraint_if_new(
                            f"x_var_path_{device}_{i}_{j}",
                            cplex.SparsePair([x_var], [1.0]),
                            'E',
                            1.0
                        )

                        # Constraint: sij * rij ≥ xij^2
                        add_constraint_if_new(
                            f"srp_nonlinear_constraint_{device}_{i}_{j}",
                            cplex.SparsePair([s_var, flow_rate_var, x_var], [1.0, 1.0, -2.0]),
                            'G',
                            0
                        )

                        # Constraint: sij ≥ 0
                        add_constraint_if_new(
                            f"srp_s_nonnegative_{device}_{i}_{j}",
                            cplex.SparsePair([s_var], [1.0]),
                            'G',
                            0
                        )

                        # Calculate |P(i,j)|
                        paths_using_edge = flow_number(i, j, combination)

                        bandwidth = resource_graph[i][j]['bandwidth']
                        latency = resource_graph[i][j]['latency']
                        # Calculate θij = L * sij + (L/rij+ L/wij + lij + ni)|P(i, j) * xij
                        # Create linear expression for θij
                        theta_ij = cplex.SparsePair(
                            ind=[s_var, x_var],
                            val=[L, (latency + L / bandwidth + l_ij + n_i) * paths_using_edge]
                        )

                        # Update WCD by combining SparsePairs
                        wcd = cplex.SparsePair(
                            ind=wcd.ind + theta_ij.ind,
                            val=wcd.val + theta_ij.val
                        )


                        # Update delay slack
                        delay_slack -=  latency + (paths_using_edge - 1) * L / bandwidth + l_ij + n_i

                    # WCD constraint: WCD ≤ δ
                    #add_constraint_if_new(f"wcd_deadline_{device}",cplex.SparsePair(wcd.ind, wcd.val),'L',field_devices_delta[device])


                    # Admission control constraint
                    admission_control_expr = cplex.SparsePair(
                        ind=[generate_edge_variable_name((i, j)) for i, j in zip(path[:-1], path[1:])],
                        val=[L / resource_graph[i][j]['bandwidth'] for i, j in zip(path[:-1], path[1:])]
                    )

                    add_constraint_if_new(
                        f"admission_control_{device}",
                        admission_control_expr,
                        'L',
                        delay_slack
                    )


            # Frame-Based (FB) scheduling algorithm
            elif scheduling_algorithm == 4:
                for device, path in combination.items():
                    # Calculate t = σ / min{rij: (i,j) ∈ p}
                    t_value = sigma / min(rmin[(i, j)] for i, j in zip(path[:-1], path[1:]))

                    wcd_var = f"wcd_{device}"
                    add_variable_if_new(wcd_var, 'C')

                    # Initialize WCD with t_value
                    add_constraint_if_new(
                        f"wcd_init_{device}",
                        cplex.SparsePair([wcd_var], [1.0]),
                        'E',
                        t_value
                    )

                    # Constraint: t * rmin ≥ σ
                    add_constraint_if_new(
                        f"t_rmin_sigma_{device}",
                        cplex.SparsePair([f"r_{device}_{path[0]}_{path[1]}"], [t_value]),
                        'G',
                        sigma
                    )

                    # Initialize WCD as a SparsePair
                    wcd = cplex.SparsePair(ind=[wcd_var], val=[1.0])

                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        x_var = generate_edge_variable_name(edge)
                        flow_rate_var = f"r_{device}_{i}_{j}"
                        s_var = f"s_{device}_{i}_{j}"
                        v_var = f"v_{device}_{i}_{j}"
                        z_var = f"z_{device}_{i}_{j}"

                        add_variable_if_new(s_var, 'C')
                        add_variable_if_new(x_var, 'B')
                        add_variable_if_new(flow_rate_var, 'C')
                        add_variable_if_new(v_var, 'C')
                        add_variable_if_new(z_var, 'C')

                        # Ensure x_var is 1 for edges in the path
                        add_constraint_if_new(
                            f"x_var_path_{device}_{i}_{j}",
                            cplex.SparsePair([x_var], [1.0]),
                            'E',
                            1.0
                        )

                        bandwidth = resource_graph[i][j]['bandwidth']
                        latency = resource_graph[i][j]['latency']

                        # Calculate |P(i,j)|
                        paths_using_edge = flow_number(i, j, combination)

                        # θij = L*sij*L/wij *|P(i, j)|* xij + vij
                        theta_ij = cplex.SparsePair(
                            ind=[s_var, x_var, v_var],
                            val=[L * (L/ latency + L / bandwidth) * paths_using_edge, 1.0, 1.0]
                        )

                        # vij ≥ L*sij –L/wij
                        add_constraint_if_new(
                            f"v_constraint1_{device}_{i}_{j}",
                            cplex.SparsePair([v_var, s_var], [1.0, -L]),
                            'G',
                            -L / bandwidth
                        )

                        # vij ≥ (L/ rijmin) xij –L*rij /(wij ∗rijmin)
                        add_constraint_if_new(
                            f"v_constraint2_{device}_{i}_{j}",
                            cplex.SparsePair([v_var, x_var, flow_rate_var],
                                             [1.0, -latency,  (bandwidth * rmin[(i, j)])]),
                            'G',
                            0
                        )

                        # vij ≥ 0
                        add_constraint_if_new(
                            f"v_nonnegative_{device}_{i}_{j}",
                            cplex.SparsePair([v_var], [1.0]),
                            'G',
                            0
                        )

                        # sij rij ≥ xij^2
                        add_constraint_if_new(
                            f"srp_nonlinear_constraint_{device}_{i}_{j}",
                            cplex.SparsePair([s_var, flow_rate_var, x_var], [1.0, 1.0, -2.0]),
                            'G',
                            0
                        )

                        # sij ≥ 0
                        add_constraint_if_new(
                            f"srp_s_nonnegative_{device}_{i}_{j}",
                            cplex.SparsePair([s_var], [1.0]),
                            'G',
                            0
                        )

                        # Update WCD by combining SparsePairs
                        wcd = cplex.SparsePair(
                            ind=wcd.ind + theta_ij.ind + [f"l_{i}_{j}", f"n_{i}"],
                            val=wcd.val + theta_ij.val + [1.0, 1.0]
                        )

                    # WCD constraint: WCD ≤ δ
                    # WCD constraint: WCD ≤ δ
                    #print(device)
                    #add_constraint_if_new(f"wcd_deadline_{device}",cplex.SparsePair(wcd.ind, wcd.val),'L',field_devices_delta[device])

                    # Admission control constraint
                    admission_control_expr = cplex.SparsePair([], [])
                    for i, j in zip(path[:-1], path[1:]):
                        bandwidth = resource_graph[i][j]['bandwidth']
                        x_var = generate_edge_variable_name((i, j))
                        z_var = f"z_{device}_{i}_{j}"
                        flow_rate_var = f"r_{device}_{i}_{j}"

                        admission_control_expr.ind.extend([x_var, z_var])
                        admission_control_expr.val.extend([L / bandwidth, L * (1 - 1 / rmin[(i, j)])])

                        # zij ≥ 1 /rijmin
                        add_constraint_if_new(
                            f"z_constraint1_{device}_{i}_{j}",
                            cplex.SparsePair([z_var], [1.0]),
                            'G',
                            1 / rmin[(i, j)]
                        )

                        # zij ≥ sij
                        add_constraint_if_new(
                            f"z_constraint2_{device}_{i}_{j}",
                            cplex.SparsePair([z_var, f"s_{device}_{i}_{j}"], [1.0, -1.0]),
                            'G',
                            0
                        )

                    add_constraint_if_new(
                        f"admission_control_{device}",
                        admission_control_expr,
                        'L',
                        field_devices_delta[device]
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

                for device, path in combination.items():
                    wcd_var = f"wcd_{device}"
                    calculated_wcd = prob.solution.get_values(wcd_var)
                    deadline = field_devices_delta[device]

                    print(f"Device: {device}")
                    print(f"Calculated WCD: {calculated_wcd}")
                    print(f"Deadline (field_devices_delta): {deadline}")
                    print(f"Path: {path}")

                    total_wcd = 0
                    for i, j in zip(path[:-1], path[1:]):
                        edge = (i, j)
                        bandwidth = resource_graph[i][j]['bandwidth']
                        latency = resource_graph[i][j]['latency']
                        L = 1500  # Make sure this matches your packet size
                        n_i = 40 / 1000  # Node processing delay, ensure this matches your definition

                        # Get the flow rate for this edge and device
                        flow_rate_var = f"r_{device}_{i}_{j}"
                        flow_rate = prob.solution.get_values(flow_rate_var)

                        # Get the s variable for this edge and device
                        s_var = f"s_{device}_{i}_{j}"
                        s_value = prob.solution.get_values(s_var)

                        # Calculate edge delay based on scheduling algorithm
                        if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP)
                            edge_delay = L * s_value + (L / bandwidth + latency + n_i)
                        elif scheduling_algorithm == 2:  # Group-Based (GB)
                            edge_delay = 6 * L * s_value + (2 * L / bandwidth + latency + n_i)
                        elif scheduling_algorithm == 3:  # Weakly Rate-Proportional (WRP)
                            paths_using_edge = flow_number(i, j, combination)
                            edge_delay = L * s_value + (
                                        L / flow_rate + L / bandwidth + latency + n_i) * paths_using_edge
                        elif scheduling_algorithm == 4:  # Frame-Based (FB)
                            v_var = f"v_{device}_{i}_{j}"
                            v_value = prob.solution.get_values(v_var)
                            paths_using_edge = flow_number(i, j, combination)
                            edge_delay = L * s_value * (
                                        L / latency + L / bandwidth) * paths_using_edge + v_value + latency + n_i
                        else:
                            raise ValueError("Invalid scheduling algorithm")

                        total_wcd += edge_delay

                        print(f"Edge {i}-{j}:")
                        print(f"  Flow rate: {flow_rate}")
                        print(f"  s value: {s_value}")
                        print(f"  Edge delay: {edge_delay}")

                    print(f"Total calculated WCD: {total_wcd}")
                    print(f"Difference from CPLEX WCD: {abs(total_wcd - calculated_wcd)}")
                    print("--------------------")

        print(f"Total valid solutions found: {valid_solution_count}")
        print(wcd_values)
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
