import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice, product

from collections import OrderedDict
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
import cplex
import networkx as nx


def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm=1, sigma=1500, rho=1500):
   try: # Ensure the graph is directed
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
            'rate': rho * tau,
            'reserved_rates': rho * tau + sigma
        }

    # Add variables and basic constraints for all edges
    for i, j in resource_graph.edges:
        add_variable_if_new(f'x_{i}_{j}', prob.variables.type.binary)
        add_variable_if_new(f'r_{i}_{j}', prob.variables.type.continuous)
        add_variable_if_new(f'r_min_{i}_{j}', prob.variables.type.continuous)
        add_variable_if_new(f'c_ij_{i}_{j}', prob.variables.type.continuous)

        # Basic constraints
        add_constraint_if_new(f'non_neg_r_{i}_{j}', [[f'r_{i}_{j}'], [1.0]], 'G', 0)
        add_constraint_if_new(f'non_neg_r_min_{i}_{j}', [[f'r_min_{i}_{j}'], [1.0]], 'G', rho)
        add_constraint_if_new(f'non_neg_c_ij_{i}_{j}', [[f'c_ij_{i}_{j}'], [1.0]], 'G', 0)
        add_constraint_if_new(f'capacity_{i}_{j}', [[f'c_ij_{i}_{j}'], [1.0]], 'L', resource_graph[i][j]['bandwidth'])

    # Add flow-specific constraints
    for device in flow:
        for path in flow[device]['paths']:
            path_nodes = path[0]
            for i, j in zip(path_nodes[:-1], path_nodes[1:]):
                r_ij_var = f'r_{i}_{j}'
                c_ij_var = f'c_ij_{i}_{j}'
                x_ij_var = f'x_{i}_{j}'

                # Flow rate constraints
                add_constraint_if_new(
                    f'flow_rate_lower_{device}_{i}_{j}',
                    [[r_ij_var], [1.0]],
                    'G',
                    flow[device]['rate']
                )
                add_constraint_if_new(
                    f'flow_rate_upper_{device}_{i}_{j}',
                    [[r_ij_var], [1.0]],
                    'L',
                    flow[device]['reserved_rates']
                )

                # Capacity constraint
                add_constraint_if_new(
                    f'capacity_constraint_{device}_{i}_{j}',
                    [[r_ij_var, c_ij_var, x_ij_var], [1.0, -1.0, -resource_graph[i][j]['bandwidth']]],
                    'L',
                    0
                )

    # Scheduling algorithm specific constraints
    if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP)
        for device in flow:
            for path in flow[device]['paths']:
                path_nodes = path[0]
                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

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
        for device in flow:
            for path in flow[device]['paths']:
                path_nodes = path[0]
                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

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
        for device in flow:
            for path in flow[device]['paths']:
                path_nodes = path[0]
                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

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
        for device in flow:
            for path in flow[device]['paths']:
                path_nodes = path[0]
                path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

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

        # Objective function: min ∑((i,j)∈A)〖f_ij * r_ij〗
        """
            Adds the linear objective function to the CPLEX model to minimize
            the weighted amount of allocated rate along the path using the given reservation costs f_ij.

            Args:
                prob: The CPLEX problem object.
                resource_graph: The resource graph containing the edges.
                field_devices: List of field devices.
                flow: The flow information for each device.
            """
    objective_dict = {}
    for i, j in resource_graph.edges:
        # Iterate through each field device
        for primary_device in field_devices_delta:  # Changed to field_devices_delta
            primary_device_paths = flow[primary_device]['paths']
            # Consider each path of the primary device
            for primary_path in primary_device_paths:
                total_reserved_rates = 0
                # Initialize reserved rates with the primary device's path if it uses edge (i, j)
                if Check_edge_inpath(primary_path[0], i, j):
                    total_reserved_rates += flow[primary_device]['reserved_rates']
                # For all other devices, consider their paths without generating all combinations
                other_devices = [device for device in field_devices_delta if
                                 device != primary_device]  # Changed to field_devices_delta
                combinations_iterator = product(*[flow[device]['paths'] for device in other_devices])
                
                # Limit to the first 10 combinations
                for combination in islice(combinations_iterator, 1):
                    combination_reserved_rates = total_reserved_rates
                    print(combination)
                    # Add reserved rates from the paths of other devices in the combination
                    for other_path in combination:
                        other_device = next(device for device in other_devices if flow[device]['paths'].count(other_path))
                        if Check_edge_inpath(other_path[0], i, j):
                            combination_reserved_rates += flow[other_device]['reserved_rates']
                    # Calculate the cost for the current combination
                    # Only consider the reservation value if it's less than or equal to the bandwidth
                    if combination_reserved_rates <= resource_graph[i][j]['bandwidth']:
                        r_ij_var = f'r_{i}_{j}'  # Changed to f-string for consistency
                        if r_ij_var not in objective_dict:
                            objective_dict[r_ij_var] = 0
                        objective_dict[r_ij_var] += combination_reserved_rates
                    else:
                        return 'inf', None  # Changed to return a tuple for consistency

    # Convert the dictionary to the format required by CPLEX
    objective = [(var, cost) for var, cost in objective_dict.items()]

    # Set the objective function to minimize the weighted amount of allocated rate
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.objective.set_linear(objective)
    info(prob)
    # Solve the problem
    prob.solve()
    print('uuuu')

    # Return the solution status and variables
    return prob.solution.get_status(), prob.solution.get_values()

   except CplexError as e:
       print(f"Cplex error: {e}")
       return None, None  # Changed to return a tuple for consistency

# if __name__ == "__main__":
result = solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm, 1500, 1500)
print(result)
