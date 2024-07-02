from docplex.mp.model import Model
#from docplex.mp.error_handler import DocplexError
from docplex.mp.utils import DOcplexException
from FindPaths import sorted_paths
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice, product
from collections import OrderedDict
import re
import cplex
print(cplex.__version__)

def is_valid_variable_name(name):
    # Variable names must start with a letter or underscore and contain only alphanumeric characters and underscores
    return re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) is not None

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
def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm=1, sigma=1500, rho=1500):
   try:
        # Create a docplex model
        model = Model(name="OptimalPath")
        
        # Dictionary to keep track of created variables
        created_vars = {}

        def get_or_create_var(model, var_name, var_type):
            if var_name not in created_vars:
                if var_type == 'binary':
                    created_vars[var_name] = model.binary_var(name=var_name)
                elif var_type == 'continuous':
                    created_vars[var_name] = model.continuous_var(name=var_name)
            return created_vars[var_name]

        # Decision variables
        edge_variables_binary = set()
        rate_variables = set()
        r_min_variables = set()
        x_variables = set()
        r_reservation_variables = set()
        c_ij_variables = set()

        tau = 1  # Assuming 'tau' is the time needed to receive 1 packet in the queue ~ here = 1 ms / in the forwarding edges

        # Add edge variables
        for edge in resource_graph.edges:
            i, j = edge
            edge_name = 'x_{}_{}'.format(i, j)
            rate_name = 'r_{}_{}'.format(i, j)
            r_min_name = 'r_min_{}_{}'.format(i, j)
            r_reservation_name = 'r_reservation_{}_{}'.format(i, j)
            c_ij_name = 'c_ij_{}_{}'.format(i, j)

            edge_variables_binary.add(edge_name)
            rate_variables.add(rate_name)
            r_min_variables.add(r_min_name)
            x_variables.add(edge_name)
            r_reservation_variables.add(r_reservation_name)
            c_ij_variables.add(c_ij_name)

        # Convert sets to lists
        edge_variables_binary = list(edge_variables_binary)
        rate_variables = list(rate_variables)
        r_min_variables = list(r_min_variables)
        x_variables = list(x_variables)
        r_reservation_variables = list(r_reservation_variables)
        c_ij_variables = list(c_ij_variables)

        # Add all variables to docplex model
        for var in edge_variables_binary + x_variables:
            get_or_create_var(model, var, 'binary')
        for var in rate_variables + r_min_variables + r_reservation_variables + c_ij_variables:
            get_or_create_var(model, var, 'continuous')

        # The set K of existing flows
        flow = {}
        for device in field_devices_delta:
            flow[device] = {}
            flow[device]['paths'] = paths[device]
            flow[device]['deadline'] = field_devices_delta[device]
            flow[device]['burst'] = sigma
            flow[device]['rate'] = rho * tau
            flow[device]['reserved_rates'] = rho * tau + sigma

        # Add constraints for total reserved rates and bandwidth
        check = []
        for i, j in resource_graph.edges:
            expr = []  # Constraint 1: Total reserved rates on edge (i, j) should not exceed the link bandwidth
            for device in field_devices:
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    if Check_edge_inpath(path_nodes, i, j):
                        rate_var_name = 'r_{}_{}'.format(i, j)
                        if rate_var_name not in rate_variables:
                            print(f"Error: {rate_var_name} not found in rate_variables")
                        rate_var = created_vars[rate_var_name]

                        if rate_var_name not in check:
                            check.append(rate_var_name)
                            # Constraint: r_ij <= reserved_rates
                            model.add_constraint(
                                rate_var <= flow[device]['reserved_rates'],
                                ctname=f"constraint_upper_{rate_var_name}"
                            )
                            # Constraint: r_ij >= rate
                            model.add_constraint(
                                rate_var >= flow[device]['rate'],
                                ctname=f"constraint_lower_{rate_var_name}"
                            )
                            # Accumulate rate variables for bandwidth constraint
                            expr.append(rate_var)
                            break

            # Total reserved rates on edge (i, j) should not exceed the link bandwidth
            model.add_constraint(
                model.sum(expr) <= resource_graph[i][j]['bandwidth'],
                ctname=f"bandwidth_constraint_{i}_{j}"
            )

        num_edges = resource_graph.number_of_edges()
        all_min_rate = {edge: idx for idx, edge in enumerate(resource_graph.edges)}

        # Create a mapping from edges to integer indices
        edge_to_index = {edge: idx for idx, edge in enumerate(resource_graph.edges)}

        # Add constraints for r_min_ij (point 4 + 6)
        for edge in resource_graph.edges:
            i, j = edge
            min_rate = []
            for device in field_devices_delta:
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    if Check_edge_inpath(path_nodes, i, j):
                        min_rate.append(flow[device]['rate'])  # Rate value for the flow
                        break
            if min_rate:
                r_min_var_name = 'r_min_{}_{}'.format(i, j)
                r_min_var = created_vars[r_min_var_name]
                # Constraint: r_ij^min = min { r_ij^k : k ∈ P(i, j) }
                model.add_constraint(
                    r_min_var == min(min_rate),
                    ctname=f"r_min_constraint_{i}_{j}"
                )
                # Constraint: r_ij^min >= rho
                model.add_constraint(
                    r_min_var >= rho,
                    ctname=f"r_min_lower_bound_{i}_{j}"
                )
                all_min_rate[i,j] = min(min_rate)

        # Add constraints for reservable capacity c_ij
        for i, j in resource_graph.edges:
            reserved_capacity = []
            for device in field_devices_delta:
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    if Check_edge_inpath(path_nodes, i, j):
                        c_ij_name = 'c_ij_{}_{}'.format(i, j)
                        c_ij_var = created_vars[c_ij_name]

                        # Constraint: r_ij = sum{r_k_ij : k ∈ P(i, j)}
                        model.add_constraint(
                            c_ij_var >= flow[device]['reserved_rates'],
                            ctname=f"c_ij_lower_bound_{i}_{j}_{device}"
                        )
                        # Add constraint: c_ij ≤ w_ij - r_ij
                        if not reserved_capacity:
                            model.add_constraint(
                                c_ij_var == resource_graph[i][j]['bandwidth'],
                                ctname=f"c_ij_initial_{i}_{j}"
                            )
                        else:
                            reserved_capacity_summation = sum(reserved_capacity)
                            model.add_constraint(
                                c_ij_var <= resource_graph[i][j]['bandwidth'] - reserved_capacity_summation,
                                ctname=f"c_ij_upper_bound_{i}_{j}"
                            )

                        if flow[device]['reserved_rates'] <= (resource_graph[i][j]['bandwidth'] - sum(reserved_capacity)):
                            reserved_capacity.append(flow[device]['reserved_rates'])

                        # Constraints: 0 ≤ r_ij ≤ c_ij * x_ij (point 6)
                        model.add_constraint(
                            c_ij_var >= 0,
                            ctname=f"c_ij_non_negative_{i}_{j}"
                        )
                        x_ij_var = created_vars['x_{}_{}'.format(i, j)]
                        model.add_constraint(
                            c_ij_var <= resource_graph[i][j]['bandwidth'] * x_ij_var,
                            ctname=f"c_ij_upper_bound_x_{i}_{j}"
                        )

                        # Add constraints: ρ ≤ r_min_ij ≤ r_ij + c_max * (1 - x_ij)
                        r_min_var = created_vars['r_min_{}_{}'.format(i, j)]
                        r_ij_var = created_vars['r_{}_{}'.format(i, j)]
                        c_max_variable = resource_graph[i][j]['bandwidth']
                        model.add_constraint(
                            r_min_var >= rho,
                            ctname=f"r_min_lower_bound_{i}_{j}"
                        )
                        model.add_constraint(
                            r_min_var <= r_ij_var + c_max_variable * (1 - x_ij_var),
                            ctname=f"r_min_upper_bound_{i}_{j}"
                        )

        # Scheduling algorithm constraints (point 7)
        if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP) scheduling algorithm
            for device in field_devices:
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path

                    # Determine the minimum rate on the path
                    path_min_rate = [
                        all_min_rate[key] if key in all_min_rate else all_min_rate[reversed_key]
                        for i in range(len(path_nodes) - 1)
                        for key, reversed_key in [(tuple(path_nodes[i:i + 2]), tuple(path_nodes[i:i + 2])[::-1])]
                        if key in all_min_rate or reversed_key in all_min_rate
                    ]

                    if not path_min_rate:
                        continue

                    t = sigma / min(path_min_rate)
                    l_ij = 0.0005  # Propagation delay in ms
                    n_i = 40 / 1000  # Node delay in ms
                    r_ij = flow[device]['rate']
                    L = 1500  # MTU fixed to 1500 bytes

                    wcd_vars = []
                    wcd_coeffs = []

                    for i in range(len(path_nodes) - 1):
                        j = i + 1
                        key = (path_nodes[i], path_nodes[j])
                        w_ij = resource_graph[key[0]][key[1]]['bandwidth']  # Available bandwidth for the edge

                        s_ij_var = get_or_create_var(model, f"s_{path_nodes[i]}_{path_nodes[j]}", 'continuous')
                        x_ij_var = get_or_create_var(model, f"x_{path_nodes[i]}_{path_nodes[j]}", 'binary')

                        # Add constraint s_ij * r_ij >= 1.0 (since x_ij is binary, x_ij^2 is just x_ij)
                        model.add_constraint(
                            s_ij_var * r_ij >= 1.0,
                            ctname=f"s_ij_constraint_{path_nodes[i]}_{path_nodes[j]}"
                        )

                        # Add constraint s_ij >= 0
                        model.add_constraint(
                            s_ij_var >= 0,
                            ctname=f"s_ij_non_negative_{path_nodes[i]}_{path_nodes[j]}"
                        )

                        # Define t as a continuous variable
                        t_var = get_or_create_var(model, f"t_{device}_{i}", 'continuous')

                        # Add constraint t >= 0
                        model.add_constraint(
                            t_var >= 0,
                            ctname=f"t_non_negative_{device}_{i}"
                        )

                        # Calculate theta_ij
                        theta_ij = L * s_ij_var + (L / w_ij) * x_ij_var
                        wcd_vars.extend([s_ij_var, x_ij_var])
                        wcd_coeffs.extend([L, (L / w_ij) + l_ij + n_i])

                    # Add the constant delay `t` to the wcd constraint directly
                    wcd_coeffs = [coeff for coeff in wcd_coeffs]
                    wcd_rhs = flow[device]['deadline'] - t  # Subtract t from the right-hand side

                    # Add constraint for worst-case delay
                    model.add_constraint(
                        model.sum(var * coeff for var, coeff in zip(wcd_vars, wcd_coeffs)) >= wcd_rhs,
                        ctname=f"wcd_constraint_{device}"
                    )

        elif scheduling_algorithm == 2:  # Group-Based (GB) scheduling algorithm
         for device in field_devices:
            H = 0
            for path in flow[device]['paths']:
                path_nodes = path[0]  # List of nodes in the path

                # Determine the minimum rate on the path
                path_min_rate = [
                    all_min_rate[key] if key in all_min_rate else all_min_rate[reversed_key]
                    for i in range(len(path_nodes) - 1)
                    for key, reversed_key in [(tuple(path_nodes[i:i + 2]), tuple(path_nodes[i:i + 2])[::-1])]
                    if key in all_min_rate or reversed_key in all_min_rate
                ]
                t = sigma / min(path_min_rate)
                l_ij = 0.0005  # Propagation delay in ms
                n_i = 40 / 1000  # Node delay in ms
                L = 1500  # The MTU L is fixed to 1500 bytes.
                r_ij = flow[device]['rate']
                wcd = t

                wcd_vars = []
                wcd_coeffs = []

                for i in range(len(path_nodes) - 1):
                    j = i + 1
                    key = (path_nodes[i], path_nodes[j])
                    w_ij = resource_graph[key[0]][key[1]]['bandwidth']  # Available bandwidth for the edge

                    s_ij_var = get_or_create_var(model, f"s_{path_nodes[i]}_{path_nodes[j]}", 'continuous')
                    x_ij_var = get_or_create_var(model, f"x_{path_nodes[i]}_{path_nodes[j]}", 'binary')

                    # Add constraint s_ij * r_ij >= 1.0
                    model.add_constraint(
                        s_ij_var * r_ij >= 1.0,
                        ctname=f"s_ij_constraint_{path_nodes[i]}_{path_nodes[j]}_{device}_{H}"
                    )

                    # Add constraint s_ij >= 0
                    model.add_constraint(
                        s_ij_var >= 0,
                        ctname=f"s_ij_non_negative_{path_nodes[i]}_{path_nodes[j]}_{device}_{H}"
                    )

                    # Append to wcd variables and coefficients
                    wcd_vars.extend([s_ij_var, x_ij_var])
                    wcd_coeffs.extend([6 * L, 2 * L / w_ij + l_ij + n_i])

                # Add the constant delay `t` to the wcd constraint directly
                wcd_coeffs = [coeff for coeff in wcd_coeffs]
                wcd_rhs = flow[device]['deadline'] - t  # Subtract t from the right-hand side

                # Add constraint for worst-case delay
                model.add_constraint(
                    model.sum(var * coeff for var, coeff in zip(wcd_vars, wcd_coeffs)) >= wcd_rhs,
                    ctname=f"wcd_constraint_{device}_{H}"
                )

                H += 1

        elif scheduling_algorithm == 3:  # Weakly Rate-Proportional (WRP) algorithm
         for device in field_devices:
            H = 0
            for path in flow[device]['paths']:
                path_nodes = path[0]  # List of nodes in the path

                # Determine the minimum rate on the path
                path_min_rate = [
                    all_min_rate[key] if key in all_min_rate else all_min_rate[reversed_key]
                    for i in range(len(path_nodes) - 1)
                    for key, reversed_key in [(tuple(path_nodes[i:i + 2]), tuple(path_nodes[i:i + 2])[::-1])]
                    if key in all_min_rate or reversed_key in all_min_rate
                ]
                t = sigma / min(path_min_rate)
                l_ij = 0.0005  # Propagation delay in ms
                n_i = 40 / 1000  # Node delay in ms
                L = 1500  # The MTU L is fixed to 1500 bytes.
                r_ij = flow[device]['rate']
                delay_slack = flow[device]['deadline'] - t
                constraint_sum = 0

                wcd_vars = []
                wcd_coeffs = []
                delay_slack_vars = []
                delay_slack_coeffs = []

                for k in range(len(path_nodes) - 1):
                    i = path_nodes[k]
                    j = path_nodes[k + 1]
                    s_ij_var = get_or_create_var(model, f"s_{i}_{j}", 'continuous')
                    x_ij_var = get_or_create_var(model, f"x_{i}_{j}", 'binary')

                    # Add constraint s_ij * r_ij >= x_ij^2
                    model.add_constraint(
                        s_ij_var * r_ij >= x_ij_var,
                        ctname=f"s_ij_constraint_{i}_{j}_{device}_{H}"
                    )

                    # Add constraint s_ij >= 0
                    model.add_constraint(
                        s_ij_var >= 0,
                        ctname=f"s_ij_non_negative_{i}_{j}_{device}_{H}"
                    )

                    P = flow_number(i, j, field_devices, flow)

                    wcd_vars.extend([s_ij_var, x_ij_var])
                    wcd_coeffs.extend([6 * L, 2 * L / resource_graph[i][j]['bandwidth'] + l_ij + n_i])

                    delay_slack_vars.append(x_ij_var)
                    delay_slack_coeffs.append(
                        L / r_ij + (P - 1) * (L / resource_graph[i][j]['bandwidth']) + l_ij + n_i)

                    constraint_sum += (L / resource_graph[i][j]['bandwidth'])

                # Add constraint for worst-case delay
                model.add_constraint(
                    model.sum(var * coeff for var, coeff in zip(wcd_vars, wcd_coeffs)) >= flow[device]['deadline'],
                    ctname=f"wcd_constraint_{device}_{H}"
                )

                # Add constraint for delay slack
                model.add_constraint(
                    model.sum(var * coeff for var, coeff in zip(delay_slack_vars, delay_slack_coeffs)) <= delay_slack,
                    ctname=f"delay_slack_constraint_{device}_{H}"
                )

                H += 1

        elif scheduling_algorithm == 4:  # Frame-Based (FB) scheduler algorithm
         for device in field_devices:
            H = 0
            for path in flow[device]['paths']:
                path_nodes = path[0]  # List of nodes in the path
                # Determine the minimum rate on the path
                path_min_rate = [
                    all_min_rate[key] if key in all_min_rate else all_min_rate[reversed_key]
                    for i in range(len(path_nodes) - 1)
                    for key, reversed_key in [(tuple(path_nodes[i:i + 2]), tuple(path_nodes[i:i + 2])[::-1])]
                    if key in all_min_rate or reversed_key in all_min_rate
                ]
                t = sigma / min(path_min_rate)
                l_ij = 0.0005  # Propagation delay in ms
                n_i = 40 / 1000  # Node delay in ms
                L = 1500  # The MTU L is fixed to 1500 bytes.
                r_ij = flow[device]['rate']
                delay_slack = flow[device]['deadline'] - t
                constraint_sum = 0

                wcd_vars = []
                wcd_coeffs = []
                delay_slack_vars = []
                delay_slack_coeffs = []

                for k in range(len(path_nodes) - 1):
                    i = path_nodes[k]
                    j = path_nodes[k + 1]
                    w_ij = resource_graph[i][j]['bandwidth']  # Available bandwidth for the edge

                    v_ij_var = get_or_create_var(model, f"v_{i}_{j}", 'continuous')
                    z_ij_var = get_or_create_var(model, f"z_{i}_{j}", 'continuous')
                    s_ij_var = get_or_create_var(model, f"s_{i}_{j}", 'continuous')
                    x_ij_var = get_or_create_var(model, f"x_{i}_{j}", 'binary')

                    P = flow_number(i, j, field_devices, flow)

                    # Frame-Based (FB) scheduler algorithm constraints
                    model.add_constraint(
                        v_ij_var >= L * s_ij_var - (L / w_ij),
                        ctname=f"v_ij_constraint1_{i}_{j}_{device}_{H}"
                    )
                    model.add_constraint(
                        v_ij_var >= (L / min(path_min_rate)) * x_ij_var - (L * r_ij / (w_ij * min(path_min_rate))),
                        ctname=f"v_ij_constraint2_{i}_{j}_{device}_{H}"
                    )
                    model.add_constraint(
                        v_ij_var >= 0,
                        ctname=f"v_ij_non_negative_{i}_{j}_{device}_{H}"
                    )
                    model.add_constraint(
                        s_ij_var >= 0,
                        ctname=f"s_ij_non_negative_{i}_{j}_{device}_{H}"
                    )
                    model.add_constraint(
                        s_ij_var * r_ij >= x_ij_var,
                        ctname=f"s_ij_constraint_{i}_{j}_{device}_{H}"
                    )
                    model.add_constraint(
                        z_ij_var >= 1 / min(path_min_rate),
                        ctname=f"z_ij_constraint1_{i}_{j}_{device}_{H}"
                    )
                    model.add_constraint(
                        z_ij_var >= s_ij_var,
                        ctname=f"z_ij_constraint2_{i}_{j}_{device}_{H}"
                    )

                    wcd_vars.extend([s_ij_var, x_ij_var, v_ij_var])
                    wcd_coeffs.extend([L, (L / w_ij) * P, 1])

                    delay_slack_vars.extend([x_ij_var])
                    delay_slack_coeffs.extend([L / r_ij + (P - 1) * (L / w_ij) + l_ij + n_i])

                    constraint_sum += (L / w_ij) * x_ij_var
                    constraint_sum += (L / w_ij) * (w_ij - r_ij) * z_ij_var

                # Add constraint for worst-case delay
                model.add_constraint(
                    model.sum(var * coeff for var, coeff in zip(wcd_vars, wcd_coeffs)) >= flow[device]['deadline'],
                    ctname=f"wcd_constraint_{device}_{H}"
                )

                # Add constraint for delay slack
                model.add_constraint(
                    model.sum(var * coeff for var, coeff in zip(delay_slack_vars, delay_slack_coeffs)) <= delay_slack,
                    ctname=f"delay_slack_constraint_{device}_{H}"
                )

                H += 1

        # Objective function: min ∑((i,j)∈A)〖f_ij * r_ij〗
        objective_dict = {}
        for i, j in resource_graph.edges:
            for primary_device in field_devices:
                primary_device_paths = flow[primary_device]['paths']
                for primary_path in primary_device_paths:
                    total_reserved_rates = 0
                    if Check_edge_inpath(primary_path[0], i, j):
                        total_reserved_rates += flow[primary_device]['reserved_rates']
                    
                    other_devices = [device for device in field_devices if device != primary_device]
                    combinations_iterator = product(*[flow[device]['paths'] for device in other_devices])
                    
                    for combination in islice(combinations_iterator, 10):  # Limit to first 10 combinations
                        combination_reserved_rates = total_reserved_rates
                        for other_path in combination:
                            other_device = next(device for device in other_devices if flow[device]['paths'].count(other_path))
                            if Check_edge_inpath(other_path[0], i, j):
                                combination_reserved_rates += flow[other_device]['reserved_rates']
                        
                        if combination_reserved_rates <= resource_graph[i][j]['bandwidth']:
                            r_ij_var = f'r_{i}_{j}'
                            objective_dict[r_ij_var] = objective_dict.get(r_ij_var, 0) + combination_reserved_rates
                        else:
                            return 'inf', None  # Arbitrarily high cost if the reservation exceeds bandwidth

        # Set the objective function to minimize the weighted amount of allocated rate
        objective_expr = model.sum(created_vars[var] * cost for var, cost in objective_dict.items())
        model.minimize(objective_expr)

        # Solve the problem
        solution = model.solve()

        # Return the solution status and variables
        if solution:
            return solution.solve_status, {var_name: var.solution_value for var_name, var in created_vars.items()}
        else:
            return None, None

   except DOcplexException as e:
        print(f"Docplex error: {e}")
        return None, None

# if __name__ == "__main__":
result = solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm, 1500, 1500)
#print(result)
