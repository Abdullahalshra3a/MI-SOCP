import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
from collections import OrderedDict
import re

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



def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm = 1, sigma=1500, rho=1500):
    try:
        # Ensure the graph is directed
        #if not isinstance(resource_graph, nx.DiGraph):
         #   resource_graph = nx.DiGraph(resource_graph)

        # Create a CPLEX problem
        prob = Cplex()

        # Set the problem type to MIQCP
        prob.set_problem_type(Cplex.problem_type.MIQCP)

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

        # Add all variables to CPLEX model
        prob.variables.add(names=edge_variables_binary, types=[prob.variables.type.binary] * len(edge_variables_binary))
        prob.variables.add(names=rate_variables, types=[prob.variables.type.continuous] * len(rate_variables))
        prob.variables.add(names=r_min_variables, types=[prob.variables.type.continuous] * len(r_min_variables))
        prob.variables.add(names=x_variables, types=[prob.variables.type.binary] * len(x_variables))
        prob.variables.add(names=r_reservation_variables,
                           types=[prob.variables.type.continuous] * len(r_reservation_variables))
        prob.variables.add(names=c_ij_variables, types=[prob.variables.type.continuous] * len(c_ij_variables))
        """""""""
        # Debug: Print variable names to verify
        print("Edge Variables Binary: \n", edge_variables_binary)
        print("Rate Variables: \n", rate_variables)
        print("R Min Variables:\n", r_min_variables)
        print("X Variables:\n", x_variables)
        print("R Reservation Variables:\n", r_reservation_variables)
        print("C IJ Variables:\n", c_ij_variables)
        """""""""
        # The set K of existing flows
        flow = {}
        for device in field_devices_delta:
            flow[device] = {}
            flow[device]['paths'] = paths[device]
            flow[device]['deadline'] = field_devices_delta[device]
            flow[device]['burst'] = sigma
            flow[device]['rate'] = rho * tau
            flow[device]['reserved_rates'] = rho * tau + sigma
            #print(flow[device])

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
                        rate_var_index = rate_variables.index(rate_var_name)
                        rate_var = rate_variables[rate_var_index]

                        if rate_var not in check:
                          check.append(rate_var)
                          # Constraint: r_ij <= reserved_rates
                          prob.linear_constraints.add(
                            lin_expr=[[[rate_var], [1.0]]],
                            senses=['L'],
                            rhs=[flow[device]['reserved_rates']]
                          )
                          # Constraint: r_ij >= rate
                          prob.linear_constraints.add(
                            lin_expr=[[[rate_var], [1.0]]],
                            senses=['G'],
                            rhs=[flow[device]['rate']]
                          )
                          # Accumulate rate variables for bandwidth constraint
                          expr.append(rate_var)
                          break

            # Total reserved rates on edge (i, j) should not exceed the link bandwidth
            prob.linear_constraints.add(
                lin_expr=[[[var], [1.0]] for var in expr],
                senses=['L'],
                rhs=[resource_graph[i][j]['bandwidth']]
            )
        """""""""
        # Routing constraints (point 5)# Routing constraints for an undirected graph
        # Routing constraints for an undirected graph
        for node in resource_graph.nodes:
            node_type = resource_graph.nodes[node].get('type', '')
            print(f"Processing node: {node}, Type: {node_type}")

            if node_type == 'N':
                connected_edges = list(resource_graph.edges(node))

                print(f"Connected edges for node {node}: {connected_edges}")

                # Generate variable names ensuring consistency (alphabetical order in naming)
                edge_vars = [generate_edge_variable_name(edge) for edge in connected_edges]

                # Validate variable names
                for var in edge_vars:
                    if not is_valid_variable_name(var):
                        raise ValueError(f"Invalid variable name: {var}")

                # Generate coefficients for the linear expression
                coefficients = [1 if edge[1] == node else -1 for edge in connected_edges]

                print(f"Node: {node}, Edge Variables: {edge_vars}")
                print(f"Coefficients: {coefficients}")

                # Flow conservation constraint
                prob.linear_constraints.add(
                    lin_expr=[[edge_vars, coefficients]],
                    senses=['E'],
                    rhs=[-1 if node == 's' else (1 if node == 'd' else 0)],
                    names=['c_flow_conservation_{}'.format(node)]
                )

        print("All variable names are valid and constraints added successfully.")
        """""""""




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
                r_min_var_index = r_min_variables.index(r_min_var_name)
                r_min_var = r_min_variables[r_min_var_index]
                #print(min(min_rate))
                # Constraint: r_ij^min = min { r_ij^k : k ∈ P(i, j) }
                prob.linear_constraints.add(
                    lin_expr=[[[r_min_var], [1.0]]],
                    senses=['E'],
                    rhs=[min(min_rate)]
                )
                # Constraint: r_ij^min >= rho
                prob.linear_constraints.add(
                    lin_expr=[[[r_min_var], [1.0]]],
                    senses=['G'],
                    rhs=[rho]
                )
                all_min_rate[i,j] = min(min_rate)

        #print(all_min_rate)
        # Add constraints for reservable capacity c_ij
        for i, j in resource_graph.edges:
            reserved_capacity = []
            for device in field_devices_delta:
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    if Check_edge_inpath(path_nodes, i, j):
                        c_ij_name = 'c_ij_{}_{}'.format(i, j)
                        c_ij_var_index = c_ij_variables.index(c_ij_name)
                        c_ij_var = c_ij_variables[c_ij_var_index]

                        # Constraint: r_ij = sum{r_k_ij : k ∈ P(i, j)}
                        prob.linear_constraints.add(
                            lin_expr=[[[c_ij_var], [1.0]]],
                            senses=['G'],
                            rhs=[flow[device]['reserved_rates']]
                        )
                        # Add constraint: c_ij ≤ w_ij - r_ij
                        if not reserved_capacity:
                            prob.linear_constraints.add(
                                lin_expr=[[[c_ij_var], [1.0]]],
                                senses=['E'],
                                rhs=[resource_graph[i][j]['bandwidth']]
                            )
                        else:
                            reserved_capacity_summation = sum(reserved_capacity)
                            prob.linear_constraints.add(
                                lin_expr=[[[c_ij_var], [1.0]]],
                                senses=['L'],
                                rhs=[resource_graph[i][j]['bandwidth'] - reserved_capacity_summation]
                            )

                        if flow[device]['reserved_rates'] <= ( resource_graph[i][j]['bandwidth'] - sum(reserved_capacity)):
                            reserved_capacity.append(flow[device]['reserved_rates'])

                        # Constraints: 0 ≤ r_ij ≤ c_ij * x_ij (point 6)
                        prob.linear_constraints.add(
                            lin_expr=[[[c_ij_var], [1.0]]],
                            senses=['G'],
                            rhs=[0]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[c_ij_var, 'x_{}_{}'.format(i, j)], [1.0, 1.0]]],
                            senses=['L'],
                            rhs=[resource_graph[i][j]['bandwidth']]
                        )

                        # Add constraints: ρ ≤ r_min_ij ≤ r_ij + c_max * (1 - x_ij)
                        # Adding the constraint ρ ≤ r_min_ij
                        r_min_var_name = 'r_min_{}_{}'.format(i, j)
                        r_min_var_index = r_min_variables.index(r_min_var_name)
                        r_min_var = r_min_variables[r_min_var_index]
                        prob.linear_constraints.add(
                            lin_expr=[[[r_min_var], [1.0]]],
                            senses=['G'],
                            rhs=[rho]
                        )

                        # Adding the constraint r_min_ij ≤ r_ij + c_max * (1 - x_ij)
                        c_max_variable = resource_graph[i][j]['bandwidth']
                        prob.linear_constraints.add(
                            lin_expr=[[[r_min_var, 'r_{}_{}'.format(i, j), 'x_{}_{}'.format(i, j)],
                                       [1.0, -1.0, c_max_variable]]],
                            senses=['L'],
                            rhs=[c_max_variable]
                        )
        # Scheduling algorithm constraints (point 7)
        path_min_rate = []
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

                        s_ij_var = f"s_{path_nodes[i]}_{path_nodes[j]}"
                        x_ij_var = f"x_{path_nodes[i]}_{path_nodes[j]}"
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        # Add constraint s_ij * r_ij >= 1.0 (since x_ij is binary, x_ij^2 is just x_ij)
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [r_ij]]],
                            senses=['G'],
                            rhs=[1.0]
                        )

                        # Add constraint s_ij >= 0
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )

                        # Define t as a continuous variable
                        t_var = f"t_{device}_{i}"
                        prob.variables.add(names=[t_var], types=[prob.variables.type.continuous])

                        # Add constraint t >= 0
                        prob.linear_constraints.add(
                            lin_expr=[[[t_var], [1.0]]],
                            senses=['G'],
                            rhs=[0.0]
                        )

                        # Calculate theta_ij
                        s_ij_index = prob.variables.get_indices(s_ij_var)
                        x_ij_index = prob.variables.get_indices(x_ij_var)
                        theta_ij = L * s_ij_index + (L / w_ij) * x_ij_index
                        wcd_vars.extend([s_ij_var, x_ij_var])
                        wcd_coeffs.extend([L, (L / w_ij) + l_ij + n_i])

                    # Add the constant delay `t` to the wcd constraint directly
                    wcd_coeffs = [coeff for coeff in wcd_coeffs]
                    wcd_rhs = flow[device]['deadline'] - t  # Subtract t from the right-hand side

                    # Add constraint for worst-case delay
                    prob.linear_constraints.add(
                        lin_expr=[[wcd_vars, wcd_coeffs]],
                        senses=['G'],
                        rhs=[wcd_rhs]
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

                        s_ij_var = f"s_{path_nodes[i]}_{path_nodes[j]}"
                        x_ij_var = f"x_{path_nodes[i]}_{path_nodes[j]}"
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        # Add constraint s_ij * r_ij >= 1.0
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [r_ij]]],
                            senses=['G'],
                            rhs=[1.0]
                        )

                        # Add constraint s_ij >= 0
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )

                        s_ij_index = prob.variables.get_indices(s_ij_var)
                        x_ij_index = prob.variables.get_indices(x_ij_var)

                        # Append to wcd variables and coefficients
                        wcd_vars.extend([s_ij_var, x_ij_var])
                        wcd_coeffs.extend([6 * L, 2 * L / w_ij + l_ij + n_i])

                    # Add the constant delay `t` to the wcd constraint directly
                    wcd_coeffs = [coeff for coeff in wcd_coeffs]
                    wcd_rhs = flow[device]['deadline'] - t  # Subtract t from the right-hand side

                    # Add constraint for worst-case delay
                    prob.linear_constraints.add(
                        lin_expr=[[wcd_vars, wcd_coeffs]],
                        senses=['G'],
                        rhs=[wcd_rhs]
                    )

                    H += 1
                    #flow[device]['paths'][H][1]['wcd'] = wcd


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
                        s_ij_var = f"s_{i}_{j}"
                        x_ij_var = f"x_{i}_{j}"
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        # Add constraint s_ij * r_ij >= x_ij^2
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [r_ij]]],
                            senses=['G'],
                            rhs=[1.0]  # x_ij^2 becomes a constant 1.0 for binary variables
                        )

                        # Add constraint s_ij >= 0
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )

                        P = flow_number(i, j, field_devices, flow)

                        wcd_vars.extend([s_ij_var, x_ij_var])
                        wcd_coeffs.extend([6 * L, 2 * L / resource_graph[i][j]['bandwidth'] + l_ij + n_i])

                        delay_slack_vars.append(x_ij_var)
                        delay_slack_coeffs.append(
                            L / r_ij + (P - 1) * (L / resource_graph[i][j]['bandwidth']) + l_ij + n_i)

                        constraint_sum += (L / resource_graph[i][j]['bandwidth'])

                    # Add the constant terms to wcd and delay_slack
                    wcd = t
                    for s_ij_var, x_ij_var, wcd_coeff in zip(wcd_vars[::2], wcd_vars[1::2], wcd_coeffs[::2]):
                        wcd += wcd_coeff * prob.variables.get_indices(s_ij_var)
                        wcd += wcd_coeffs[wcd_coeffs.index(wcd_coeff) + 1] * prob.variables.get_indices(x_ij_var)

                    delay_slack -= sum(delay_slack_coeffs)

                    # Add constraint for worst-case delay
                    prob.linear_constraints.add(
                        lin_expr=[[wcd_vars, wcd_coeffs]],
                        senses=['G'],
                        rhs=[flow[device]['deadline']]
                    )

                    # Add constraint for delay slack
                    prob.linear_constraints.add(
                        lin_expr=[[delay_slack_vars, delay_slack_coeffs]],
                        senses=['L'],
                        rhs=[delay_slack]
                    )

                    H += 1
                    #flow[device]['paths'][H][1]['wcd'] = wcd


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
                    w_ij = resource_graph[i][j]['bandwidth']  # Available bandwidth for the edge from node i to node j
                    wcd = t
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

                        v_ij_var = f"v_{i}_{j}"
                        z_ij_var = f"z_{i}_{j}"
                        s_ij_var = f"s_{i}_{j}"
                        x_ij_var = f"x_{i}_{j}"
                        prob.variables.add(names=[v_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[z_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        P = flow_number(i, j, field_devices, flow)

                        # Frame-Based (FB) scheduler algorithm constraints
                        prob.linear_constraints.add(
                            lin_expr=[[[v_ij_var], [1]]],
                            senses=['G'],
                            rhs=[L * prob.variables.get_indices(s_ij_var) - (L / w_ij)]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[v_ij_var], [1]]],
                            senses=['G'],
                            rhs=[(L / min(path_min_rate)) * prob.variables.get_indices(x_ij_var) - (
                                        L * r_ij / (w_ij * min(path_min_rate)))]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[v_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [r_ij]]],
                            senses=['G'],
                            rhs=[prob.variables.get_indices(x_ij_var)]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[z_ij_var], [1]]],
                            senses=['G'],
                            rhs=[1 / min(path_min_rate)]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[z_ij_var], [1]]],
                            senses=['G'],
                            rhs=[prob.variables.get_indices(s_ij_var)]
                        )

                        wcd_vars.extend([s_ij_var, x_ij_var, v_ij_var])
                        wcd_coeffs.extend([L, (L / w_ij) * P, 1])

                        delay_slack_vars.extend([x_ij_var])
                        delay_slack_coeffs.extend([L / r_ij + (P - 1) * (L / w_ij) + l_ij + n_i])

                        constraint_sum += (L / w_ij) * prob.variables.get_indices(x_ij_var)
                        constraint_sum += (L / w_ij) * (w_ij - r_ij) * prob.variables.get_indices(z_ij_var)

                        wcd += (L * prob.variables.get_indices(s_ij_var) * (L / w_ij) * P * prob.variables.get_indices(
                            x_ij_var)) + prob.variables.get_indices(v_ij_var) + l_ij + n_i

                        # Add constraint for worst-case delay
                    prob.linear_constraints.add(
                        lin_expr=[[wcd_vars, wcd_coeffs]],
                        senses=['G'],
                        rhs=[flow[device]['deadline']]
                    )

                    # Add constraint for delay slack
                    prob.linear_constraints.add(
                        lin_expr=[[delay_slack_vars, delay_slack_coeffs]],
                        senses=['L'],
                        rhs=[delay_slack]
                    )

                    H += 1
                    # flow[device]['paths'][H][1]['wcd'] = wcd  # Dictionary with 'wcd' value

        # Objective function: min ∑((i,j)∈A)〖f_ij * r_ij〗
        objective = []
        for i, j in resource_graph.edges:
            total_cost = 0

            # Iterate through each field device
            for primary_device in field_devices:
                primary_device_paths = flow[primary_device]['paths']

                # Consider each path of the primary device
                for primary_path in primary_device_paths:
                    total_reserved_rates = 0

                    # Initialize reserved rates with the primary device's path if it uses edge (i, j)
                    if Check_edge_inpath(primary_path[0], i, j):
                        total_reserved_rates += flow[primary_device]['reserved_rates']

                    # For all other devices, consider their paths without generating all combinations
                    other_devices = [device for device in field_devices if device != primary_device]
                    for other_device in other_devices:
                        other_device_paths = flow[other_device]['paths']
                        for other_path in other_device_paths:
                            if Check_edge_inpath(other_path[0], i, j):
                                total_reserved_rates += flow[other_device]['reserved_rates']

                            # Calculate the cost for the current combination
                            # Only consider the reservation value if it's less than or equal to the bandwidth
                            if total_reserved_rates <= resource_graph[i][j]['bandwidth']:
                                f_ij = reservation_costs.get((i, j),
                                                             1) * total_reserved_rates  # Apply the reservation cost f_ij
                                r_ij_var = 'r_{}_{}'.format(i, j)
                                objective.append((r_ij_var, f_ij))
                            else:
                                return str('inf: Arbitrarily high cost if the reservation exceeds bandwidth')  # Arbitrarily high cost if the reservation exceeds bandwidth

        # Set the objective function to minimize the weighted amount of allocated rate
        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.objective.set_linear(objective)

        #prob.objective.set_linear(objective)
        #prob.objective.set_sense(prob.objective.sense.minimize)

        # Solve the problem
        prob.solve()
        # Return the solution status and variables
        return prob.solution.get_status(), prob.solution.get_values()

    except CplexError as e:
        print(f"Cplex error: {e}")
        return None

# if __name__ == "__main__":
result = solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm, 1500, 1500)
print(result)



