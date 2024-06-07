import cplex
from cplex import Cplex
from cplex.exceptions import CplexError
from FindPaths import sorted_paths
import networkx as nx
import matplotlib.pyplot as plt


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
    return scheduling_algorithm
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

def calculate_cost(resource_graph, i, j, reservation_value):
    bandwidth = resource_graph[i][j]['bandwidth']
    # Only consider the reservation value if it's less than or equal to the bandwidth
    if reservation_value <= bandwidth:
        return reservation_value
    else:
        return float('inf: Arbitrarily high cost if the reservation exceeds bandwidth')  # Arbitrarily high cost if the reservation exceeds bandwidth


def solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm, sigma=1500, rho=1500):
    try:
        # Create a CPLEX problem
        prob = Cplex()

        '''
         # Set the problem type to MI-SOCP
         # Configures the problem type as Max Integer Quadratically Constrained Problem (Max IQCP),
         # which encompasses the Max Integer Second Order Cone Problem (Max ISOCP) since it is a special case.
        '''
        prob.set_problem_type(Cplex.problem_type.MIQCP)

        # Decision variables
        edge_variables_binary = []
        rate_variables = []
        r_min_variables = []
        x_variables = []  # Binary variables for routing
        r_reservation_variables = []  # Continuous rate reservation variables
        c_ij_variables = []  # Continuous reservable capacity variables

        tau = 1  # Assuming 'tau' is the time needed to receive 1 packet in the queue ~ here = 1 ms / in the forwarding edges

        # Add edge variables
        for edge in resource_graph.edges:
            i, j = edge
            edge_variables_binary.append('x_{}_{}'.format(i, j))
            rate_variables.append('r_{}_{}'.format(i, j))
            r_min_variables.append('r_min_{}_{}'.format(i, j))
            x_variables.append('x_{}_{}'.format(i, j))
            r_reservation_variables.append('r_reservation_{}_{}'.format(i, j))
            c_ij_variables.append('c_ij_{}_{}'.format(i, j))

        # Add all variables separately
        prob.variables.add(names=edge_variables_binary, types=[prob.variables.type.binary] * len(edge_variables_binary))
        prob.variables.add(names=rate_variables, types=[prob.variables.type.continuous] * len(rate_variables))
        prob.variables.add(names=r_min_variables, types=[prob.variables.type.continuous] * len(r_min_variables))
        prob.variables.add(names=x_variables, types=[prob.variables.type.binary] * len(x_variables))
        prob.variables.add(names=r_reservation_variables, types=[prob.variables.type.continuous] * len(r_reservation_variables))
        prob.variables.add(names=c_ij_variables, types=[prob.variables.type.continuous] * len(c_ij_variables))

        '''
        The set K of existing flows: each flow in k ∈ K is characterized by its chosen path p(k) between its source (Field device) and its destination (Server),
        its deadline δk, its burst and rate parameters σk and ρk, and its reserved rates r_ij^k for each link (i, j) ∈ p(k).
        '''
        flow = {}
        for device in field_devices_delta:  # We could use this part in the future to handle different burst and rate values in addition to the flow priority.
            flow[device] = {}  # Initialize nested dictionary for each device
            flow[device]['paths'] = paths[device]  # Tuple of lists of paths (e.g., ([0, 1, 3], {'wcd': 0}), ([0, 2, 3], {'wcd': 0}), ([0, 3], {'wcd': 0}))
            flow[device]['deadline'] = field_devices_delta[device]  # Deadline for the flow (delta)
            flow[device]['burst'] = sigma  # Burst value for the flow
            flow[device]['rate'] = rho * tau  # Rate value for the flow
            flow[device]['reserved_rates'] = rho * tau + sigma  # Dictionary containing reserved rates demanded for each link (i, j)


                    # Add constraint: rate_variable = burst + rate * tau
                    # The arrival curve A(τ) = σ+ ρ·τ is affine,  where its two non-negative parameters, σ = 1500 Byte/ms and ρ = 1500/ms Byte are called burst and rate, respectively.
                    #prob.linear_constraints.add(lin_expr=[rate_variables[edge_variables_binary.index('r_{}_{}'.format(i, j))]],senses=['E'],  rhs=[rho * tau + sigma]  )


                    #'''''''''''
                    #With adding a new flow path rate r_ij ,
                    #we should consider the edge bandwidth (wij) as well as the pre-mentioned information,we have the following constraints:
                    #r_ij + ∑_(k ∈ P(i,j))▒r_ij^k   ≤ wij.
                    #r_ij ≥ ρ ∀(i, j) ∈ p .
                    #'''''''''''
        for i, j in resource_graph.edges:
            expr = []  # Constraint 1: Total reserved rates on edge (i, j) should not exceed the link bandwidth
            for device in field_devices:
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    # Add constraint: If link (i, j) is in the path, then flow is using it.
                    if Check_edge_inpath(path_nodes, i, j):
                      # Constraint: r_ij <= reserved_rates
                        prob.linear_constraints.add(
                            lin_expr=[[rate_variables[edge_variables_binary.index('r_{}_{}'.format(i, j))]]],
                            senses=['L'],
                            rhs=[flow[device]['reserved_rates']]
                        )
                        # Constraint: r_ij >= rate
                        prob.linear_constraints.add(
                            lin_expr=[[rate_variables[edge_variables_binary.index('r_{}_{}'.format(i, j))]]],
                            senses=['G'],
                            rhs=[flow[device]['rate']]
                        )
                        # Accumulate rate variables for bandwidth constraint
                        expr.append(1 * rate_variables[edge_variables_binary.index('r_{}_{}'.format(i, j))])
                        break
                      # Total reserved rates on edge (i, j) should not exceed the link bandwidth
                prob.linear_constraints.add(lin_expr=[expr], senses=['L'], rhs=[resource_graph[i][j]['bandwidth']])


        # Routing constraints (point 5)
        for node in resource_graph.nodes:
            if resource_graph.nodes[node]['type'] == 'N':
                incoming_edges = [edge for edge in resource_graph.in_edges(node)]
                outgoing_edges = [edge for edge in resource_graph.out_edges(node)]
                bs_i = ['x_{}_{}'.format(j, i) for j, i in incoming_edges]
                fs_i = ['x_{}_{}'.format(i, j) for i, j in outgoing_edges]

                # Standard flow conservation constraints
                prob.linear_constraints.add(
                    lin_expr=[[bs_i + fs_i, [-1] * len(bs_i) + [1] * len(fs_i)]],
                    senses=['E'],
                    rhs=[-1 if node == 's' else (1 if node == 'd' else 0)],
                    names=['c_flow_conservation_{}'.format(node)]
                )

        num_edges = resource_graph.number_of_edges()

        # Create a 2D list with dimensions corresponding to the number of edges
        all_min_rate = [[None for _ in range(num_edges)] for _ in range(num_edges)]
        # Add constraints for r_min_ij (point 4 + 6)
        for i, j in resource_graph.edges:
            expr_min = []
            min_rate = []
            for path in flow[device]['paths']:
                path_nodes = path[0]  # List of nodes in the path
                if Check_edge_inpath(path_nodes, i, j):
                    min_rate.append(flow[device]['rate'])  # Rate value for the flow
                    break
            expr_min.append(1 * r_min_variables[edge_variables_binary.index('r_min_{}_{}'.format(i, j))])

            # Constraint: r_ij^min = min { r_ij^k : k ∈ P(i, j) }
            prob.linear_constraints.add(
                lin_expr=[expr_min],
                senses=['E'],
                rhs=[min(min_rate)]
            )
            # Constraint: r_ij^min >= rho
            prob.linear_constraints.add(
                lin_expr=[expr_min],
                senses=['G'],
                rhs=[rho]
            )
            all_min_rate[i][j] = min(min_rate)
            # Add constraints for reservable capacity c_ij
            for i, j in resource_graph.edges:
                reserved_capacity = []
                for device in field_devices:
                    for path in flow[device]['paths']:
                        path_nodes = path[0]  # List of nodes in the path
                        if Check_edge_inpath(path_nodes, i, j):
                            # Constraint: r_ij = sum{r_k_ij : k ∈ P(i, j)}
                            prob.linear_constraints.add(
                                lin_expr=[[c_ij_variables[edge_variables_binary.index('c_ij_{}_{}'.format(i, j))]]],
                                senses=['G'],
                                rhs=[flow[device]['reserved_rates']]
                            )
                            # Add constraint: c_ij ≤ w_ij - r_ij
                            if not reserved_capacity:
                                prob.linear_constraints.add(
                                    lin_expr=[[c_ij_variables[edge_variables_binary.index('c_ij_{}_{}'.format(i, j))]]],
                                    senses=['E'],
                                    rhs=[resource_graph[i][j]['bandwidth']]
                                )
                            else:
                                reserved_capacity_summation = sum(reserved_capacity)
                                prob.linear_constraints.add(
                                    lin_expr=[[c_ij_variables[edge_variables_binary.index('c_ij_{}_{}'.format(i, j))]]],
                                    senses=['L'],
                                    rhs=[resource_graph[i][j]['bandwidth'] - reserved_capacity_summation]
                                )

                            if flow[device]['reserved_rates'] <= (
                                    resource_graph[i][j]['bandwidth'] - sum(reserved_capacity)):
                                reserved_capacity.append(flow[device]['reserved_rates'])

                            # Constraints: 0 ≤ r_ij ≤ c_ij * x_ij (point 6)
                            prob.linear_constraints.add(
                                lin_expr=[[rate_variables[edge_variables_binary.index('r_{}_{}'.format(i, j))]]],
                                senses=['G'],
                                rhs=[0]
                            )
                            prob.linear_constraints.add(
                                lin_expr=[[rate_variables[edge_variables_binary.index('r_{}_{}'.format(i, j))],
                                           c_ij_variables[edge_variables_binary.index('c_ij_{}_{}'.format(i, j))],
                                           x_variables[edge_variables_binary.index('x_{}_{}'.format(i, j))]]],
                                senses=['L'],
                                rhs=[c_ij_variables[edge_variables_binary.index('c_ij_{}_{}'.format(i, j))] *
                                     x_variables[edge_variables_binary.index('x_{}_{}'.format(i, j))]]
                            )

                            # Add constraints: ρ ≤ r_min_ij ≤ r_ij + c_max * (1 - x_ij)
                            # Adding the constraint ρ ≤ r_min_ij
                            prob.linear_constraints.add(
                                lin_expr=[[r_min_variables[edge_variables_binary.index('r_min_{}_{}'.format(i, j))]]],
                                senses=['G'],
                                rhs=[rho]
                            )

                            # Adding the constraint r_min_ij ≤ r_ij + c_max * (1 - x_ij)
                            c_max_variable = resource_graph[i][j]['bandwidth']
                            prob.linear_constraints.add(
                                lin_expr=[[r_min_variables[edge_variables_binary.index('r_min_{}_{}'.format(i, j))],
                                           rate_variables[edge_variables_binary.index('r_{}_{}'.format(i, j))],
                                           c_max_variable * (1 - x_variables[
                                               edge_variables_binary.index('x_{}_{}'.format(i, j))])]],
                                senses=['L'],
                                rhs=[rate_variables[
                                         edge_variables_binary.index('r_{}_{}'.format(i, j))] + c_max_variable * (
                                                 1 - x_variables[edge_variables_binary.index('x_{}_{}'.format(i, j))])]
                            )
        '''''''''
         #point 7
        '''''''''
        # Scheduling algorithm constraints (point 7)
        if scheduling_algorithm == 1:  # Strictly Rate-Proportional (SRP) scheduling algorithm
            for device in field_devices:
                H = 0
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    path_min_rate = [all_min_rate[path_nodes[i]][path_nodes[i + 1]] for i in range(len(path_nodes) - 1)]
                    t = sigma / min(path_min_rate)
                    l_ij = propagation_delay / 1000  # Propagation delay converted to ms
                    n_i = 40 / 1000  # Node delay converted to ms
                    wcd = t
                    r_ij = flow[device]['rate']
                    w_ij = resource_graph[i][j]['bandwidth']  # Available bandwidth for the edge from node i to node j
                    L = 1500  # The MTU L is fixed to 1500 bytes.
                    for i in range(len(path_nodes) - 1):
                        j = i + 1
                        s_ij_var = "s_{}_{}".format(path_nodes[i], path_nodes[j])
                        x_ij_var = "x_{}_{}".format(path_nodes[i], path_nodes[j])
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        # Add constraint s_ij * r_ij >= x_ij^2
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [r_ij]]],
                            senses=['G'],
                            rhs=[x_ij_var ** 2]
                        )
                        # Add constraint s_ij >= 0
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[t], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )

                        theta_ij = L * s_ij_var + (L / w_ij) * x_ij_var
                        wcd = wcd + (L * s_ij_var + (L / w_ij) + l_ij + n_i) * x_ij_var
                    prob.linear_constraints.add(
                        lin_expr=[[[wcd], [1]]],
                        senses=['G'],
                        rhs=[flow[device]['deadline']] )
                    H = H + 1
                    flow[device]['paths'][H][1]['wcd'] = wcd  # Save the calculated WCD value



        elif scheduling_algorithm == 2:  # Group-Based (GB) scheduling algorithm
            for device in field_devices:
                H = 0
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    path_min_rate = [all_min_rate[path_nodes[i]][path_nodes[i + 1]] for i in range(len(path_nodes) - 1)]
                    t = sigma / min(path_min_rate)
                    l_ij = propagation_delay / 1000  # Propagation delay converted to ms
                    n_i = 40 / 1000  # Node delay converted to ms
                    w_ij = resource_graph[i][j]['bandwidth']  # Available bandwidth for the edge from node i to node j
                    L = 1500  # The MTU L is fixed to 1500 bytes.
                    wcd = t
                    r_ij = flow[device]['rate']
                    for i in range(len(path_nodes) - 1):
                        j = i + 1
                        s_ij_var = "s_{}_{}".format(path_nodes[i], path_nodes[j])
                        x_ij_var = "x_{}_{}".format(path_nodes[i], path_nodes[j])
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [r_ij]]],
                            senses=['G'],
                            rhs=[x_ij_var ** 2]
                        )

                        # Add constraint s_ij >= 0
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )
                        wcd = wcd + (6 * L * s_ij_var + (2 * L / w_ij + l_ij + n_i) * x_ij_var)
                    prob.linear_constraints.add(
                        lin_expr=[[[wcd], [1]]],
                        senses=['G'],
                        rhs=[flow[device]['deadline']])
                    H = H + 1
                    flow[device]['paths'][H][1]['wcd']= wcd

        elif scheduling_algorithm == 3:  # Weakly Rate-Proportional (WRP) algorithm
            for device in field_devices:
                H = 0
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    path_min_rate = [all_min_rate[path_nodes[i]][path_nodes[i + 1]] for i in range(len(path_nodes) - 1)]
                    t = sigma / min(path_min_rate)
                    l_ij = propagation_delay / 1000  # Propagation delay converted to ms
                    n_i = 40 / 1000  # Node delay converted to ms
                    w_ij = resource_graph[i][j]['bandwidth']  # Available bandwidth for the edge from node i to node j
                    L = 1500  # The MTU L is fixed to 1500 bytes.
                    wcd = t
                    r_ij = flow[device]['rate']
                    delay_slack = flow[device]['deadline'] - t
                    constraint_sum = 0
                    for i in range(len(path_nodes) - 1):
                        j = i + 1
                        s_ij_var = "s_{}_{}".format(path_nodes[i], path_nodes[j])
                        x_ij_var = "x_{}_{}".format(path_nodes[i], path_nodes[j])
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [r_ij]]],
                            senses=['G'],
                            rhs=[x_ij_var ** 2]
                        )

                        # Add constraint s_ij >= 0
                        prob.linear_constraints.add(
                            lin_expr=[[[s_ij_var], [1]]],
                            senses=['G'],
                            rhs=[0]
                        )
                        P[path[i]][path[j]] = flow_number(path[i], path[j], field_devices, flow)
                        wcd = wcd + (L * s_ij_var + ((L / w_ij) * P[path[i]][path[j]]) * x_ij_var) + l_ij + n_i
                        delay_slack = delay_slack - (L / r_ij + (P[path[i]][path[j]] - 1) * (L / w_ij) + l_ij + n_i)
                        constraint_sum += (L / w_ij) * x_ij_var
                    prob.linear_constraints.add(
                        lin_expr=[[[wcd], [1]]],
                        senses=['G'],
                        rhs=[flow[device]['deadline']]
                    )
                    prob.linear_constraints.add(
                        lin_expr=[[[constraint_sum], [1]]],
                        senses=['L'],
                        rhs=[delay_slack])
                    H = H + 1
                    flow[device]['paths'][H][1]['wcd']= wcd

        elif scheduling_algorithm == 4:  # Frame-Based (FB) scheduler algorithm
            for device in field_devices:
                H = 0
                for path in flow[device]['paths']:
                    path_nodes = path[0]  # List of nodes in the path
                    path_min_rate = [all_min_rate[path_nodes[i]][path_nodes[i + 1]] for i in range(len(path_nodes) - 1)]
                    t = sigma / min(path_min_rate)
                    l_ij = propagation_delay / 1000  # Propagation delay converted to ms
                    n_i = 40 / 1000  # Node delay converted to ms
                    w_ij = resource_graph[i][j]['bandwidth']  # Available bandwidth for the edge from node i to node j
                    L = 1500  # The MTU L is fixed to 1500 bytes.
                    wcd = t
                    r_ij = flow[device]['rate']
                    delay_slack = flow[device]['deadline'] - t
                    constraint_sum = 0
                    for i in range(len(path_nodes) - 1):
                        j = i + 1
                        v_ij_var = "v_{}_{}".format(path_nodes[i], path_nodes[j])
                        z_ij_var = "z_{}_{}".format(path_nodes[i], path_nodes[j])
                        s_ij_var = "s_{}_{}".format(path_nodes[i], path_nodes[j])
                        x_ij_var = "x_{}_{}".format(path_nodes[i], path_nodes[j])
                        prob.variables.add(names=[v_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[z_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[s_ij_var], types=[prob.variables.type.continuous])
                        prob.variables.add(names=[x_ij_var], types=[prob.variables.type.binary])

                        P[path_nodes[i]][path_nodes[j]] = flow_number(path_nodes[i], path_nodes[j], field_devices, flow)

                        # Frame-Based (FB) scheduler algorithm constraints
                        prob.linear_constraints.add(
                            lin_expr=[[[v_ij_var], [1]]],
                            senses=['G'],
                            rhs=[L * s_ij_var - (L / w_ij)]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[v_ij_var], [1]]],
                            senses=['G'],
                            rhs=[(L / min(path_min_rate)) * x_ij_var - (L * r_ij / (w_ij * min(path_min_rate)))]
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
                            rhs=[x_ij_var ** 2]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[z_ij_var], [1]]],
                            senses=['G'],
                            rhs=[1 / min(path_min_rate)]
                        )
                        prob.linear_constraints.add(
                            lin_expr=[[[z_ij_var], [1]]],
                            senses=['G'],
                            rhs=[s_ij_var]
                        )
                        constraint_sum += (L / w_ij) * (x_ij_var + (w_ij - r_ij) * z_ij_var)
                        delay_slack -= (L / r_ij + (P[path[i]][path[j]] - 1) * (L / w_ij) + l_ij + n_i)
                        wcd += ((L * s_ij_var * (L / w_ij) * P[path[i]][path[j]] * x_ij_var) + v_ij_var) + l_ij + n_i
                    prob.linear_constraints.add(
                        lin_expr=[[[wcd], [1]]],
                        senses=['G'],
                        rhs=[flow[device]['deadline']]
                    )
                    prob.linear_constraints.add(
                        lin_expr=[[[constraint_sum], [1]]],
                        senses=['L'],
                        rhs=[delay_slack])
                    H = H + 1
                    flow[device]['paths'][H][1]['wcd'] = wcd  # Dictionary with 'wcd' value

        # Objective function: min ∑((i,j)∈A)〖f_ij * r_ij〗
        objective = []
        for i, j in resource_graph.edges:
            f_ij = calculate_cost(resource_graph, i, j,sum([flow[device]['reserved_rates'] for device in field_devices if any(Check_edge_inpath(path[0], i, j) for path in flow[device]['paths'])]))
            r_ij_var = 'r_{}_{}'.format(i, j)
            objective.append((f_ij, r_ij_var))

        prob.objective.set_linear(objective)
        prob.objective.set_sense(prob.objective.sense.minimize)

        # Solve the problem
        prob.solve()

        # Return the solution status and variables
        return prob.solution.get_status(), prob.solution.get_values()

    except CplexError as e:
        print(f"Cplex error: {e}")
        return None

#if __name__ == "__main__":
result = solve_optimal_path(resource_graph, paths, field_devices_delta, scheduling_algorithm, 1500, 1500)
print(result)
