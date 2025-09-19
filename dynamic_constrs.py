import sys
import os
import math
import numpy as np
import pandas as pd
import mip
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

# In the main it is possible to modify some parameters of interest
# Auxiliary functions
# For parameter initialization
def get_building_parameters(filename):
    basename = os.path.basename(filename)
    if "Edificio1" in basename:
        return {
            "base_points": [(x, y, 0) for x in range(-8, 6) for y in range(-17, -14)],
            'attack_condition': lambda p: p[1] <= -12.5,
            'battery_capacity': 3600 # 1Wh -> 3600 Joules
        }
    elif "Edificio2" in basename:
        return {
            "base_points": [(x, y, 0) for x in range(-10, 11) for y in range(-31, -29)],
            'attack_condition': lambda p: p[1] <= -20,
            'battery_capacity': 6*3600
        }
    else:
        # Default parameters for unrecognized files
        print(f"Warning: File {basename} not recognized. Using default parameters.")
        return {
            "base_points": [(x, y, 0) for x in range(-8, 6) for y in range(-17, -14)],
            'attack_condition': lambda p: p[1] <= -12.5,
            'battery_capacity': 3600 # 1Wh -> 3600 Joules
        }

# Checks connectivity between i and j
def are_connected(i, j, nodes, num_bp ,attack_indices):
    is_i_base = (i < num_bp)
    is_j_base = (j < num_bp)
    # No connection between two base points
    if is_i_base and is_j_base:
        return False
    # Connections involving a base; all and only the attack points
    if is_i_base:
        return j in attack_indices
    if is_j_base:
        return i in attack_indices
    a = nodes[i]
    b = nodes[j]
    delta = [abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])]
    dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
    if dist <= 4:
        return True
    if dist <= 11:
        count = 0
        for k in range(0, 3):
            if delta[k] <= 0.5:
                count += 1
        return count >= 2
    return False

# Builds the adjacency matrix
def build_graph(nodes, num_bp, attack_indices):
    num_nodes = len(nodes)
    graph_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Symmetry
            if are_connected(i, j, nodes, num_bp, attack_indices):
                graph_matrix[i][j] = True
                graph_matrix[j][i] = True
    return graph_matrix

def compute_time(i, j, points):
    a = points[i]
    b = points[j]
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = a[2] - b[2]  # No absolute value to distinguish ascent and descent
    horizontal_dist = math.sqrt(dx ** 2 + dy ** 2)
    horizontal_time = horizontal_dist / horizontal_speed
    if dz > 0: # Ascent
        vertical_time = dz / upward_speed
        return max(horizontal_time, vertical_time)
    elif dz < 0: # Descent
        vertical_time = -dz / downward_speed
        return max(horizontal_time, vertical_time)
    else:
        return horizontal_time

def compute_time_costs(nodes,connection_matrix):
    num_points = len(nodes)
    travel_times = np.zeros((num_points, num_points), dtype=float)
    for i in range(num_points):
        for j in range(num_points):
            if connection_matrix[i][j]:
                travel_times[i][j] = compute_time(i, j, nodes)
            else:
                travel_times[i][j] = float('inf')
    return travel_times

def compute_battery_consume(i,j,points):
    a = points[i]
    b = points[j]
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = a[2] - b[2]  # No absolute value to distinguish ascent and descent
    horizontal_dist = math.sqrt(dx ** 2 + dy ** 2)
    horizontal_energy = 10*horizontal_dist
    vertical_energy=0
    if dz > 0:  # Ascent
        vertical_energy = dz*50
    if dz < 0:  # Descent
        vertical_energy = abs(dz*5)
    return horizontal_energy + vertical_energy

def compute_energy_costs(nodes,connection_matrix):
    num_points = len(nodes)
    energy_costs = np.zeros((num_points, num_points), dtype=float)
    for i in range(num_points):
        for j in range(num_points):
            if connection_matrix[i][j]:
                energy_costs[i][j] = compute_battery_consume(i, j, nodes)
            else:
                energy_costs[i][j] = float('inf')
    return energy_costs

# Dijkstra algorithm based on energy cost
# Returns: list dist[i] = minimum energy from start_idx to i
def dijkstra_energy(start_idx, energy_costs, connection_matrix):
    n = energy_costs.shape[0]
    dist = [float('inf')] * n
    dist[start_idx] = 0.0
    pq = [(0.0, start_idx)]
    visited = [False] * n
    while pq:
        curr_e, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        for v in range(n):
            if connection_matrix[u][v]:
                new_e = curr_e + energy_costs[u][v]
                if new_e < dist[v]:
                    dist[v] = new_e
                    heapq.heappush(pq, (new_e, v))
    return dist

# Returns energy and minimum energy path from src to tgt avoiding forbidden indices
# (energy_cost, path) or (inf, []) if no path exists.
def dijkstra_energy_pair(src, tgt, energy_costs, connection_matrix, forbidden_nodes=None):
    if forbidden_nodes is None: # adaptation for heuristic
        forbidden_nodes = set()
    n = energy_costs.shape[0]
    dist = [math.inf] * n
    parent = [-1] * n
    dist[src] = 0.0
    pq = [(0.0, src)]
    visited = [False] * n
    while pq:
        curr_energy, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        if u == tgt:
            # Path reconstruction
            path = []
            node = tgt
            while node != -1:
                path.append(node)
                node = parent[node]
            path.reverse()
            return curr_energy, path
        # Explore neighbours
        for v in range(n):
            # Explore if:
            # 1. There is a connection.
            # 2. The destination node 'v' not yet finalized.
            # 3. The node 'v' is NOT in the forbidden set (unless it is the target 'tgt').
            if connection_matrix[u][v] and not visited[v] and (v not in forbidden_nodes or v == tgt):
                new_energy = curr_energy + energy_costs[u][v]
                if new_energy < dist[v]:
                    dist[v] = new_energy
                    parent[v] = u
                    heapq.heappush(pq, (new_energy, v)) # heap for efficiency
    return math.inf, []

# Quick greedy heuristic. Start from base_idx and in each trip add the node at minimum travel time until battery_capacity is reached.
# Returns: number of trips, total flight time
def fast_greedy(base_idx, energy_costs, travel_times, connection_matrix, attack_indices, battery_capacity):
    unvisited = set(attack_indices)
    trips = 0
    total_time = 0.0
    while unvisited:
        trips += 1
        remaining_energy = battery_capacity
        current = base_idx
        trip_time = 0.0
        # Time to return to base
        def return_time(from_node):
            return travel_times[from_node][base_idx]
        while True:
            best_p = None
            best_time = float('inf')
            best_energy = float('inf')
            for p in unvisited:
                if not connection_matrix[current][p]:
                    continue
                e_req = energy_costs[current][p]
                e_back = energy_costs[p][base_idx]
                if e_req + e_back > remaining_energy:
                    continue
                t_req = travel_times[current][p]
                if t_req < best_time:
                    best_p, best_time, best_energy = p, t_req, e_req
            if best_p is None:
                break
            # Visit best_p
            trip_time += best_time
            remaining_energy -= best_energy
            current = best_p
            unvisited.remove(best_p)
        # Return to base
        trip_time += return_time(current)
        total_time += trip_time
    return trips, total_time

# Selection of top K most promising base points
def select_top_k_bases(num_bp,num_nodes, energy_costs, travel_times, connection_matrix, attack_indices, battery_capacity, K=3):
    results = {}
    all_base_candidates = list(range(num_bp))
    for bp in all_base_candidates:
        dist_energy = dijkstra_energy(bp, energy_costs, connection_matrix)
        for p in range(num_bp, num_nodes):
            p_b_energy, path = dijkstra_energy_pair(p, bp, energy_costs, connection_matrix)
            if not (dist_energy[p] == float('inf') or dist_energy[p] + p_b_energy > battery_capacity):
                trips, total_t = fast_greedy(bp, energy_costs, travel_times,
                                             connection_matrix, attack_indices, battery_capacity)
                results[bp] = (trips, total_t)
            else:
                print(f"Base point {bp} removed due to capacity")
    # Sort bases by tuple (number of trips, total time)
    sorted_bases = sorted(results.items(), key=lambda item: (item[1][0], item[1][1]))
    # Extract top K
    top_k = sorted_bases[:K]
    candidates = [bp for bp, _ in top_k]
    return candidates, {bp: results[bp] for bp in candidates}

def get_subtours_excluding_full_tours(x_sol, depot, ub_travel):
    subtours = []
    for k in range(ub_travel):
        out_edges = defaultdict(list)
        visited_nodes = set()
        # Build graph for trip k
        for (i, j, kk), val in x_sol.items():
            if kk == k and val == 1:
                out_edges[i].append(j)
                visited_nodes.add(i)
                visited_nodes.add(j)
        if not visited_nodes:
            continue
        unvisited = set(visited_nodes)
        while unvisited:
            start = unvisited.pop()
            stack = [start]
            visited_comp = set([start])
            parent_map = {start: None}
            in_stack = set([start])
            found_cycles = set()
            while stack:
                node = stack[-1]
                next_neighbor = None
                # Find the first unvisited neighbor
                for neighbor in out_edges.get(node, []):
                    if neighbor not in visited_comp:
                        next_neighbor = neighbor
                        break
                if next_neighbor is not None:
                    visited_comp.add(next_neighbor)
                    in_stack.add(next_neighbor)
                    parent_map[next_neighbor] = node
                    stack.append(next_neighbor)
                else:
                    # Check back edges for cycles
                    for neighbor in out_edges.get(node, []):
                        if neighbor in in_stack:
                            cycle = []
                            temp = node
                            while temp != neighbor:
                                cycle.append(temp)
                                temp = parent_map[temp]
                            cycle.append(neighbor)
                            cycle.reverse()
                            found_cycles.add(tuple(cycle))
                    stack.pop()
                    in_stack.remove(node)
            # Add all cycles that do not contain the depot
            for cycle in found_cycles:
                if depot not in cycle:
                    subtours.append((list(cycle), k))
            unvisited -= visited_comp
    return subtours

def main():
    file_path = sys.argv[1]
    # Get building specific parameters
    building_params = get_building_parameters(file_path)
    base_points = building_params['base_points']
    attack_condition = building_params['attack_condition']
    battery_capacity = building_params['battery_capacity']
    points_df = pd.read_csv(file_path)
    to_visit = points_df[['x', 'y', 'z']].values
    nodes = np.vstack([base_points, to_visit])
    num_bp = len(base_points)
    num_nodes = len(nodes)
    attack_indices = []
    for i in range(num_bp, num_nodes):  # Start from index 1
        if attack_condition(nodes[i]):
            attack_indices.append(i)
    connection_matrix = build_graph(nodes, num_bp, attack_indices)
    travel_times = compute_time_costs(nodes, connection_matrix)
    energy_costs = compute_energy_costs(nodes, connection_matrix)
    # Prints to monitor parameter initialization:
    print("\nTotal number of nodes (base + to_visit):", num_nodes)
    print("\nCandidate base indices:", list(range(num_bp)))
    print("\n'to_visit' indices:", list(range(num_bp, num_nodes)))
    print("\nattack_indices:", attack_indices)
    if euristica_basi:
        print("\nHeuristic on.")
        candidates, stats = select_top_k_bases(
            num_bp, num_nodes, energy_costs,
            travel_times, connection_matrix, attack_indices, battery_capacity, K=basi_considerate
        )
        print("\nTop K bases:", candidates)
        # Rebuild parameters
        filtered_base_points = [base_points[bp] for bp in candidates]
        nodes = np.vstack([filtered_base_points, to_visit])
        num_nodes = len(nodes)
        num_bp = len(filtered_base_points)
        # Rebuild parameters with filtered base points
        attack_indices = []
        for i in range(num_bp, num_nodes):
            if attack_condition(nodes[i]):
                attack_indices.append(i)
        connection_matrix = build_graph(nodes, num_bp, attack_indices)
        travel_times = compute_time_costs(nodes, connection_matrix)
        energy_costs = compute_energy_costs(nodes, connection_matrix)
    max_ub = -1
    for bp in list(range(num_bp)):
        dist_energy = dijkstra_energy(bp, energy_costs, connection_matrix)
        total_round_trip = 0.0
        for p in range(num_bp, num_nodes):
            total_round_trip += 2 * dist_energy[p]
        ub_est = math.ceil(total_round_trip / battery_capacity)
        if ub_est > max_ub:
            max_ub = ub_est
    # Model
    # Variables
    m = mip.Model(solver_name=solutore)
    m.max_seconds = tempo_massimo_per_iterazione
    x = {} # x[i][j][k] == 1 <==> edge i->j travelled in trip k
    ub_travel = max_ub
    print(f"CHOSEN UB = {ub_travel}")
    edge_count = 0
    for k in range(ub_travel):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if connection_matrix[i][j]:
                    x[i,j,k] = m.add_var(var_type=mip.BINARY)
                    edge_count += 1
    active_trip = {}    # active_trip[k] == 1 <==> trip k used
    for k in range(ub_travel):
        active_trip[k] = m.add_var(var_type = mip.BINARY)
    active_bp = {}
    for bp in range(num_bp):
        active_bp[bp] = m.add_var(var_type = mip.BINARY)
    # Auxiliary variable
    z = {}
    for bp in range(num_bp):
        for k in range(ub_travel):
            z[bp, k] = m.add_var(var_type=mip.BINARY)
    # Objective function
    m.objective = mip.minimize(mip.xsum(x[i,j,k]*travel_times[i,j] for k in range(ub_travel) for i in range(num_nodes) for j in range(num_nodes) if connection_matrix[i][j]))
    # Constraints
    # 1) Only one base point
    m.add_constr(mip.xsum(active_bp[bp] for bp in range(num_bp)) == 1)
    # 2) Visit all nodes
    for i in range(num_bp, num_nodes):
        m.add_constr(
            mip.xsum(x[j, i, k] for j in range(num_nodes) if connection_matrix[j][i] for k in range(ub_travel)) >= 1)
    # 3) Flow conservation
    for i in range(num_nodes):
        for k in range(ub_travel):
            m.add_constr(mip.xsum(x[i, j, k] for j in range(num_nodes) if connection_matrix[i][j]) == mip.xsum(
                x[j, i, k] for j in range(num_nodes) if connection_matrix[j][i]))
    # 4) Energy consumed in each trip must be <= capacity
    for k in range(ub_travel):
        m.add_constr(mip.xsum(energy_costs[i, j] * x[i, j, k] for i in range(num_nodes) for j in range(num_nodes) if
                              connection_matrix[i][j]) <= battery_capacity * active_trip[k])
    # 5) Trips are consecutive
    for k in range(1, ub_travel):
        m.add_constr(active_trip[k] - active_trip[k - 1] <= 0)
    big_m = num_nodes
    # 6) linking constraint: active_trip -- visits
    for k in range(ub_travel):
        m.add_constr(active_trip[k] <= mip.xsum(
            x[i, j, k] for i in range(num_bp, num_nodes) for j in range(num_bp, num_nodes) if connection_matrix[i][j]))
        # m.add_constr(active_trip[k]<=mip.xsum(visits[i,k] for i in range(num_bp,num_nodes)))  #active_trip[k]=0 --> sum(visits[...,k]=0)
        for i in range(num_bp, num_nodes):
            m.add_constr((big_m * active_trip[k] >= mip.xsum(x[j, i, k] for j in range(num_bp, num_nodes) if
                                                             connection_matrix[i][
                                                                 j])))  # visits[...,k]-->active_trip[k]
    # 7) linking constraint: active_bp -- x (no outgoing edges if not base point)
    # Linearization constraints for z[bp,k] = active_bp[bp] * active_trip[k]
    for bp in range(num_bp):
        for k in range(ub_travel):
            m.add_constr(z[bp, k] <= active_bp[bp])
            m.add_constr(z[bp, k] <= active_trip[k])
            m.add_constr(z[bp, k] >= active_bp[bp] + active_trip[k] - 1)
    # Sum of z over all bp = active_trip[k]
    for k in range(ub_travel):
        m.add_constr(
            mip.xsum(z[bp, k] for bp in range(num_bp))
            == active_trip[k]
        )
    # Original linearized constraint
    for bp in range(num_bp):
        for k in range(ub_travel):
            m.add_constr(mip.xsum(x[bp, j, k] for j in attack_indices) == z[bp, k])

    print("── GENERAL MODEL DETAILS ──")
    print("Number of binary variables x[i,j,k]:", len(x))
    print("Number of variables z[bp,k]:        ", len(z))
    print("Number of variables active_trip:    ", len(active_trip))
    print("Number of variables active_bp:      ", len(active_bp))

    for iter_num in range(max_iters):
        print(f"\n Iteration {iter_num} - optimization...")
        status = m.optimize()
        if status == mip.OptimizationStatus.INFEASIBLE:
            print("⚠ Model infeasible:", status)
            break
        # Extract solution
        x_sol = {}
        for (i, j, k), var in x.items():
            if var.x >= 0.99:
                x_sol[i, j, k] = var.x
        # Extract depot
        for bp, var in active_bp.items():
            if var.x >= 0.99:
                depot = bp
        # Find subtours
        subtours = get_subtours_excluding_full_tours(x_sol, depot, ub_travel)
        if not subtours:
            if status == mip.OptimizationStatus.OPTIMAL:
                print("No subtours found. Optimal solution!")
                break
            else:
                total_round_trip = 0.0
                for p in range(num_bp, num_nodes):
                    total_round_trip += dijkstra_energy_pair(depot,p,energy_costs, connection_matrix)
                lb_sum = math.ceil(total_round_trip / battery_capacity)
                if lb_sum < ub_travel-1: #otherwise close to optimal
                    ub_travel-=1
                    print(f"️⚠ New UB = {ub_travel}")
        print(f"❌ Found {len(subtours)} subtours. Added elimination constraints.")
        for path, k in subtours:
            if len(path) < 2:
                continue  # ignore degenerate cycles
            m.add_constr(
                mip.xsum(x[i, j, k] for i in path for j in path if i != j and (i, j, k) in x)
                <= len(path) - 1
            )
    if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
        print("\n✅ Solution found:")
        print("\nSolution found. Variables:")
        print("\n--- Active variables x[i,j,k] (traveled edges) ---")
        x_sol = {}
        for (i, j, k), var in x.items():
            if var.x >= 0.99:
                print(f"Trip {k}: {i} -> {j}")
                x_sol[i, j, k] = var.x  # extract solution
        print("\n--- Active variables active_trip[k] (active trips) ---")
        for k, var in active_trip.items():
            print(f"Trip {k}: {'ACTIVE' if var.x >= 0.99 else 'inactive'}")
        print("\n--- Active variables active_bp[bp] ---(the only active one is the chosen base point)")
        for bp, var in active_bp.items():
            print(f"Base point {bp}: {'ACTIVE' if var.x >= 0.99 else 'inactive'}")
            if var.x >= 0.99:
                base_index = bp
        print("\n - 🛰  - SOLUTION - 🛰  -")
        # Determine index of active base point
        if euristica_basi:
            print(
                f"Chosen base point (original indices): {candidates[base_index]}")  # report base index in original numbering (pre-filtering)
        else:
            print(f"Chosen base point (original indices): {base_index}")
        trip_paths = defaultdict(list)
        base_index = next(bp for bp in range(num_bp) if active_bp[bp].x >= 0.99)  # use filtered base index for construction and visualization
        # Reconstruct paths
        for k in range(ub_travel):
            if active_trip[k].x >= 0.99:
                current_path = [base_index]
                visited = set()
                current = base_index
                while True:
                    found_next = False
                    for j in range(num_nodes):
                        if connection_matrix[current][j] and x.get((current, j, k), None) and x[current, j, k].x >= 0.99:
                            if (current, j) not in visited:
                                visited.add((current, j))
                                current_path.append(j)
                                current = j
                                found_next = True
                                break
                    if not found_next or current == base_index:
                        break
                if current != base_index:
                    current_path.append(base_index)
                trip_paths[k] = current_path
        # Textual output as specified
        for k, path in trip_paths.items():
            print(f"Trip {k+1}:", "-".join(map(str, path)))
        # ❷: Extract any remaining subtours
        subtours = get_subtours_excluding_full_tours(x_sol, base_index, ub_travel)
        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Draw all "to_visit" nodes in light gray
        xs_all = [nodes[i][0] for i in range(num_bp, num_nodes)]
        ys_all = [nodes[i][1] for i in range(num_bp, num_nodes)]
        zs_all = [nodes[i][2] for i in range(num_bp, num_nodes)]
        ax.scatter(xs_all, ys_all, zs_all, color='lightgray', s=10, label='Points to cover')
        # Draw attack points in red
        xs_att = [nodes[i][0] for i in attack_indices]
        ys_att = [nodes[i][1] for i in attack_indices]
        zs_att = [nodes[i][2] for i in attack_indices]
        ax.scatter(xs_att, ys_att, zs_att, color='red', s=20, label='Attack points')
        # Draw base in blue with marker '^'
        xb, yb, zb = nodes[base_index]
        ax.scatter([xb], [yb], [zb], color='blue', s=80, marker='^', label='Base')
        # Colormap for "regular" trips
        n_trips = len(trip_paths)
        cmap = plt.get_cmap('tab10', max(n_trips, 1))  # careful if n_trips=0
        # Plot valid paths
        for k, path in trip_paths.items():
            color_k = cmap(k % 10)  # tab10 has 10 colors; recycles if >10
            # Edges are consecutive pairs (i,j)
            edges_k = list(zip(path[:-1], path[1:]))
            for (i, j) in edges_k:
                xi, yi, zi = nodes[i]
                xj, yj, zj = nodes[j]
                ax.plot([xi, xj], [yi, yj], [zi, zj], color=color_k, linewidth=2)
        # Plot remaining subtours (dashed line, dark gray)
        for cycle_path, k in subtours:
            cycle_edges = []
            for idx in range(len(cycle_path) - 1):
                cycle_edges.append((cycle_path[idx], cycle_path[idx + 1]))
            # close cycle adding (last -> first)
            cycle_edges.append((cycle_path[-1], cycle_path[0]))
            for (i, j) in cycle_edges:
                xi, yi, zi = nodes[i]
                xj, yj, zj = nodes[j]
                ax.plot([xi, xj], [yi, yj], [zi, zj],
                        color='dimgray', linestyle='dashed', linewidth=1)
        # Legend
        from matplotlib.lines import Line2D
        legend_items = [
            Line2D([0], [0], color='lightgray', marker='o', linestyle='None', markersize=6, label='Points to cover'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=6, label='Attack points'),
            Line2D([0], [0], color='blue', marker='^', linestyle='None', markersize=10, label='Base')
        ]
        # Add an entry for each "regular" trip
        for k in sorted(trip_paths.keys()):
            legend_items.append(Line2D([0], [0], color=cmap(k % 10), lw=2, label=f'Trip {k + 1}'))
        # Add an entry for the subtours
        legend_items.append(Line2D([0], [0], color='dimgray', lw=1, linestyle='dashed', label='Subtour'))
        ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.set_title("Drone paths and possible subtours")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.tight_layout()
    plt.savefig("drone_paths.png")  # Save the plot

if __name__ == "__main__":
    horizontal_speed = 1.5
    upward_speed = 1
    downward_speed = 2
    solver = 'gurobi' # CBC for default open-source solver
    heuristic_bases = True
    selected_bases = 3
    max_time_per_iteration = 120
    max_iters =  150
    main()
