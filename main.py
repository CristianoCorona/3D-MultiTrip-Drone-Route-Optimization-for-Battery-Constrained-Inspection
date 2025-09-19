import sys
import os
import math
import numpy as np
import pandas as pd
import mip
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

#Abbiamo riflettuto a lungo sulla selezione del punto base ottimo nel tentativo di
#scorporare questo problema dalla struttura dell'edificio; tutto sommato sembra sensato
#assumere che esista un modo ottimo di esplorare i punti to_visit a prescindere dalla posizione
#dei punti base dal momento che i punti d'attacco sono l'unico punto d'accesso e possono
#essere immaginati come un'unica grande base "virtuale".
#Tuttavia non abbiamo trovato un modello soddisfacente per questo approccio soprattutto a causa
#della capacità;
#senza conoscere il punto base, infatti, non si riesce a vincolare in maniera esatta il drone
# a tornare ai punti d'attacco (base virtuale) con una capacità precisa poiché essa dipende dal
#punto base.

#Per questo motivo abbiamo deciso di adottare un approccio euristico, scegliendo le basi più
#promettenti e includendo solo quelle nel modello, riducendo così significativamente il numero
#di variabili bp e stimando un upper-bound per il numero di viaggi. Abbiamo osservato che
#sul singolo arco, un minor consumo energetico è associato a un minor dispendio temporale; abbiamo
#quindi deciso di seguire questa strada per identificare una soluzione iniziale feasible da
#fornire al solutore per migliorare l'efficacia del Branch&Bound.

#è possibile modificare il numero di basi da considerare nel modello nel main (in fondo),
#insieme al solutore, il tempo massimo di ottimizzazione e un interruttore per attivare/disattivare l'euristica.

#All'interno del .zip è presente anche un altro approccio che non presenta vincoli espliciti
#per escludere i subtour, ma li individua ed esclude tramite vincoli dinamici (dynamic_constrs.py)

#Abbiamo allegato anche alcuni edifici costruiti per testing; sono basati sui parametri di
#Edificio1


# Funzioni ausiliarie
#Per l'inizializzazione dei parametri
def get_building_parameters(filename):
    basename = os.path.basename(filename)

    if "Edificio1" in basename:
        return {
            "base_points": [(x, y, 0) for x in range(-8, 6) for y in range(-17, -14)],
            'attack_condition': lambda p: p[1] <= -12.5,
            'battery_capacity': 3600 #1Wh->3600 Joule
        }
    elif "Edificio2" in basename:
        return {
            "base_points": [(x, y, 0) for x in range(-10, 11) for y in range(-31, -29)],
            'attack_condition': lambda p: p[1] <= -20,
            'battery_capacity': 6*3600
        }
    else:
        # Parametri di default per file non riconosciuti
        print(f"Warning: File {basename} non riconosciuto. Usando parametri di default.")
        return {
            "base_points": [(x, y, 0) for x in range(-8, 6) for y in range(-17, -14)],
            'attack_condition': lambda p: p[1] <= -12.5,
            'battery_capacity': 3600 #1Wh->3600 Joule
        }

#verifica la connettività di i e j
def are_connected(i, j, nodes, num_bp ,attack_indices):
    is_i_base = (i < num_bp)
    is_j_base = (j < num_bp)

    # Nessuna connessione tra due punti base
    if is_i_base and is_j_base:
        return False

    # Connessioni che coinvolgono una base; tutti e soli i punti d'attacco
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

#costruisce la matrice d'adiacenza
def build_graph(nodes, num_bp, attack_indices):
    num_nodes = len(nodes)

    graph_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Simmetria
            if are_connected(i, j, nodes, num_bp, attack_indices):
                graph_matrix[i][j] = True
                graph_matrix[j][i] = True
    return graph_matrix

def compute_time(i, j, points):
    a = points[i]
    b = points[j]

    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    dz = a[2] - b[2]  # No valore assoluto per distinguere tra salita e discesa

    horizontal_dist = math.sqrt(dx ** 2 + dy ** 2)
    horizontal_time = horizontal_dist / horizontal_speed

    if dz > 0: # Salita
        vertical_time = dz / upward_speed
        return max(horizontal_time, vertical_time)
    elif dz < 0: # Discesa
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
    dz = a[2] - b[2]  # No valore assoluto per distinguere tra salita e discesa

    horizontal_dist = math.sqrt(dx ** 2 + dy ** 2)
    horizontal_energy = 10*horizontal_dist

    vertical_energy=0

    if dz > 0:  # Salita
        vertical_energy = dz*50
    if dz < 0:  # Discesa
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

#Dijkstra basato sul costo energetico
#ret: lista dist[i] = energia minima per start_idx -> i
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

#Restituisce l'energia e il cammino di costo minimo energetico da src a tgt, evitando gli indici proibiti
#(costo_energia, percorso) o (inf, []) se non esiste.
def dijkstra_energy_pair(src, tgt, energy_costs, connection_matrix, forbidden_nodes=None):
    if forbidden_nodes is None: # adattamento per l'euristica
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
            # Ricostruzione percorso
            path = []
            node = tgt
            while node != -1:
                path.append(node)
                node = parent[node]
            path.reverse()
            return curr_energy, path

        # Esploriamo i vicini
        for v in range(n):
            # Esploriamo se:
            # 1. Esiste una connessione.
            # 2. Il nodo di destinazione 'v' non è stato ancora finalizzato.
            # 3. Il nodo 'v' NON è nel set proibito (a meno che non sia proprio 'tgt').
            if connection_matrix[u][v] and not visited[v] and (v not in forbidden_nodes or v == tgt):
                new_energy = curr_energy + energy_costs[u][v]
                if new_energy < dist[v]:
                    dist[v] = new_energy
                    parent[v] = u
                    heapq.heappush(pq, (new_energy, v)) #heap per efficienza

    return math.inf, []

#Euristica greedy: permette di calcolare una soluzione iniziale feasible e di dare un ub sensato al numero di viaggi,
# per aiutare il più possibile il solutore.
#ret: trips_needed, heuristic_paths (dizionario {k:path})
def generate_greedy_start_and_ub(nodes, num_bp, battery_capacity, energy_costs, connection_matrix): #assumendo non si passi due volte per lo stesso nodo in un viaggio

    if num_bp == 0:
        return 0, {}

    all_to_visit = set(range(num_bp, len(nodes)))

    for base_idx in range(num_bp): #proviamo con diversi punti base qualora non fosse possibile per un dato bp
        unvisited = set(all_to_visit)
        trips_needed = 0
        heuristic_paths = {}

        while unvisited:
            trips_needed += 1
            trip_idx = trips_needed - 1

            current_energy = battery_capacity
            current_pos = base_idx
            current_path = [base_idx]

            while True:
                # Assumiamo di non rivisitare un nodo all'interno dello stesso viaggio
                nodes_in_this_trip = set(current_path)

                best_next = None
                best_cost_to_next = math.inf
                best_path_to_next = []

                # Cerchiamo il prossimo nodo 'p' da aggiungere al viaggio corrente
                for p in unvisited:
                    # Cerchiamo un percorso da 'current_pos' a 'p' che non usi nodi già visitati in questo viaggio
                    cost_to_p, path_to_p = dijkstra_energy_pair(
                        current_pos, p, energy_costs, connection_matrix,
                        forbidden_nodes=nodes_in_this_trip
                    )
                    if not path_to_p:  # Percorso non trovato, prossimo bp
                        continue

                    # Calcoliamo il ritorno da 'p' alla base, evitando tutti i nodi del percorso di andata esteso
                    prospective_path_nodes = nodes_in_this_trip.union(path_to_p)
                    cost_back, _ = dijkstra_energy_pair(
                        p, base_idx, energy_costs, connection_matrix,
                        forbidden_nodes=prospective_path_nodes
                    )
                    if cost_back == math.inf: #proviamo col prossimo bp (non siamo riusciti a tornare)
                        continue

                    # Verifichiamo che l'energia totale (segmento + ritorno) sia sufficiente
                    if cost_to_p + cost_back <= current_energy:
                        if cost_to_p < best_cost_to_next:
                            best_cost_to_next = cost_to_p
                            best_next = p
                            best_path_to_next = path_to_p

                if best_next is None:
                    # Non ci sono più nodi che possiamo aggiungere a questo viaggio.
                    # Chiudiamo il percorso e terminiamo il ciclo interno.
                    break

                # Aggiungiamo il segmento migliore trovato al percorso corrente
                # path_to_next[0] è 'current_pos', quindi lo escludiamo
                for node in best_path_to_next[1:]:
                    current_path.append(node)

                current_energy -= best_cost_to_next
                current_pos = best_next

                # Rimuoviamo tutti i nodi visitati nel nuovo segmento dalla lista globale `unvisited`
                for node in best_path_to_next[1:]:
                    if node in unvisited:
                        unvisited.remove(node)

            # Chiusura del viaggio: torna alla base dall'ultima posizione raggiunta
            nodes_in_this_trip = set(current_path)
            cost_back_to_base, path_back_to_base = dijkstra_energy_pair(
                current_pos, base_idx, energy_costs, connection_matrix,
                forbidden_nodes=nodes_in_this_trip
            )

            # Aggiungiamo il percorso di ritorno per completare il ciclo
            for node in path_back_to_base[1:]:
                current_path.append(node)
            # Salviamo il percorso se ha visitato almeno un punto
            if len(current_path) > 1:
                heuristic_paths[trip_idx] = current_path
            else:
                # Il viaggio non ha visitato nulla, non va conteggiato
                trips_needed -= 1

        # Se arriviamo qui, abbiamo servito tutti i punti con questa base
        print(f"Base {base_idx} OK con {trips_needed} viaggi.")
        return trips_needed, heuristic_paths

    # Se nessun base point ha funzionato
    raise RuntimeError("Nessun base point può costruire una soluzione iniziale valida.")

#Rapida euristica greedy. Partiamo da base_idx e a ogni viaggio aggiungiamo il nodo a distanza temporale minima fino a battery_capacity.
#ret: numero di viaggi, tempo di volo
def fast_greedy(base_idx, energy_costs, travel_times, connection_matrix, attack_indices, battery_capacity):
    unvisited = set(attack_indices)

    trips = 0
    total_time = 0.0

    while unvisited:
        trips += 1
        remaining_energy = battery_capacity
        current = base_idx
        trip_time = 0.0

        # Tempo per tornare a base
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

            # Visito best_p
            trip_time += best_time
            remaining_energy -= best_energy
            current = best_p
            unvisited.remove(best_p)

        # Ritorno a base
        trip_time += return_time(current)
        total_time += trip_time

    return trips, total_time


# Selezione dei primi K bp più promettenti
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
                print(f"Punto base {bp} rimosso per capacità")


    # Ordiniamo le basi per tuple (numero di viaggi, tempo totale)
    sorted_bases = sorted(results.items(), key=lambda item: (item[1][0], item[1][1]))

    # Estraiamo i primi K
    top_k = sorted_bases[:K]
    candidates = [bp for bp, _ in top_k]
    return candidates, {bp: results[bp] for bp in candidates}

def main():
    file_path = sys.argv[1]

    # Ottieni parametri specifici dell'edificio
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
    for i in range(num_bp,num_nodes):  # Inizia da indice 1
        if attack_condition(nodes[i]):
            attack_indices.append(i)

    connection_matrix = build_graph(nodes,num_bp,attack_indices)

    travel_times = compute_time_costs(nodes, connection_matrix)

    energy_costs = compute_energy_costs(nodes, connection_matrix)

    #Stampe per monitorare l'inizializzazione dei parametri:
    print("\nNumero totale di nodi (base + da_visitare):", num_nodes)
    print("\nIndici candidati base:", list(range(num_bp)))
    print("\nIndici 'to_visit':", list(range(num_bp, num_nodes)))
    print("\nIndici attack_indices:", attack_indices)

    if euristica_basi:
        print("\nEuristica on.")
        candidates, stats = select_top_k_bases(
            num_bp, num_nodes, energy_costs,
            travel_times, connection_matrix,attack_indices, battery_capacity, K=basi_considerate
        )
        print("\nTop K basi:", candidates)
        #Ricostruzione parametri
        filtered_base_points = [base_points[bp] for bp in candidates]
        nodes = np.vstack([filtered_base_points, to_visit])
        num_nodes = len(nodes)
        num_bp = len(filtered_base_points)
        # Ricostruiamo il parametri con i punti base filtrati
        attack_indices = []
        for i in range(num_bp, num_nodes):
            if attack_condition(nodes[i]):
                attack_indices.append(i)

        connection_matrix = build_graph(nodes, num_bp, attack_indices)

        travel_times = compute_time_costs(nodes, connection_matrix)

        energy_costs = compute_energy_costs(nodes, connection_matrix)

    # Eseguiamo l'euristica per avere una soluzione iniziale feasible e un ub sul numero di viaggi legato a una soluzione promettente. Per un
    #approccio più conservativo si potrebbe usare il seguente approccio:
    max_ub = -1
    for bp in list(range(num_bp)):
        dist_energy = dijkstra_energy(bp, energy_costs, connection_matrix)
        total_round_trip = 0.0
        for p in range(num_bp, num_nodes):
            total_round_trip += 2 * dist_energy[p]
        ub_est = math.ceil(total_round_trip / battery_capacity)
        if ub_est > max_ub:
            max_ub = ub_est
    # e utilizzare max_ub come upper bound; questo però fa esplodere le dimensioni del modello per istanze grandi
    ub_travel, heuristic_paths = generate_greedy_start_and_ub(nodes, num_bp, battery_capacity, energy_costs,
                                                              connection_matrix)

    #Modello
    #Variabili
    m = mip.Model(solver_name=solutore)
    m.max_seconds = tempo_massimo

    ub_travel += 2 #per conservatività su edifici piccoli
    x = {} #x[i][j][k] == 1 <==> arco i->j percorso nel viaggio k
    print(f"\nUB SCELTO = {ub_travel}")
    edge_count = 0
    for k in range(ub_travel):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if connection_matrix[i][j]:
                    x[i, j, k] = m.add_var(var_type=mip.BINARY)
                    edge_count += 1

    active_trip = {}  # active_trip[k] == 1 <==> il viaggio k è usato
    for k in range(ub_travel):
        active_trip[k] = m.add_var(var_type=mip.BINARY)

    active_bp = {}
    for bp in range(num_bp):
        active_bp[bp] = m.add_var(var_type=mip.BINARY)

    # Variabile ausiliaria
    z = {}
    for bp in range(num_bp):
        for k in range(ub_travel):
            z[bp, k] = m.add_var(var_type=mip.BINARY)

    # Variabili continue per la formulazione MTZ
    u = {}
    for i in range(num_nodes):
        for k in range(ub_travel):
            # u[i,k] può essere 0 se il nodo i non è visitato nel viaggio k
            u[i, k] = m.add_var(var_type=mip.CONTINUOUS, lb=0)

    # Funzione obiettivo
    m.objective = mip.minimize(mip.xsum(
        x[i, j, k] * travel_times[i, j] for k in range(ub_travel) for i in range(num_nodes) for j in range(num_nodes) if
        connection_matrix[i][j]))

    # Vincoli
    # 1) Un solo punto base
    m.add_constr(mip.xsum(active_bp[bp] for bp in range(num_bp)) == 1)

    # 2) Visitare tutti i nodi
    for i in range(num_bp, num_nodes):
        m.add_constr(
            mip.xsum(x[j, i, k] for j in range(num_nodes) if connection_matrix[j][i] for k in range(ub_travel)) >= 1)

    # 3) Conservazione del flusso
    for i in range(num_nodes):
        for k in range(ub_travel):
            m.add_constr(mip.xsum(x[i, j, k] for j in range(num_nodes) if connection_matrix[i][j]) == mip.xsum(
                x[j, i, k] for j in range(num_nodes) if connection_matrix[j][i]))

    # 5) L'energia consumata in ogni viaggio deve essere <= della capacità
    for k in range(ub_travel):
        m.add_constr(mip.xsum(energy_costs[i, j] * x[i, j, k] for i in range(num_nodes) for j in range(num_nodes) if
                              connection_matrix[i][j]) <= battery_capacity * active_trip[k])

    # 6) i viaggi sono consecutivi
    for k in range(1, ub_travel):
        m.add_constr(active_trip[k] - active_trip[k - 1] <= 0)

    big_m = num_nodes
    # 7) linking constraint: active_trip -- visits
    for k in range(ub_travel):
        m.add_constr(active_trip[k] <= mip.xsum(
            x[i, j, k] for i in range(num_bp, num_nodes) for j in range(num_bp, num_nodes) if connection_matrix[i][j]))
        # m.add_constr(active_trip[k]<=mip.xsum(visits[i,k] for i in range(num_bp,num_nodes)))  #active_trip[k]=0 --> sum(visits[...,k]=0)
        for i in range(num_bp, num_nodes):
            m.add_constr((big_m * active_trip[k] >= mip.xsum(x[j, i, k] for j in range(num_bp, num_nodes) if
                                                             connection_matrix[i][
                                                                 j])))  # visits[...,k]-->active_trip[k]

    # 8) linking constraint: active_bp -- x (non ho archi uscenti se non è il punto base)
    # Vincoli di linearizzazione per z[bp,k] = active_bp[bp] * active_trip[k]
    for bp in range(num_bp):
        for k in range(ub_travel):
            m.add_constr(z[bp, k] <= active_bp[bp])
            m.add_constr(z[bp, k] <= active_trip[k])
            m.add_constr(z[bp, k] >= active_bp[bp] + active_trip[k] - 1)

    # Somma di z su tutti i bp = active_trip[k]
    for k in range(ub_travel):
        m.add_constr(
            mip.xsum(z[bp, k] for bp in range(num_bp))
            == active_trip[k]
        )

    # Vincolo originale linearizzato
    for bp in range(num_bp):
        for k in range(ub_travel):
            m.add_constr(mip.xsum(x[bp, j, k] for j in attack_indices) == z[bp, k])

    N_nodes = num_nodes  # big M

    for k in range(ub_travel):
        # Il deposito (base point) ha sempre posizione 1
        for bp in range(num_bp):
            # Il vincolo agisce solo se il bp è quello attivo per il viaggio k
            # Usiamo la variabile z[bp, k] che è 1 solo se bp e k sono attivi
            m.add_constr(u[bp, k] <= 1 + (N_nodes * (1 - z[bp, k])))
            m.add_constr(u[bp, k] >= 1 - (N_nodes * (1 - z[bp, k])))

        for i in range(num_bp, num_nodes):  # Per tutti i nodi che non sono base
            # La posizione di un nodo visitato deve essere almeno 2
            m.add_constr(u[i, k] >= 2 * mip.xsum(x[j, i, k] for j in range(num_nodes) if connection_matrix[j, i]))
            m.add_constr(u[i, k] <= N_nodes)

        # Il vincolo principale MTZ
        for i in range(num_bp, num_nodes):
            for j in range(num_bp, num_nodes):
                if i != j and connection_matrix[i, j]:
                    m.add_constr(
                        u[i, k] - u[j, k] + N_nodes * x[i, j, k] <= N_nodes - 1
                    )

    print("\n--- DETTAGLIO DEL MODELLO ---")
    print("Numero di variabili binarie x[i,j,k]:", len(x))
    print("Numero di variabili z[bp,k]:        ", len(z))
    print("Numero di variabili active_trip:    ", len(active_trip))
    print("Numero di variabili active_bp:      ", len(active_bp))

    # Preparazione e passaggio della soluzione iniziale per il warm start
    start_solution = []
    active_bp_set = False

    for k, path in heuristic_paths.items():
        # Attiviamo il trip k
        start_solution.append((active_trip[k], 1.0))

        # Attiviamo il punto base
        base_node = path[0]
        if not active_bp_set:
            start_solution.append((active_bp[base_node], 1.0))
            active_bp_set = True

        # Attiviamo la variabile z[bp, k]
        start_solution.append((z[base_node, k], 1.0))

        # Impostiamo le variabili x[i,j,k] per gli archi del percorso
        for i in range(len(path) - 1):
            u_node, v_node = path[i], path[i + 1]
            if (u_node, v_node, k) in x:
                start_solution.append((x[u_node, v_node, k], 1.0))

        # Assegniamo le posizioni in modo strettamente crescente lungo il percorso
        positions = {}
        for idx, node in enumerate(path[:-1]):  # Escludiamo il ritorno al bp
            if node not in positions:
                positions[node] = float(idx + 1)

        # Aggiungiamo le posizioni calcolate alla soluzione iniziale
        for node, pos in positions.items():
            start_solution.append((u[node, k], pos))

        # Forziamo le u associate a nodi esterni al path a essere 0
        nodes_in_path = set(path)
        for i in range(num_nodes):
            if i not in nodes_in_path:
                start_solution.append((u[i, k], 0.0))

    # Rimuove eventuali duplicati tenendo traccia solo dell'ultimo valore (per robustezza)
    seen_vars = set()
    final_solution = []
    for var, val in reversed(start_solution):
        if var not in seen_vars:
            final_solution.append((var, val))
            seen_vars.add(var)

    start_solution = list(reversed(final_solution))

    m.start = start_solution

    status = m.optimize()

    if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
        print("\nSoluzione trovata. Variabili:")
        print("\n--- Variabili x[i,j,k] attive (archi percorsi) ---")
        x_sol = {}
        for (i, j, k), var in x.items():
            if var.x >= 0.99:
                print(f"Trip {k}: {i} -> {j}")
                x_sol[i, j, k] = var.x  # estrazione della soluzione

        print("\n--- Variabili active_trip[k] (viaggi attivi) ---")
        for k, var in active_trip.items():
            print(f"Trip {k}: {'ATTIVO' if var.x >= 0.99 else 'inattivo'}")

        print("\n--- Variabili active_bp[bp] ---(l'unico attivo è il punto base scelto)")
        for bp, var in active_bp.items():
            print(f"Punto base {bp}: {'ATTIVO' if var.x >= 0.99 else 'inattivo'}")
            if var.x >= 0.99:
                base_index = bp
        print("\n - 🛰  - SOLUZIONE - 🛰  -")
        # Determiniamo l'indice del punto base attivo
        if euristica_basi:
            print(f"Punto base scelto (indici iniziali): {candidates[base_index]}") #riportiamo l'indice della base scelta nella numerazione originale (pre-filtraggio)
        else:
            print(f"Punto base scelto (indici iniziali): {base_index}")
        trip_paths = defaultdict(list)
        base_index = next(bp for bp in range(num_bp) if active_bp[bp].x >= 0.99) #per la costruzione e la visualizzazione usiamo l'indice nelle basi filtrate

        #Ricostruzione dei cammini

        for k in range(ub_travel):
            if active_trip[k].x < 0.5:
                continue
            # Ricostruzione partendo dal base
            current = base_index
            path = [current]
            visited_arcs = set()
            while True:
                trovato = False
                for j in range(num_nodes):
                    if connection_matrix[current][j] and (current, j, k) in x and x[current, j, k].x >= 0.99:
                        if (current, j) not in visited_arcs:
                            visited_arcs.add((current, j))
                            path.append(j)
                            current = j
                            trovato = True
                            break
                if not trovato:
                    break
            # Se non sono ritornato al base, lo aggiungo alla fine
            if path[-1] != base_index:
                path.append(base_index)
            trip_paths[k] = path

        # Output testuale come richiesto
        for k, path in trip_paths.items():
            print(f"Viaggio {k + 1}:", "-".join(map(str, path)))

        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Disegniamo tutti i nodi "to_visit" in grigio chiaro
        xs_all = [nodes[i][0] for i in range(num_bp, num_nodes)]
        ys_all = [nodes[i][1] for i in range(num_bp, num_nodes)]
        zs_all = [nodes[i][2] for i in range(num_bp, num_nodes)]
        ax.scatter(xs_all, ys_all, zs_all, color='lightgray', s=10, label='Punti da coprire')

        # Disegniamo i punti di attacco in rosso
        xs_att = [nodes[i][0] for i in attack_indices]
        ys_att = [nodes[i][1] for i in attack_indices]
        zs_att = [nodes[i][2] for i in attack_indices]
        ax.scatter(xs_att, ys_att, zs_att, color='red', s=20, label='Punti di attacco')

        # Disegniamo il base in blu con marker '^'
        xb, yb, zb = nodes[base_index]
        ax.scatter([xb], [yb], [zb], color='blue', s=80, marker='^', label='Base')

        # Colormap per distinguere i viaggi
        n_trips = len(trip_paths)
        cmap = plt.get_cmap('tab10', max(n_trips, 1))  # attenzione se n_trips=0

        # Tracciamo i percorsi
        for k, path in trip_paths.items():
            color_k = cmap(k % 10)  # tab10 ha 10 colori; in caso di >10, ricicla
            # Gli archi sono coppie consecutive (i,j)
            edges_k = list(zip(path[:-1], path[1:]))

            for (i, j) in edges_k:
                xi, yi, zi = nodes[i]
                xj, yj, zj = nodes[j]
                ax.plot([xi, xj], [yi, yj], [zi, zj], color=color_k, linewidth=2)

        # Legenda

        from matplotlib.lines import Line2D

        legend_items = [
            Line2D([0], [0], color='lightgray', marker='o', linestyle='None', markersize=6, label='Punti da coprire'),
            Line2D([0], [0], color='red', marker='o', linestyle='None', markersize=6, label='Punti di attacco'),
            Line2D([0], [0], color='blue', marker='^', linestyle='None', markersize=10, label='Base')
        ]

        # Aggiungo una voce per ogni viaggio “regolare”
        for k in sorted(trip_paths.keys()):
            legend_items.append(Line2D([0], [0], color=cmap(k % 10), lw=2, label=f'Viaggio {k + 1}'))

        ax.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.05, 1))
        ax.set_title("Tragitti dei droni e eventuali subtour")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.tight_layout()
    plt.savefig("drone_paths.png")  # Salviamo il grafico

if __name__ == "__main__":
    horizontal_speed = 1.5
    upward_speed = 1
    downward_speed = 2

    solutore = 'gurobi' #CBC per solutore standard (opensource)
    euristica_basi  = True
    basi_considerate = 3
    tempo_massimo = 10800 #3 ore, [secondi]

    main()
