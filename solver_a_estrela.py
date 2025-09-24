import time
import heapq
import json
import copy

class Node:
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()

    def __repr__(self):
        return f"Node(f={self.f()}, g={self.g}, h={self.h}, state={self.state})"

GOAL_POSITIONS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 0: (2, 2)
}

#Heurísticas

def heuristica_custo_uniforme(state, goal_state):
    return 0

def heuristica_simples_admissivel(state, goal_state):
    misplaced = 0
    for r in range(3):
        for c in range(3):
            if state[r][c] != 0 and state[r][c] != goal_state[r][c]:
                misplaced += 1
    return misplaced

def heuristica_manhattan_admissivel(state, goal_state):
    distance = 0
    for r in range(3):
        for c in range(3):
            tile = state[r][c]
            if tile != 0:
                goal_r, goal_c = GOAL_POSITIONS[tile]
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

def heuristica_nao_admissivel(state, goal_state):
    return heuristica_manhattan_admissivel(state, goal_state) * 2


def find_empty_tile(state):
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                return r, c
    return None

def get_successors(node):
    successors = []
    r, c = find_empty_tile(node.state)
    moves = {
        'CIMA': (-1, 0),
        'BAIXO': (1, 0),
        'ESQUERDA': (0, -1),
        'DIREITA': (0, 1)
    }
    for action, (dr, dc) in moves.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state = copy.deepcopy(node.state)
            new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
            successors.append((new_state, action))
    return successors

def reconstruct_path(node):
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    return path[::-1]

def visualize_path(initial_state, path):
    def apply_move(state, move):
        r, c = next((r, c) for r, row in enumerate(state) for c, val in enumerate(row) if val == 0)
        moves = {'CIMA': (-1, 0), 'BAIXO': (1, 0), 'ESQUERDA': (0, -1), 'DIREITA': (0, 1)}
        dr, dc = moves[move]
        nr, nc = r + dr, c + dc
        new_state = copy.deepcopy(state)
        new_state[r][c], new_state[nr][nc] = new_state[nr][nc], new_state[r][c]
        return new_state

    def print_board(board, move_name="Estado Inicial"):
        print(f"\n--- {move_name} ---")
        for row in board:
            print(" ".join(map(str, row)).replace('0', '_'))

    current_state = copy.deepcopy(initial_state)
    print_board(current_state)

    for i, move in enumerate(path):
        current_state = apply_move(current_state, move)
        print_board(current_state, f"Passo {i+1}: {move}")

#A* search

def a_star_search(initial_state, goal_state, heuristic_func):
    start_time = time.time()
    initial_state_tuple = tuple(map(tuple, initial_state))
    goal_state_tuple = tuple(map(tuple, goal_state))

    fronteira = []
    start_node = Node(state=initial_state, g=0, h=heuristic_func(initial_state, goal_state))
    heapq.heappush(fronteira, start_node)

    visitados = {initial_state_tuple: start_node.g}
    nodes_visited_count = 0
    max_fronteira_size = 1

    while fronteira:
        max_fronteira_size = max(max_fronteira_size, len(fronteira))
        current_node = heapq.heappop(fronteira)
        nodes_visited_count += 1
        current_state_tuple = tuple(map(tuple, current_node.state))

        if current_state_tuple == goal_state_tuple:
            end_time = time.time()
            path = reconstruct_path(current_node)
            fronteira_list = [list(map(list, node.state)) for node in fronteira]
            visitados_list = {str(k): v for k, v in visitados.items()}
            output_data = {
                'fronteira_final': fronteira_list,
                'visitados_final': visitados_list
            }
            return {
                "path": path,
                "nodes_visited": nodes_visited_count,
                "path_length": len(path),
                "execution_time": end_time - start_time,
                "max_fronteira_size": max_fronteira_size,
                "output_data": output_data
            }

        for new_state_list, action in get_successors(current_node):
            new_g = current_node.g + 1
            new_state_tuple = tuple(map(tuple, new_state_list))
            if new_state_tuple in visitados and visitados[new_state_tuple] <= new_g:
                continue
            visitados[new_state_tuple] = new_g
            new_h = heuristic_func(new_state_list, goal_state)
            successor_node = Node(state=new_state_list, parent=current_node, action=action, g=new_g, h=new_h)
            heapq.heappush(fronteira, successor_node)

    return None


if __name__ == "__main__":
    GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    test_cases = {
        "facil": [[1, 2, 3], [4, 5, 6], [0, 7, 8]],
        "medio": [[1, 0, 3], [4, 2, 5], [7, 8, 6]],
        "dificil": [[8, 7, 1], [6, 0, 2], [5, 4, 3]],
        "muito_dificil": [[6, 4, 7], [8, 5, 0], [3, 2, 1]]
    }

    heuristics = {
        "1": ("custo uniforme", heuristica_custo_uniforme),
        "2": ("a* nao admissivel", heuristica_nao_admissivel),
        "3": ("a* admissivel simples", heuristica_simples_admissivel),
        "4": ("a* admissivel precisa", heuristica_manhattan_admissivel),
    }

    print("bem-vindo ao solucionador do 8-puzzle!")
    print("\nescolha um tabuleiro inicial:")
    for i, name in enumerate(test_cases.keys()):
        print(f"{i+1}. {name}")

    board_choice = int(input("digite o numero do tabuleiro: "))
    board_name = list(test_cases.keys())[board_choice-1]
    initial_board = test_cases[board_name]

    print("\nescolha o algoritmo de busca:")
    for key, (name, _) in heuristics.items():
        print(f"{key}. {name}")
    print("5. rodar todas as heuristicas para comparacao")

    algo_choice = input("digite o numero do algoritmo: ")

    if algo_choice == "5":
        print(f"\nrodando todas as heurísticas para o tabuleiro '{board_name}'...\n")
        resultados = []
        for key, (algo_name, heuristic_function) in heuristics.items():
            print(f"\n--- resolvendo com '{algo_name}' ---")
            result = a_star_search(initial_board, GOAL, heuristic_function)
            if result:
                print(f"caminho: {' -> '.join(result['path'])}")
                print(f"nodos visitados: {result['nodes_visited']}")
                print(f"tamanho do caminho: {result['path_length']}")
                print(f"tempo: {result['execution_time']:.6f}s")
                print(f"max fronteira: {result['max_fronteira_size']}")
                resultados.append((algo_name, result))
        print("\n=== comparacao final ===")
        for algo_name, result in resultados:
            print(f"{algo_name:23} | nodos: {result['nodes_visited']:2} | caminho: {result['path_length']:2} | tempo: {result['execution_time']:.6f}s | fronteira: {result['max_fronteira_size']:2}")

    else:
        algo_name, heuristic_function = heuristics[algo_choice]
        print(f"\nresolvendo o tabuleiro '{board_name}' com '{algo_name}'...")
        result = a_star_search(initial_board, GOAL, heuristic_function)

        if result:
            print("\n--- solucao encontrada! ---")
            print(f"caminho: {' -> '.join(result['path'])}")
            print(f"nodos visitados: {result['nodes_visited']}")
            print(f"tamanho do caminho: {result['path_length']}")
            print(f"tempo de execucao: {result['execution_time']:.6f} segundos")
            print(f"maior tamanho da fronteira: {result['max_fronteira_size']}")

            filename = f"output_{board_name}_{algo_choice}.json"
            with open(filename, 'w') as f:
                json.dump(result['output_data'], f, indent=4)
            print(f"fronteira e visitados salvos em '{filename}'")

            show_viz = input("\nDeseja visualizar o caminho passo a passo? (s/n): ")
            if show_viz.lower() == 's':
                visualize_path(initial_board, result['path'])
        else:
            print("nao foi possivel encontrar uma solucao.")