import time
import heapq
import json
import copy

class Node:
    """
    Representa um nó na árvore de busca. Contém o estado do tabuleiro,
    o nó pai (para reconstruir o caminho), a ação que levou a este estado,
    e os custos g (custo do caminho) e h (heurística).
    """
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # Custo do início até o nó atual
        self.h = h  # Custo estimado do nó atual até o objetivo

    def f(self):
        """ Retorna o custo total f(n) = g(n) + h(n). """
        return self.g + self.h

    def __lt__(self, other):
        """ Comparador para a fila de prioridade (heapq). """
        return self.f() < other.f()

    def __repr__(self):
        """ Representação em string do nó (para debugging). """
        return f"Node(f={self.f()}, g={self.g}, h={self.h}, state={self.state})"


# Posições do objetivo para cálculo rápido da Distância de Manhattan
GOAL_POSITIONS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 0: (2, 2)  # 0 representa o espaço vazio
}

def heuristica_custo_uniforme(state, goal_state):
    """ Heurística para a Busca de Custo Uniforme (sempre retorna 0). """
    return 0

def heuristica_simples_admissivel(state, goal_state):
    """
    Heurística admissível simples: Contagem de peças fora do lugar (Hamming distance).
    É admissível porque cada movimento pode, no máximo, corrigir a posição de uma peça.
    """
    misplaced = 0
    for r in range(3):
        for c in range(3):
            if state[r][c] != 0 and state[r][c] != goal_state[r][c]:
                misplaced += 1
    return misplaced

def heuristica_manhattan_admissivel(state, goal_state):
    """
    Heurística admissível mais precisa: Distância de Manhattan.
    Soma das distâncias vertical e horizontal de cada peça à sua posição objetivo.
    É admissível porque cada movimento reduz a distância de Manhattan total em exatamente 1.
    """
    distance = 0
    for r in range(3):
        for c in range(3):
            tile = state[r][c]
            if tile != 0:
                goal_r, goal_c = GOAL_POSITIONS[tile]
                distance += abs(r - goal_r) + abs(c - goal_c)
    return distance

def heuristica_nao_admissivel(state, goal_state):
    """
    Heurística não admissível. Usa a distância de Manhattan, mas a superestima.
    Isso pode levar a uma solução mais rápida, mas não garante o caminho mais curto.
    """
    return heuristica_manhattan_admissivel(state, goal_state) * 2

# --- LÓGICA DO ALGORITMO A* ---

def find_empty_tile(state):
    """ Encontra as coordenadas (linha, coluna) do espaço vazio (0). """
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                return r, c
    return None

def get_successors(node):
    """ Gera todos os estados sucessores válidos a partir do estado atual. """
    successors = []
    r, c = find_empty_tile(node.state)
    
    # Movimentos possíveis: CIMA, BAIXO, ESQUERDA, DIREITA
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
    """ Reconstrói o caminho da solução a partir do nó objetivo. """
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    return path[::-1] # Retorna o caminho na ordem correta

def a_star_search(initial_state, goal_state, heuristic_func):
    """
    implementa o algoritimo a*.
    
    Args:
        initial_state: O estado inicial do tabuleiro.
        goal_state: O estado objetivo.
        heuristic_func: A função heurística a ser utilizada.
    """
    start_time = time.time()
    
    # Converte estados para tuplas para serem usados como chaves de dicionário/set
    initial_state_tuple = tuple(map(tuple, initial_state))
    goal_state_tuple = tuple(map(tuple, goal_state))

    # Fronteira (lista de abertos) é uma fila de prioridade
    fronteira = []
    start_node = Node(state=initial_state, g=0, h=heuristic_func(initial_state, goal_state))
    heapq.heappush(fronteira, start_node)

    # Dicionário de visitados para armazenar o menor custo 'g' para cada estado
    visitados = {initial_state_tuple: start_node.g}

    # metricas de desempenho
    nodes_visited_count = 0
    max_fronteira_size = 1

    while fronteira:
        max_fronteira_size = max(max_fronteira_size, len(fronteira))
        
        # pega o oó com o menor f(n) da fronteira
        current_node = heapq.heappop(fronteira)
        nodes_visited_count += 1

        current_state_tuple = tuple(map(tuple, current_node.state))

        # verifica se alcançou o objetivo
        if current_state_tuple == goal_state_tuple:
            end_time = time.time()
            path = reconstruct_path(current_node)
            
            # prepara dados para o arquivo de saída
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

        # gera sucessores
        for new_state_list, action in get_successors(current_node):
            new_g = current_node.g + 1  # Custo de cada movimento é 1
            new_state_tuple = tuple(map(tuple, new_state_list))

            if new_state_tuple in visitados and visitados[new_state_tuple] <= new_g:
                continue
            
            # Adiciona ou atualiza o no na lista de visitados e na fronteira
            visitados[new_state_tuple] = new_g
            new_h = heuristic_func(new_state_list, goal_state)
            successor_node = Node(state=new_state_list, parent=current_node, action=action, g=new_g, h=new_h)
            heapq.heappush(fronteira, successor_node)
            
    return None 

if __name__ == "__main__":
    GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    # Casos de teste
    test_cases = {
        "facil": [[1, 2, 3], [4, 5, 6], [0, 7, 8]],
        "medio": [[1, 0, 3], [4, 2, 5], [7, 8, 6]],
        "dificil": [[8, 7, 1], [6, 0, 2], [5, 4, 3]],
        "muito_dificil": [[6, 4, 7], [8, 5, 0], [3, 2, 1]] # requer muitos passos
    }
    
    heuristics = {
        "1": ("Custo Uniforme", heuristica_custo_uniforme),
        "2": ("A* Nao Admissivel", heuristica_nao_admissivel),
        "3": ("A* Admissivel Simples (Pecas Fora do Lugar)", heuristica_simples_admissivel),
        "4": ("A* Admissivel Precisa (Manhattan)", heuristica_manhattan_admissivel),
    }

    print("Bem-vindo ao Solucionador do 8-Puzzle!")
    print("\nEscolha um tabuleiro inicial:")
    for i, name in enumerate(test_cases.keys()):
        print(f"{i+1}. {name}")
    
    board_choice = int(input("Digite o numero do tabuleiro: "))
    board_name = list(test_cases.keys())[board_choice-1]
    initial_board = test_cases[board_name]

    print("\nEscolha o algoritmo de busca:")
    for key, (name, _) in heuristics.items():
        print(f"{key}. {name}")

    algo_choice = input("Digite o numero do algoritmo: ")
    algo_name, heuristic_function = heuristics[algo_choice]
    
    print(f"\nResolvendo o tabuleiro '{board_name}' com '{algo_name}'...")

    result = a_star_search(initial_board, GOAL, heuristic_function)

    if result:
        print("\n--- Solucao Encontrada! ---")
        print(f"Caminho: {' -> '.join(result['path'])}")
        print("\n--- Metricas de Desempenho ---")
        print(f"1) Total de nodos visitados: {result['nodes_visited']}")
        print(f"2) Tamanho do caminho: {result['path_length']}")
        print(f"3) Tempo de execucao: {result['execution_time']:.6f} segundos")
        print(f"4) Maior tamanho da fronteira: {result['max_fronteira_size']}")

        # Salva o arquivo de saída
        filename = f"output_{board_name}_{algo_choice}.json"
        with open(filename, 'w') as f:
            json.dump(result['output_data'], f, indent=4)
        print(f"\n5) Fronteira e visitados salvos em '{filename}'")
    else:
        print("Nao foi possivel encontrar uma solucao.")
