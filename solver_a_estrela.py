import time
import heapq
import json
import copy

class Node:
    """
    abstracao de um no na arvore de busca. encapsula o estado do quebra-cabeca,
    a referencia ao no progenitor para reconstrucao da trajetoria, a operacao
    que resultou neste estado, e as metricas de custo g (distancia da origem)
    e h (estimativa heuristica ao objetivo).
    """
    def __init__(self, state, parent=None, action=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g  # metrica g: custo acumulado desde o estado inicial ate o no corrente.
        self.h = h  # metrica h: estimativa heuristica do custo do no corrente ao estado-meta.

    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f() < other.f()

    def __repr__(self):
        return f"Node(f={self.f()}, g={self.g}, h={self.h}, state={self.state})"


# mapeamento pre-computado das coordenadas do estado-meta para otimizar o calculo da distancia de manhattan.
GOAL_POSITIONS = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 0: (2, 2)  # o valor 0 simboliza a posicao vazia no tabuleiro.
}

def heuristica_custo_uniforme(state, goal_state):
    """ funcao heuristica nula, h(n)=0, que caracteriza a busca de custo uniforme (dijkstra). """
    return 0

def heuristica_simples_admissivel(state, goal_state):
    """
    heuristica admissivel baseada na contagem de pecas mal posicionadas (distancia de hamming).
    a admissibilidade e garantida pois uma unica operacao pode corrigir, no maximo, a posicao de uma peca.
    """
    misplaced = 0
    for r in range(3):
        for c in range(3):
            if state[r][c] != 0 and state[r][c] != goal_state[r][c]:
                misplaced += 1
    return misplaced

def heuristica_manhattan_admissivel(state, goal_state):
    """
    heuristica admissivel informada pela distancia de manhattan. corresponde ao somatorio
    das distancias l1 de cada peca a sua respectiva posicao-meta. sua admissibilidade
    decorre do fato que cada movimento unitario altera a distancia de manhattan total em exatamente uma unidade.
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
    funcao heuristica nao admissivel que superestima o custo real ao objetivo.
    embora possa acelerar a convergencia, compromete a garantia de otimalidade da solucao encontrada.
    """
    return heuristica_manhattan_admissivel(state, goal_state) * 2

# logica do algoritmo a* 

def find_empty_tile(state):
    """ localiza e retorna as coordenadas (linha, coluna) da posicao vacante (0) no estado atual. """
    for r in range(3):
        for c in range(3):
            if state[r][c] == 0:
                return r, c
    return None

def get_successors(node):
    # expande o no corrente, gerando o conjunto de todos os estados sucessores alcancaveis mediante operacoes validas.
    successors = []
    r, c = find_empty_tile(node.state)

    # definicao do conjunto de operadores de transicao de estado: norte, sul, oeste, leste.
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
    # realiza a reconstrucao da trajetoria otima, percorrendo a arvore de busca retroativamente a partir do no-meta ate o no-raiz.
    path = []
    while node.parent is not None:
        path.append(node.action)
        node = node.parent
    return path[::-1] # inverte a sequencia de acoes para apresentar a solucao na ordem cronologica correta.

def a_star_search(initial_state, goal_state, heuristic_func):
    # implementacao canonica do algoritmo de busca a*.
    start_time = time.time()

    # converte as matrizes de estado para tuplas imutaveis, permitindo sua utilizacao como chaves em estruturas de hash (dicionarios e conjuntos).
    initial_state_tuple = tuple(map(tuple, initial_state))
    goal_state_tuple = tuple(map(tuple, goal_state))

    # a fronteira (open list) e implementada como uma fila de prioridade (min-heap).
    fronteira = []
    start_node = Node(state=initial_state, g=0, h=heuristic_func(initial_state, goal_state))
    heapq.heappush(fronteira, start_node)

    visitados = {initial_state_tuple: start_node.g}

    # variaveis para coleta de metricas de performance do algoritmo.
    nodes_visited_count = 0
    max_fronteira_size = 1

    while fronteira:
        max_fronteira_size = max(max_fronteira_size, len(fronteira))

        # extrai o no com o menor valor de f(n) da fronteira para expansao.
        current_node = heapq.heappop(fronteira)
        nodes_visited_count += 1

        current_state_tuple = tuple(map(tuple, current_node.state))

        # condicao de terminacao: verifica se o estado do no corrente corresponde ao estado-meta.
        if current_state_tuple == goal_state_tuple:
            end_time = time.time()
            path = reconstruct_path(current_node)

            # prepara a estrutura de dados para a serializacao do estado final da fronteira e do conjunto de visitados.
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

        # expansao do no: geracao de todos os nos sucessores.
        for new_state_list, action in get_successors(current_node):
            new_g = current_node.g + 1  # o custo de transicao entre estados adjacentes e uniforme e igual a 1.
            new_state_tuple = tuple(map(tuple, new_state_list))

            if new_state_tuple in visitados and visitados[new_state_tuple] <= new_g:
                continue

            # atualiza o custo do no no conjunto de visitados e o insere na fronteira para avaliacao futura.
            visitados[new_state_tuple] = new_g
            new_h = heuristic_func(new_state_list, goal_state)
            successor_node = Node(state=new_state_list, parent=current_node, action=action, g=new_g, h=new_h)
            heapq.heappush(fronteira, successor_node)

    return None

if __name__ == "__main__":
    GOAL = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    # definicao dos cenarios (quadrados) de teste com diferentes niveis de complexidade.
    test_cases = {
        "facil": [[1, 2, 3], [4, 5, 6], [0, 7, 8]],
        "medio": [[1, 0, 3], [4, 2, 5], [7, 8, 6]],
        "dificil": [[8, 7, 1], [6, 0, 2], [5, 4, 3]],
        "muito_dificil": [[6, 4, 7], [8, 5, 0], [3, 2, 1]] # instancia que exige um numero elevado de movimentos para a solucao.
    }

    heuristics = {
        "1": ("custo uniforme", heuristica_custo_uniforme),
        "2": ("a* nao admissivel", heuristica_nao_admissivel),
        "3": ("a* admissivel simples (pecas fora do lugar)", heuristica_simples_admissivel),
        "4": ("a* admissivel precisa (manhattan)", heuristica_manhattan_admissivel),
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

    algo_choice = input("digite o numero do algoritmo: ")
    algo_name, heuristic_function = heuristics[algo_choice]

    print(f"\nresolvendo o tabuleiro '{board_name}' com '{algo_name}'...")

    result = a_star_search(initial_board, GOAL, heuristic_function)

    if result:
        print("\n--- solucao encontrada! ---")
        print(f"caminho: {' -> '.join(result['path'])}")
        print("\n--- metricas de desempenho ---")
        print(f"1) total de nodos visitados: {result['nodes_visited']}")
        print(f"2) tamanho do caminho: {result['path_length']}")
        print(f"3) tempo de execucao: {result['execution_time']:.6f} segundos")
        print(f"4) maior tamanho da fronteira: {result['max_fronteira_size']}")

        # serializa os dados de saida para um arquivo no formato json.
        filename = f"output_{board_name}_{algo_choice}.json"
        with open(filename, 'w') as f:
            json.dump(result['output_data'], f, indent=4)
        print(f"\n5) fronteira e visitados salvos em '{filename}'")
    else:
        print("nao foi possivel encontrar uma solucao.")