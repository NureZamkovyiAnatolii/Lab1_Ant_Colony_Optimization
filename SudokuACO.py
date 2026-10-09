import copy
import random
from concurrent.futures import ThreadPoolExecutor


class SudokuACO:
    """Max–Min Ant System (MMAS) для Судоку

    """

    def __init__(
        self,
        board,  # B (початкова матриця / розмітка графа)
        num_ants=20,  # m (кількість мурах у колонії)
        max_iterations=300,  # t_max або N_max (максимальна кількість ітерацій)
        rho=0.1,  # ρ (коефіцієнт випаровування феромону, rho ∈ (0, 1])
        alpha=1.0,  # α (вага сліду феромону τ у формулі переходу)
        beta=3.0,  # β (підвищена вага евристики η для контролю конфліктів)
    ):
        self.initial_board = copy.deepcopy(board)
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.rho = rho
        self.alpha = alpha
        self.beta = beta

        self.best_board = None
        self.best_fitness = float("inf")
        self.iteration = 0

        # Кеш координат тільки вільних клітинок
        self.empty_cells = [
            (r, c)
            for r in range(9)
            for c in range(9)
            if self.initial_board[r][c] == 0
        ]

        # Стабільні межі MMAS, скориговані під цільову функцію помилок Судоку
        self.tau_max = 5.0
        self.tau_min = 0.02

        # Ініціалізація базовим значенням 1.0 (для збереження орієнтації на евристику на старті)
        self.tau = [
            [[1.0 for _ in range(9)] for _ in range(9)]
            for _ in range(9)
        ]

        # Контроль застою (збільшений поріг)
        self.stagnation_counter = 0
        self.stagnation_limit = 45

    def count_conflicts(self, board, r, c, val):
        """Рахує конфлікти для розміщення значення val у клітинці (r, c)."""
        conflicts = 0
        for i in range(9):
            if board[r][i] == val:
                conflicts += 1
            if board[i][c] == val:
                conflicts += 1

        br, bc = (r // 3) * 3, (c // 3) * 3
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                if board[i][j] == val:
                    conflicts += 1
        return conflicts

    def total_board_conflicts(self, board):
        """Обчислює глобальну кількість конфліктів на всій дошці."""
        conflicts = 0
        for i in range(9):
            row_vals = [x for x in board[i] if x != 0]
            col_vals = [board[r][i] for r in range(9) if board[r][i] != 0]
            conflicts += len(row_vals) - len(set(row_vals))
            conflicts += len(col_vals) - len(set(col_vals))

        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                block_vals = [
                    board[r][c]
                    for r in range(br, br + 3)
                    for c in range(bc, bc + 3)
                    if board[r][c] != 0
                ]
                conflicts += len(block_vals) - len(set(block_vals))
        return conflicts

    def get_conflict_cells(self, board):
        """Повертає множину координат (r, c) клітинок, що викликають порушення правил."""
        conflict_cells = set()

        for r in range(9):
            seen = {}
            for c in range(9):
                val = board[r][c]
                if val != 0:
                    if val in seen:
                        conflict_cells.add((r, c))
                        conflict_cells.add((r, seen[val]))
                    else:
                        seen[val] = c

        for c in range(9):
            seen = {}
            for r in range(9):
                val = board[r][c]
                if val != 0:
                    if val in seen:
                        conflict_cells.add((r, c))
                        conflict_cells.add((seen[val], c))
                    else:
                        seen[val] = r

        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                seen = {}
                for r in range(br, br + 3):
                    for c in range(bc, bc + 3):
                        val = board[r][c]
                        if val != 0:
                            if val in seen:
                                conflict_cells.add((r, c))
                                conflict_cells.add(seen[val])
                            else:
                                seen[val] = (r, c)

        return conflict_cells

    def _build_ant_solution(self):
        """Побудова рішення однією мурахою з O(1) перевіркою конфліктів через бітові маски."""
        board = [row[:] for row in self.initial_board]

        row_masks = [0] * 9
        col_masks = [0] * 9
        box_masks = [0] * 9

        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val != 0:
                    mask = 1 << val
                    row_masks[r] |= mask
                    col_masks[c] |= mask
                    box_masks[(r // 3) * 3 + (c // 3)] |= mask

        for r, c in self.empty_cells:
            b_idx = (r // 3) * 3 + (c // 3)
            r_mask = row_masks[r]
            c_mask = col_masks[c]
            b_mask = box_masks[b_idx]

            weights = []
            for v in range(1, 10):
                bit = 1 << v
                conflicts = (
                    ((r_mask & bit) >> v)
                    + ((c_mask & bit) >> v)
                    + ((b_mask & bit) >> v)
                )

                eta = 1.0 / (1.0 + conflicts)
                pheromone = self.tau[r][c][v - 1]
                weights.append((pheromone**self.alpha) * (eta**self.beta))

            chosen_val = random.choices(range(1, 10), weights=weights)[0]
            board[r][c] = chosen_val

            chosen_bit = 1 << chosen_val
            row_masks[r] |= chosen_bit
            col_masks[c] |= chosen_bit
            box_masks[b_idx] |= chosen_bit

        conflicts = self.total_board_conflicts(board)
        return board, conflicts

    def step(self):
        """Виконує одну повну ітерацію MMAS.
        Повертає: (iter_best_board, iter_conflicts, is_done)
        """
        if self.iteration >= self.max_iterations or self.best_fitness == 0:
            return self.best_board, self.best_fitness, True

        self.iteration += 1

        # 1. Паралельний обхід колонією мурах
        with ThreadPoolExecutor(max_workers=min(8, self.num_ants)) as executor:
            ants_solutions = list(
                executor.map(
                    lambda _: self._build_ant_solution(), range(self.num_ants)
                )
            )

        # 2. Пошук найкращого розв'язку поточної ітерації
        iter_best_board, iter_best_conflicts = min(
            ants_solutions, key=lambda x: x[1]
        )

        if iter_best_conflicts < self.best_fitness:
            self.best_fitness = iter_best_conflicts
            self.best_board = copy.deepcopy(iter_best_board)
            self.stagnation_counter = 0

            # Динамічний підйом верхньої межі при поліпшенні рекорду
            self.tau_max = 2.0 + (10.0 / (1.0 + self.best_fitness))
        else:
            self.stagnation_counter += 1

        # 3. Випаровування феромону
        evap_factor = 1.0 - self.rho
        for r in range(9):
            for c in range(9):
                for v in range(9):
                    self.tau[r][c][v] *= evap_factor

        # 4. Підкріплення феромону (MMAS: оновлюємо найкраще рішення)
        deposit_board = self.best_board if self.best_board is not None else iter_best_board
        deposit_fitness = self.best_fitness if self.best_board is not None else iter_best_conflicts

        delta_tau = 1.0 / (1.0 + deposit_fitness)
        for r, c in self.empty_cells:
            val = deposit_board[r][c]
            self.tau[r][c][val - 1] += delta_tau

        # 5. MMAS: Обмеження діапазоном [tau_min, tau_max]
        for r in range(9):
            for c in range(9):
                for v in range(9):
                    if self.tau[r][c][v] < self.tau_min:
                        self.tau[r][c][v] = self.tau_min
                    elif self.tau[r][c][v] > self.tau_max:
                        self.tau[r][c][v] = self.tau_max

        # 6. MMAS: Запобігання застою (м'який рестарт)
        if (
            self.stagnation_counter >= self.stagnation_limit
            and self.best_fitness > 0
        ):
            # Замість повного стирання згладжуємо матрицю, повертаючи шанс альтернативам
            for r in range(9):
                for c in range(9):
                    for v in range(9):
                        self.tau[r][c][v] = (self.tau[r][c][v] + self.tau_max) / 2.0
            self.stagnation_counter = 0

        is_done = (
            self.best_fitness == 0 or self.iteration >= self.max_iterations
        )
        return iter_best_board, iter_best_conflicts, is_done