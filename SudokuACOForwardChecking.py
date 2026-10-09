import copy
import random


class SudokuACOForwardChecking:
    """Клас MMAS для Судоку з пам'яттю доменів (Forward Checking) та евристикою MRV."""

    def __init__(
        self,
        board,  # B (початкова матриця / розмітка графа)
        num_ants=20,  # m (кількість мурах у колонії)
        max_iterations=300,  # t_max або N_max (максимальна кількість ітерацій)
        rho=0.1,  # ρ (коефіцієнт випаровування феромону, rho ∈ (0, 1])
        alpha=1.0,  # α (вага сліду феромону τ у формулі переходу)
        beta=2.0,  # β (вага евристичної інформації η у формулі переходу)
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

        # Початковий кеш порожніх клітинок
        self.empty_cells = [
            (r, c)
            for r in range(9)
            for c in range(9)
            if self.initial_board[r][c] == 0
        ]

        # Базові межі MMAS
        self.tau_max = 5.0
        self.tau_min = 0.02

        # Матриця феромонів 9x9x9
        self.tau = [
            [[1.0 for _ in range(9)] for _ in range(9)]
            for _ in range(9)
        ]

        self.stagnation_counter = 0
        self.stagnation_limit = 35

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

        # Рядки
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

        # Стовпці
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

        # Блоки 3x3
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

    def _get_initial_domains(self):
        """Формує початкові домени можливих цифр для кожної клітинки на основі вихідної дошки."""
        domains = {}
        for r, c in self.empty_cells:
            possible = set(range(1, 10))
            for i in range(9):
                possible.discard(self.initial_board[r][i])
                possible.discard(self.initial_board[i][c])

            br, bc = (r // 3) * 3, (c // 3) * 3
            for i in range(br, br + 3):
                for j in range(bc, bc + 3):
                    possible.discard(self.initial_board[i][j])

            domains[(r, c)] = possible
        return domains

    def _build_ant_solution(self):
        """Побудова розв'язку мурахою за допомогою Forward Checking та евристики MRV."""
        board = [row[:] for row in self.initial_board]
        domains = self._get_initial_domains()
        unassigned = set(self.empty_cells)

        while unassigned:
            # Евристика MRV (Minimum Remaining Values):
            # Спочатку обираємо клітинку, де залишилося найменше допустимих варіантів
            best_cell = min(unassigned, key=lambda cell: len(domains[cell]))
            r, c = best_cell
            unassigned.remove(best_cell)

            valid_values = domains[(r, c)]

            if valid_values:
                # Якщо є легальні ходи без конфліктів — обираємо лише серед них
                weights = []
                val_list = list(valid_values)
                for v in val_list:
                    pheromone = self.tau[r][c][v - 1]
                    # Оскільки конфліктів 0, евристика eta максимальна (1.0)
                    weights.append(pheromone**self.alpha)

                chosen_val = random.choices(val_list, weights=weights)[0]
            else:
                # Глухий кут (домен вичерпано): обираємо значення з мінімальними конфліктами
                weights = []
                for v in range(1, 10):
                    conflicts = self.count_conflicts(board, r, c, v)
                    eta = 1.0 / (1.0 + conflicts)
                    pheromone = self.tau[r][c][v - 1]
                    weights.append((pheromone**self.alpha) * (eta**self.beta))

                chosen_val = random.choices(range(1, 10), weights=weights)[0]

            board[r][c] = chosen_val

            # Forward Checking: викреслюємо обране число з доменів сусідніх вільних клітинок
            br, bc = (r // 3) * 3, (c // 3) * 3
            for other_r, other_c in unassigned:
                if (
                    other_r == r
                    or other_c == c
                    or (br <= other_r < br + 3 and bc <= other_c < bc + 3)
                ):
                    domains[(other_r, other_c)].discard(chosen_val)

        conflicts = self.total_board_conflicts(board)
        return board, conflicts

    def step(self):
        """Виконує одну повну ітерацію MMAS з підтримкою доменів обмежень.
        Повертає: (iter_best_board, iter_conflicts, is_done)
        """
        if self.iteration >= self.max_iterations or self.best_fitness == 0:
            return self.best_board, self.best_fitness, True

        self.iteration += 1

        # 1. Паралельний обхід колонією мурах
        # Стало (швидко, без блокувань GIL та конкуренції пулів):
        ants_solutions = [self._build_ant_solution() for _ in range(self.num_ants)]
        # 2. Пошук найкращого розв'язку раунду
        iter_best_board, iter_best_conflicts = min(
            ants_solutions, key=lambda x: x[1]
        )

        if iter_best_conflicts < self.best_fitness:
            self.best_fitness = iter_best_conflicts
            self.best_board = copy.deepcopy(iter_best_board)
            self.stagnation_counter = 0

            # Підйом верхньої межі при зменшенні помилок (MMAS)[cite: 3]
            self.tau_max = 2.0 + (10.0 / (1.0 + self.best_fitness))
        else:
            self.stagnation_counter += 1

        # 3. Випаровування феромону[cite: 3]
        evap_factor = 1.0 - self.rho
        for r in range(9):
            for c in range(9):
                for v in range(9):
                    self.tau[r][c][v] *= evap_factor

        # 4. Підкріплення феромону для найкращого рішення[cite: 3]
        deposit_board = (
            self.best_board if self.best_board is not None else iter_best_board
        )
        deposit_fitness = (
            self.best_fitness
            if self.best_board is not None
            else iter_best_conflicts
        )

        delta_tau = 1.0 / (1.0 + deposit_fitness)
        for r, c in self.empty_cells:
            val = deposit_board[r][c]
            self.tau[r][c][val - 1] += delta_tau

        # 5. Обмеження діапазоном [tau_min, tau_max] (MMAS)[cite: 3]
        for r in range(9):
            for c in range(9):
                for v in range(9):
                    if self.tau[r][c][v] < self.tau_min:
                        self.tau[r][c][v] = self.tau_min
                    elif self.tau[r][c][v] > self.tau_max:
                        self.tau[r][c][v] = self.tau_max

        # 6. Контроль застою: м'який рестарт феромонів[cite: 3]
        if (
            self.stagnation_counter >= self.stagnation_limit
            and self.best_fitness > 0
        ):
            for r in range(9):
                for c in range(9):
                    for v in range(9):
                        self.tau[r][c][v] = (self.tau[r][c][v] + self.tau_max) / 2.0
            self.stagnation_counter = 0

        is_done = (
            self.best_fitness == 0 or self.iteration >= self.max_iterations
        )
        return iter_best_board, iter_best_conflicts, is_done