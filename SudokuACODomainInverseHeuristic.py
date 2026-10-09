import copy
import random
from concurrent.futures import ThreadPoolExecutor


class SudokuACODomainInverseHeuristic:
    """Модифікація MMAS для Судоку з оберненою евристикою варіантів:

    eta = 1.0 / (1.0 + opts)
    """

    def __init__(
        self,
        board,  # B (початкова матриця Судоку) 
        num_ants=20,  # m (кількість мурах) 
        max_iterations=300,  # t_max (ліміт ітерацій) 
        rho=0.1,  # ρ (випаровування) 
        alpha=1.0,  # α (вага феромону) 
        beta=2.0,  # β (вага оберненої евристики варіантів) 
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

        self.empty_cells = [
            (r, c)
            for r in range(9)
            for c in range(9)
            if self.initial_board[r][c] == 0
        ] 

        self.tau_max = 5.0 
        self.tau_min = 0.02 
        self.tau = [
            [[1.0 for _ in range(9)] for _ in range(9)] for _ in range(9)
        ] 

        self.stagnation_counter = 0 
        self.stagnation_limit = 35 

    def total_board_conflicts(self, board):
        """Обчислює глобальну кількість конфліктів на дошці."""
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
        """Повертає множину координат клітинок із помилками."""
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

    def _get_initial_domains(self):
        """Формує початкові домени допустимих значень."""
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

    def _count_affected_options(self, r, c, val, domains, unassigned):
        """Рахує, скільки варіантів віднімається у сусідів (або скільки варіантів залишається).

        Тут рахуємо саме кількість 'залишкових' опцій у сусідів (opts).
        """
        br, bc = (r // 3) * 3, (c // 3) * 3
        opts = 0

        for other_r, other_c in unassigned:
            if (other_r, other_c) == (r, c):
                continue

            is_neighbor = (
                other_r == r
                or other_c == c
                or (br <= other_r < br + 3 and bc <= other_c < bc + 3)
            )

            current_len = len(domains[(other_r, other_c)])
            if is_neighbor and (val in domains[(other_r, other_c)]):
                opts += max(0, current_len - 1)
            else:
                opts += current_len

        return opts

    def _build_ant_solution(self):
        """Побудова розв'язку мурахою з оберненою евристикою eta = 1 / (1 + opts)."""
        board = [row[:] for row in self.initial_board] 
        domains = self._get_initial_domains() 
        unassigned = set(self.empty_cells) 

        while unassigned: 
            # MRV: обираємо клітинку з найменшим доменом 
            best_cell = min(unassigned, key=lambda cell: len(domains[cell])) 
            r, c = best_cell 
            unassigned.remove(best_cell) 

            valid_values = domains[(r, c)] 

            if valid_values: 
                val_list = list(valid_values) 
                weights = []
                for v in val_list:
                    opts = self._count_affected_options(
                        r, c, v, domains, unassigned
                    )
                    # Обернена евристика:
                    eta = 1.0 / (1.0 + opts)
                    pheromone = self.tau[r][c][v - 1] 
                    weights.append((pheromone**self.alpha) * (eta**self.beta))

                chosen_val = random.choices(val_list, weights=weights)[0] 
            else:
                # Глухий кут (домен порожній): обираємо з 1..9 
                weights = [] 
                for v in range(1, 10): 
                    opts = self._count_affected_options(
                        r, c, v, domains, unassigned
                    )
                    eta = 1.0 / (1.0 + opts)
                    pheromone = self.tau[r][c][v - 1] 
                    weights.append((pheromone**self.alpha) * (eta**self.beta)) 

                chosen_val = random.choices(range(1, 10), weights=weights)[0] 

            board[r][c] = chosen_val 

            # Forward Checking 
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
        """Виконує одну повну ітерацію MMAS."""
        if self.iteration >= self.max_iterations or self.best_fitness == 0: 
            return self.best_board, self.best_fitness, True 

        self.iteration += 1 

        ants_solutions = [self._build_ant_solution() for _ in range(self.num_ants)]

        iter_best_board, iter_best_conflicts = min(
            ants_solutions, key=lambda x: x[1]
        ) 

        if iter_best_conflicts < self.best_fitness: 
            self.best_fitness = iter_best_conflicts 
            self.best_board = copy.deepcopy(iter_best_board) 
            self.stagnation_counter = 0 
            self.tau_max = 2.0 + (10.0 / (1.0 + self.best_fitness)) 
        else:
            self.stagnation_counter += 1 

        # Випаровування 
        evap_factor = 1.0 - self.rho 
        for r in range(9): 
            for c in range(9): 
                for v in range(9): 
                    self.tau[r][c][v] *= evap_factor 

        # Підкріплення феромону 
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

        # Межі MMAS [tau_min, tau_max] 
        for r in range(9): 
            for c in range(9): 
                for v in range(9): 
                    if self.tau[r][c][v] < self.tau_min: 
                        self.tau[r][c][v] = self.tau_min 
                    elif self.tau[r][c][v] > self.tau_max: 
                        self.tau[r][c][v] = self.tau_max 

        # Згладжування застою 
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