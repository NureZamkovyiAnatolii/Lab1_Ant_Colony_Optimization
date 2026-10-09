# benchmark_runner.py
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Type
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SudokuACOForwardChecking import SudokuACOForwardChecking


@dataclass
class SuiteRunResult:
    board_name: str
    iteration: int
    time_sec: float
    best_fitness: int
    success: bool
    convergence_history: List[int]


class ACOBenchmarkRunner:
    """Клас для автоматизованого тестування SudokuACO на наборі дощок."""

    def __init__(
        self,
        solver_class: Type = SudokuACOForwardChecking,
        runs_per_board: int = 3,
        max_iterations: int = 1500,
    ):
        self.solver_class = solver_class
        self.runs_per_board = runs_per_board
        self.max_iterations = max_iterations
        self.results: List[SuiteRunResult] = []

    def _solve_board(
        self,
        board: List[List[int]],
        board_name: str,
        alpha: float,
        beta: float,
        rho: float,
        num_ants: int,
    ) -> SuiteRunResult:
        solver = self.solver_class(
            board,
            num_ants=num_ants,
            max_iterations=self.max_iterations,
            rho=rho,
            alpha=alpha,
            beta=beta,
        )

        history = []
        start_t = time.perf_counter()
        is_done = False

        while not is_done:
            _, _, is_done = solver.step()
            history.append(solver.best_fitness)

        elapsed = time.perf_counter() - start_t
        success = solver.best_fitness == 0

        return SuiteRunResult(
            board_name=board_name,
            iteration=solver.iteration,
            time_sec=elapsed,
            best_fitness=solver.best_fitness,
            success=success,
            convergence_history=history,
        )

    def run_suite_comparison(
        self,
        suite: Dict[str, List[List[int]]],
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        num_ants: int = 20,
    ) -> pd.DataFrame:
        """Проганяє кожну дошку з BENCHMARK_SUITE задану кількість разів."""
        print(f"=== Запуск бенчмарку на {len(suite)} тестових дошках ===")
        self.results.clear()

        for board_idx, (name, board) in enumerate(suite.items(), start=1):
            print(f"[{board_idx}/{len(suite)}] Тестування: {name}...")

            for r_idx in range(self.runs_per_board):
                res = self._solve_board(
                    board=board,
                    board_name=name,
                    alpha=alpha,
                    beta=beta,
                    rho=rho,
                    num_ants=num_ants,
                )
                self.results.append(res)
                status = "Успіх" if res.success else "Ліміт"
                print(
                    f"   Спроба {r_idx + 1}: {status} за {res.iteration} ітер. ({res.time_sec:.2f}с)"
                )

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        records = [
            {
                "Board": r.board_name,
                "Iteration": r.iteration,
                "Time (s)": r.time_sec,
                "Best Fitness": r.best_fitness,
                "Success": 1 if r.success else 0,
            }
            for r in self.results
        ]
        return pd.DataFrame(records)