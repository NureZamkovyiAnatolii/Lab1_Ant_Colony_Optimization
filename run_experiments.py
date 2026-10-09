import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import pandas as pd

from experiment_visualizer import ExperimentVisualizer
from SudokuACODomainInverseHeuristic import SudokuACODomainInverseHeuristic
from SudokuACOForwardChecking import SudokuACOForwardChecking
from test_boards import load_benchmark_suite_from_file


def _init_worker():
    """Ігноруємо SIGINT у дочірніх процесах, щоб головний процес міг

    коректно перехопити Ctrl+C, зупинити пул і зберегти поточні результати.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_solve_single_board(task_args: Tuple) -> dict:
    """Виконується в окремому ізольованому процесі."""
    (
        board_name,
        board,
        solver_cls,
        max_iterations,
        alpha,
        beta,
        rho,
        num_ants,
    ) = task_args

    solver = solver_cls(
        board,
        num_ants=num_ants,
        max_iterations=max_iterations,
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

    return {
        "Board": board_name,
        "Iteration": solver.iteration,
        "Time (s)": elapsed,
        "Best Fitness": solver.best_fitness,
        "Success": 1 if success else 0,
        "history": history,
    }


class MockResult:
    """Адаптер для коректної побудови графіків у ExperimentVisualizer."""
    def __init__(self, d: dict):
        self.board_name = d["Board"]
        self.iteration = d["Iteration"]
        self.success = d["Success"] == 1
        self.convergence_history = d["history"]


def save_and_visualize_results(raw_results: List[dict], algo_name: str):
    """Зберігає графіки та CSV для поточної кількості зібраних результатів."""
    if not raw_results:
        print("\n[Увага] Немає жодного завершеного тесту для збереження.")
        return

    print(f"\n=== Збереження результатів для {len(raw_results)} оброблених дощок ===")
    df = pd.DataFrame(raw_results).drop(columns=["history"])

    # 1. Графіки
    objects_list = [MockResult(r) for r in raw_results]
    ExperimentVisualizer.plot_suite_summary(df, algorithm_name=algo_name)
    ExperimentVisualizer.plot_suite_convergence(objects_list, algorithm_name=algo_name)

    # 2. CSV
    csv_filename = f"{algo_name}_parallel_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Звіт збережено у: {csv_filename}")
    print(f"Графіки збережено у папку: results/")


def run_parallel_suite(
    suite: Dict[str, List[List[int]]],
    solver_cls,
    max_iterations: int = 800,
    alpha: float = 1.0,
    beta: float = 2.0,
    rho: float = 0.1,
    num_ants: int = 15,
    max_workers: int = None,
):
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    total_tasks = len(suite)
    algo_name = solver_cls.__name__

    print(f"============================================================")
    print(f"  Алгоритм: {algo_name}")
    print(f"  Всього задач: {total_tasks} | Ядер CPU: {max_workers}")
    print(f"  [Підказка] Натисніть Ctrl+C у будь-який момент, щоб")
    print(f"  зупинити тест і зберегти графіки з поточними даними!")
    print(f"============================================================\n")

    tasks = [
        (name, board, solver_cls, max_iterations, alpha, beta, rho, num_ants)
        for name, board in suite.items()
    ]

    completed_results = []
    start_time = time.perf_counter()

    executor = ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker)

    try:
        # Відправляємо всі завдання в чергу
        futures = [executor.submit(_worker_solve_single_board, task) for task in tasks]

        for idx, future in enumerate(as_completed(futures), start=1):
            res = future.result()
            completed_results.append(res)

            elapsed_total = time.perf_counter() - start_time
            avg_time = elapsed_total / idx
            eta_seconds = (total_tasks - idx) * avg_time
            success_count = sum(r["Success"] for r in completed_results)
            success_pct = (success_count / idx) * 100

            # Друк детального прогресу
            status_char = "✓" if res["Success"] else "✗"
            print(
                f"[{idx:>4}/{total_tasks}] {status_char} {res['Board']} | "
                f"Ітер: {res['Iteration']:<3} | Час: {res['Time (s)']:<4.2f}с | "
                f"Успіх: {success_pct:>5.1f}% | Залишилось: ~{eta_seconds/60:.1f} хв",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n\n>>> Отримано команду на зупинку (Ctrl+C)! Перериваємо процес...")
    finally:
        # Примусово зупиняємо незавершені процеси без зависання
        executor.shutdown(wait=False, cancel_futures=True)
        # Зберігаємо все, що встигло завершитися до моменту переривання
        save_and_visualize_results(completed_results, algo_name)


if __name__ == "__main__":
    # Обираємо клас алгоритму
    TARGET_ALGORITHM = SudokuACOForwardChecking
    
    # Файл із набором (top1465.txt або sudoku1465.txt)
    DATASET_FILE = "top1465.txt"
    suite = load_benchmark_suite_from_file(filepath=DATASET_FILE, limit=1465)

    run_parallel_suite(
        suite=suite,
        solver_cls=TARGET_ALGORITHM,
        max_iterations=800,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        num_ants=15,
        max_workers=None,  # Автоматично задіяти всі ядра
    )