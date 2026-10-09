import os
import random
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SudokuACOForwardChecking import SudokuACOForwardChecking
from test_boards import parse_sudoku_line


def _init_worker():
    """Ігноруємо SIGINT у воркерах для коректної обробки Ctrl+C у головному процесі."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _worker_eval_config(args: Tuple) -> dict:
    """Виконується в окремому процесі для конкретної дошки та набору параметрів."""
    (
        board_name,
        board,
        solver_cls,
        alpha,
        beta,
        max_iterations,
        rho,
        num_ants,
    ) = args

    solver = solver_cls(
        board,
        num_ants=num_ants,
        max_iterations=max_iterations,
        rho=rho,
        alpha=alpha,
        beta=beta,
    )

    start_t = time.perf_counter()
    is_done = False
    while not is_done:
        _, _, is_done = solver.step()

    elapsed = time.perf_counter() - start_t
    success = solver.best_fitness == 0

    return {
        "Config": f"α={alpha}, β={beta}",
        "Alpha": alpha,
        "Beta": beta,
        "Board": board_name,
        "Iteration": solver.iteration,
        "Time (s)": elapsed,
        "Success": 1 if success else 0,
    }


def load_random_sudoku_sample(filepath: str, sample_size: int = 8, seed: int = 42) -> Dict[str, List[List[int]]]:
    """Зчитує випадкові судоку з файлу для об'єктивного тестування."""
    all_boards = []
    with open(filepath, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            board = parse_sudoku_line(line)
            if board is not None:
                all_boards.append((f"Sudoku_{idx}", board))

    random.seed(seed)
    chosen = random.sample(all_boards, min(sample_size, len(all_boards)))
    return {name: board for name, board in chosen}


def plot_and_save_histogram(df: pd.DataFrame, algo_name: str, max_iterations: int = 500, save_dir: str = "results"):
    """Будує та зберігає підсумкові гістограми (ітерації, час та окремо відсоток успіху)."""
    if df.empty:
        print("[Увага] Немає даних для побудови графіків.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Агрегуємо результати за конфігураціями зі збереженням їх початкового порядку
    summary = (
        df.groupby(["Alpha", "Beta", "Config"], sort=False)
        .agg(
            Avg_Iter=("Iteration", "mean"),
            Avg_Time=("Time (s)", "mean"),
            Success_Rate=("Success", "mean"),
        )
        .reset_index()
    )

    configs = summary["Config"]
    avg_iters = summary["Avg_Iter"]
    avg_time = summary["Avg_Time"]
    success_rates = summary["Success_Rate"] * 100

    # ---------------- 1. Подвійний графік: Ітерації та Час ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Гістограма по ітераціях
    bars1 = ax1.bar(configs, avg_iters, color="#1f77b4", edgecolor="black", alpha=0.85)
    ax1.set_title(f"[{algo_name}] Середня к-сть ітерацій до розв'язання", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Параметри (α + β)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Середня кількість ітерацій", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", rotation=40)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    # Гістограма по часу
    bars2 = ax2.bar(configs, avg_time, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax2.set_title(f"[{algo_name}] Середній час виконання (секунди)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Параметри (α + β)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Час (секунди)", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=40)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    plt.tight_layout()
    hist_save_path = os.path.join(save_dir, f"{algo_name}_alpha_beta_histogram.png")
    plt.savefig(hist_save_path, dpi=300)
    plt.close()

    # ---------------- 2. Окремий графік: Відсоток успіху (Success Rate) ----------------
    fig_succ, ax_succ = plt.subplots(figsize=(8, 5.5))
    bars3 = ax_succ.bar(configs, success_rates, color="#e76f51", edgecolor="black", alpha=0.85)
    ax_succ.set_title(
        f"[{algo_name}] Успішність розв'язання (Success Rate) за {max_iterations} ітерацій",
        fontsize=11,
        fontweight="bold",
    )
    ax_succ.set_xlabel("Параметри (α + β)", fontsize=10, fontweight="bold")
    ax_succ.set_ylabel("Успішні розв'язки (%)", fontsize=10, fontweight="bold")
    ax_succ.set_ylim(0, 105)
    ax_succ.tick_params(axis="x", rotation=40)
    ax_succ.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars3:
        h = bar.get_height()
        ax_succ.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.5,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    succ_save_path = os.path.join(save_dir, f"{algo_name}_alpha_beta_success_rate.png")
    plt.savefig(succ_save_path, dpi=300)
    plt.close()

    # ---------------- 3. Експорт даних у CSV ----------------
    csv_path = os.path.join(save_dir, f"{algo_name}_param_results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n[Успіх] Графік ітерацій та часу: {hist_save_path}")
    print(f"[Успіх] Графік відсотка успішності: {succ_save_path}")
    print(f"[Успіх] CSV з результатами: {csv_path}")


def run_benchmark():
    TARGET_ALGORITHM = SudokuACOForwardChecking  
    ALGO_NAME = TARGET_ALGORITHM.__name__
    DATASET_PATH = "top1465.txt"  
    SAMPLE_SIZE = 24                # Можна змінити кількість тестових дощок
    MAX_ITERATIONS = 500
    NUM_ANTS = 10
    RHO = 0.1
    MAX_WORKERS = os.cpu_count() or 4

    # 1. Завантажуємо випадкові дошки
    boards = load_random_sudoku_sample(DATASET_PATH, sample_size=SAMPLE_SIZE, seed=42)

    # 2. Набір комбінацій alpha та beta для дослідження
    alphas = [0.5, 1.0, 1.5]
    betas = [1.0, 2.0, 3.0]

    param_configs = [(a, b) for a in alphas for b in betas]

    tasks = []
    for a, b in param_configs:
        for b_name, b_matrix in boards.items():
            tasks.append((b_name, b_matrix, TARGET_ALGORITHM, a, b, MAX_ITERATIONS, RHO, NUM_ANTS))

    total_tasks = len(tasks)
    print("=" * 65)
    print(f"  Бенчмарк параметрів для алгоритму: {ALGO_NAME}")
    print(f"  Випадкових дощок: {len(boards)} | Конфігурацій (α+β): {len(param_configs)}")
    print(f"  Всього тестів: {total_tasks} | Потоків CPU: {MAX_WORKERS}")
    print("  [Підказка] Натисніть Ctrl+C у будь-який момент для завершення")
    print("  з побудовою графіків по вже завершених тестах!")
    print("=" * 65 + "\n")

    completed_results = []
    start_time = time.perf_counter()

    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker)

    try:
        futures = [executor.submit(_worker_eval_config, t) for t in tasks]

        for idx, future in enumerate(as_completed(futures), start=1):
            res = future.result()
            completed_results.append(res)

            elapsed = time.perf_counter() - start_time
            avg_per_task = elapsed / idx
            eta_mins = ((total_tasks - idx) * avg_per_task) / 60.0
            status_char = "✓" if res["Success"] else "✗"

            print(
                f"[{idx:>3}/{total_tasks}] {status_char} {res['Config']:<15} | "
                f"{res['Board']} | Ітер: {res['Iteration']:<3} | "
                f"Час: {res['Time (s)']:<4.2f}с | Залишилось: ~{eta_mins:.1f} хв",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\n\n>>> Отримано команду переривання (Ctrl+C)! Формуємо графіки з наявних даних...")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        df_res = pd.DataFrame(completed_results)
        plot_and_save_histogram(df_res, algo_name=ALGO_NAME, max_iterations=MAX_ITERATIONS)


if __name__ == "__main__":
    run_benchmark()