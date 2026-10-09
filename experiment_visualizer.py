# experiment_visualizer.py
import os
import matplotlib.pyplot as plt
import pandas as pd


class ExperimentVisualizer:
    @staticmethod
    def plot_suite_summary(df: pd.DataFrame, algorithm_name: str = "MMAS", save_dir: str = "results"):
        """Будує підсумкові графіки та зберігає файл із префіксом алгоритму."""
        os.makedirs(save_dir, exist_ok=True)
        num_boards = df["Board"].nunique()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        if num_boards <= 20:
            # Для малої кількості задач (стовпчаста діаграма)
            summary = (
                df.groupby("Board")
                .agg({"Iteration": "mean", "Time (s)": "mean"})
                .reset_index()
            )
            boards = summary["Board"]

            bars1 = ax1.bar(boards, summary["Iteration"], color="#2b5c8f", alpha=0.85)
            ax1.set_title(f"[{algorithm_name}] Середня к-сть ітерацій", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Ітерації")
            ax1.tick_params(axis="x", rotation=30)
            ax1.grid(axis="y", linestyle="--", alpha=0.6)
            for b in bars1:
                val = b.get_height()
                ax1.text(b.get_x() + b.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

            bars2 = ax2.bar(boards, summary["Time (s)"], color="#2a9d8f", alpha=0.85)
            ax2.set_title(f"[{algorithm_name}] Середній час розв'язання (с)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Час (с)")
            ax2.tick_params(axis="x", rotation=30)
            ax2.grid(axis="y", linestyle="--", alpha=0.6)
            for b in bars2:
                val = b.get_height()
                ax2.text(b.get_x() + b.get_width() / 2, val, f"{val:.2f}s", ha="center", va="bottom", fontsize=8)
        else:
            # Для великої вибірки (наприклад, 1465 задач) — розподіл гістограмами
            success_rate = df["Success"].mean() * 100
            avg_iter = df["Iteration"].mean()
            avg_time = df["Time (s)"].mean()

            ax1.hist(df["Iteration"], bins=30, color="#2b5c8f", edgecolor="black", alpha=0.8)
            ax1.set_title(f"[{algorithm_name}] Розподіл ітерацій (Сер.: {avg_iter:.1f})", fontsize=11, fontweight="bold")
            ax1.set_xlabel("Ітерації")
            ax1.set_ylabel("Кількість головоломок")
            ax1.grid(True, linestyle="--", alpha=0.5)

            ax2.hist(df["Time (s)"], bins=30, color="#2a9d8f", edgecolor="black", alpha=0.8)
            ax2.set_title(f"[{algorithm_name}] Розподіл часу (Успіх: {success_rate:.1f}%)", fontsize=11, fontweight="bold")
            ax2.set_xlabel("Час (с)")
            ax2.set_ylabel("Кількість головоломок")
            ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{algorithm_name}_summary.png")
        plt.savefig(save_path, dpi=300)
        print(f"Графік збережено: {save_path}")
        plt.close()

    @staticmethod
    def plot_suite_convergence(results: list, algorithm_name: str = "MMAS", max_lines: int = 15, save_dir: str = "results"):
        """Будує та зберігає криві збіжності для перших `max_lines` успішних розв'язків."""
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))

        displayed = 0
        for r in results:
            if r.success and displayed < max_lines:
                plt.plot(r.convergence_history, alpha=0.7, label=f"{r.board_name} ({r.iteration} ітер.)")
                displayed += 1

        plt.xlabel("Ітерація", fontsize=11)
        plt.ylabel("Помилки (Best Fitness)", fontsize=11)
        plt.title(f"[{algorithm_name}] Криві збіжності помилок (вибірка)", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        if displayed <= 8:
            plt.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{algorithm_name}_convergence.png")
        plt.savefig(save_path, dpi=300)
        print(f"Графік збіжності збережено: {save_path}")
        plt.close()