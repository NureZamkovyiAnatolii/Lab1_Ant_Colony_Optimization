import time
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from SudokuACO import SudokuACO
from SudokuACOForwardChecking import SudokuACOForwardChecking
from SudokuACODomainInverseHeuristic import SudokuACODomainInverseHeuristic
INITIAL_BOARD = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

HARD_INITIAL_BOARD = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0],
]


class SudokuVisualizer:
    """Інтерфейс з MenuStrip (tk.Menu), налаштуванням затримки та вкладками ttk.Notebook."""

    CELL_SIZE = 48

    def __init__(
        self,
        root,
        solver_class,
        board,
        num_ants=20,
        max_iterations=30000,
        rho=0.1,
        alpha=1.0,
        beta=2.0,
        initial_delay_ms=80,  # Затримка між ітераціями за замовчуванням (у мс)
    ):
        self.root = root
        self.solver_class = solver_class
        self.initial_board = board
        self.solver_kwargs = {
            "num_ants": num_ants,
            "max_iterations": max_iterations,
            "rho": rho,
            "alpha": alpha,
            "beta": beta,
        }

        self.solver = self._create_solver()
        self.is_running = False
        self.start_time = None
        self.loop_id = None
        self.delay_ms = initial_delay_ms

        self.root.title("ACO Sudoku Solver — MenuStrip та Регулювання Швидкості")
        self._init_menu()
        self._init_ui()

    def _create_solver(self):
        return self.solver_class(self.initial_board, **self.solver_kwargs)

    def _init_menu(self):
        """Створює глобальне верхнє меню (MenuStrip)."""
        self.menu_bar = tk.Menu(self.root)

        # 1. Меню "Керування"
        self.control_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.control_menu.add_command(
            label="▶ Старт", command=self.start_solving, accelerator="Ctrl+S"
        )
        self.control_menu.add_command(
            label="🔄 Рестарт", command=self.restart_solving, accelerator="Ctrl+R"
        )

        # Підменю вибору попередньо встановленої швидкості
        speed_menu = tk.Menu(self.control_menu, tearoff=0)
        speed_menu.add_command(label="Максимальна (0 мс)", command=lambda: self.set_speed(0))
        speed_menu.add_command(label="Швидка (25 мс)", command=lambda: self.set_speed(25))
        speed_menu.add_command(label="Помірна (100 мс)", command=lambda: self.set_speed(100))
        speed_menu.add_command(label="Повільна (300 мс)", command=lambda: self.set_speed(300))
        self.control_menu.add_cascade(label="⏱ Швидкість анімації", menu=speed_menu)

        self.control_menu.add_separator()
        self.control_menu.add_command(label="❌ Вихід", command=self.root.quit)
        self.menu_bar.add_cascade(label="Керування", menu=self.control_menu)

        # 2. Меню вибору вкладок
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(
            label="1. 2D Дошка Судоку",
            command=lambda: self.switch_tab(0),
            accelerator="Alt+1",
        )
        view_menu.add_command(
            label="2. 3D Візуалізація Феромонів",
            command=lambda: self.switch_tab(1),
            accelerator="Alt+2",
        )
        view_menu.add_command(
            label="3. Параметри алгоритму",
            command=lambda: self.switch_tab(2),
            accelerator="Alt+3",
        )
        self.menu_bar.add_cascade(label="Вкладки", menu=view_menu)

        self.root.config(menu=self.menu_bar)

        # Гарячі клавіші
        self.root.bind("<Control-s>", lambda event: self.start_solving())
        self.root.bind("<Control-r>", lambda event: self.restart_solving())
        self.root.bind("<Alt-Key-1>", lambda event: self.switch_tab(0))
        self.root.bind("<Alt-Key-2>", lambda event: self.switch_tab(1))
        self.root.bind("<Alt-Key-3>", lambda event: self.switch_tab(2))

    def set_speed(self, delay):
        """Змінює затримку програмно та оновлює повзунок."""
        self.delay_ms = delay
        self.speed_scale.set(delay)

    def switch_tab(self, tab_index):
        """Програмно активує обрану вкладку."""
        self.notebook.select(tab_index)
        if tab_index == 1:
            self.update_3d_plot()

    def _init_ui(self):
        # Панель статусу під меню
        status_frame = tk.Frame(self.root, bg="#f0f0f0", pady=5)
        status_frame.pack(side=tk.TOP, fill=tk.X)

        self.info_label = tk.Label(
            status_frame,
            text="Готово до запуску. Оберіть 'Керування' -> 'Старт'",
            font=("Helvetica", 10, "bold"),
            bg="#f0f0f0",
        )
        self.info_label.pack(side=tk.LEFT, padx=15)

        # Контейнер повзунка затримки на панелі
        speed_frame = tk.Frame(status_frame, bg="#f0f0f0")
        speed_frame.pack(side=tk.RIGHT, padx=15)

        lbl_delay = tk.Label(
            speed_frame, text="Затримка (мс):", font=("Helvetica", 9), bg="#f0f0f0"
        )
        lbl_delay.pack(side=tk.LEFT, padx=3)

        self.speed_scale = tk.Scale(
            speed_frame,
            from_=0,
            to=500,
            orient=tk.HORIZONTAL,
            length=130,
            showvalue=True,
            bg="#f0f0f0",
            highlightthickness=0,
            command=self._on_scale_change,
        )
        self.speed_scale.set(self.delay_ms)
        self.speed_scale.pack(side=tk.LEFT)

        # Приховуємо графічні шапки вкладок
        style = ttk.Style()
        style.layout("Hidden.TNotebook.Tab", [])
        self.notebook = ttk.Notebook(self.root, style="Hidden.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Вкладка 1: 2D Дошка Судоку
        self.tab_2d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_2d, text="2D Дошка")

        side = self.CELL_SIZE * 9
        self.canvas = tk.Canvas(self.tab_2d, width=side, height=side, bg="white")
        self.canvas.pack(padx=15, pady=15)

        # Вкладка 2: 3D Матриця Феромонів
        self.tab_3d = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_3d, text="3D Матриця")

        self.fig = Figure(figsize=(6.2, 5.2), dpi=95)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Вкладка 3: Параметри алгоритму
        self.tab_info = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_info, text="Параметри")
        self._init_info_tab()

        self.draw_grid()
        self.draw_board(self.solver.initial_board, set())
        self.update_3d_plot()

    def _on_scale_change(self, val):
        self.delay_ms = int(val)

    def _init_info_tab(self):
        info_text = (
            f"Модифікація: {self.solver_class.__name__}\n\n"
            f"Параметри моделі:\n"
            f" • Кількість мурах (m): {self.solver_kwargs['num_ants']}\n"
            f" • Ліміт ітерацій (t_max): {self.solver_kwargs['max_iterations']}\n"
            f" • Коефіцієнт випаровування (ρ): {self.solver_kwargs['rho']}\n"
            f" • Вага феромону (α): {self.solver_kwargs['alpha']}\n"
            f" • Вага евристики конфліктів (β): {self.solver_kwargs['beta']}\n"
        )
        lbl = tk.Label(
            self.tab_info,
            text=info_text,
            justify=tk.LEFT,
            font=("Consolas", 11),
            padx=20,
            pady=20,
        )
        lbl.pack(anchor="nw")

    def draw_grid(self):
        for i in range(10):
            width = 3 if i % 3 == 0 else 1
            coord = i * self.CELL_SIZE
            self.canvas.create_line(0, coord, 9 * self.CELL_SIZE, coord, width=width, fill="black")
            self.canvas.create_line(coord, 0, coord, 9 * self.CELL_SIZE, width=width, fill="black")

    def draw_board(self, board, conflict_cells):
        self.canvas.delete("cell_bg")
        self.canvas.delete("digits")

        for r in range(9):
            for c in range(9):
                x1 = c * self.CELL_SIZE
                y1 = r * self.CELL_SIZE
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE

                if (r, c) in conflict_cells:
                    self.canvas.create_rectangle(
                        x1 + 1, y1 + 1, x2, y2, fill="#ffcccc", outline="", tags="cell_bg"
                    )

                val = board[r][c]
                if val != 0:
                    cx = x1 + self.CELL_SIZE // 2
                    cy = y1 + self.CELL_SIZE // 2
                    is_initial = self.solver.initial_board[r][c] != 0
                    color = "black" if is_initial else "#0055d4"
                    font = ("Helvetica", 15, "bold") if is_initial else ("Helvetica", 14)

                    self.canvas.create_text(
                        cx, cy, text=str(val), fill=color, font=font, tags="digits"
                    )

    def update_3d_plot(self):
        self.ax.clear()
        THRESHOLD = 0.5

        for r in range(9):
            for c in range(9):
                for v in range(9):
                    tau_val = self.solver.tau[r][c][v]
                    if tau_val >= THRESHOLD:
                        self.ax.text(
                            c + 1,
                            r + 1,
                            v + 1,
                            f"{tau_val:.2f}",
                            color="black",
                            fontsize=6,
                            ha="center",
                            va="center",
                        )

        self.ax.set_xlim(1, 9)
        self.ax.set_ylim(1, 9)
        self.ax.set_zlim(1, 9)

        self.ax.set_xticks(range(1, 10))
        self.ax.set_yticks(range(1, 10))
        self.ax.set_zticks(range(1, 10))

        self.ax.set_xlabel("Стовпець (Col)", fontsize=8)
        self.ax.set_ylabel("Рядок (Row)", fontsize=8)
        self.ax.set_zlabel("Цифра (Val)", fontsize=8)
        self.ax.view_init(elev=22, azim=45)

        self.canvas_3d.draw_idle()

    def start_solving(self):
        if not self.is_running:
            self.is_running = True
            if self.start_time is None:
                self.start_time = time.time()
            self.control_menu.entryconfig("▶ Старт", state="disabled")
            self._update_loop()

    def restart_solving(self):
        """Скидає виконання, очищує дошку та перезапускає стан моделі."""
        if self.loop_id is not None:
            self.root.after_cancel(self.loop_id)
            self.loop_id = None

        self.is_running = False
        self.start_time = None
        self.control_menu.entryconfig("▶ Старт", state="normal")

        self.solver = self._create_solver()
        self.draw_board(self.solver.initial_board, set())
        self.update_3d_plot()
        self.info_label.config(text="Стан скинуто. Натисніть 'Старт'")

    def _update_loop(self):
        board, conflicts, is_done = self.solver.step()
        conflict_cells = self.solver.get_conflict_cells(board)
        elapsed_time = time.time() - self.start_time

        self.draw_board(board, conflict_cells)
        self.info_label.config(
            text=f"Час: {elapsed_time:.1f}с | Ітер: {self.solver.iteration} | Помилок: {conflicts} | Рекорд: {self.solver.best_fitness}"
        )

        current_tab = self.notebook.index(self.notebook.select())
        if current_tab == 1 and (self.solver.iteration % 2 == 0 or is_done):
            self.update_3d_plot()

        if is_done:
            if self.solver.best_fitness == 0:
                status = f"Розв'язок знайдено за {elapsed_time:.2f}с ({self.solver.iteration} ітер.)!"
            else:
                status = f"Ліміт ітерацій вичерпано за {elapsed_time:.2f}с. Рекорд: {self.solver.best_fitness}"

            self.info_label.config(text=status)
            self.update_3d_plot()
            self.is_running = False
            self.control_menu.entryconfig("▶ Старт", state="normal")
            return

        # Використовуємо динамічне значення затримки з повзунка / налаштувань
        delay = max(1, self.delay_ms)
        self.loop_id = self.root.after(delay, self._update_loop)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("680x640")

    app = SudokuVisualizer(
        root=root,
        solver_class=SudokuACODomainInverseHeuristic,
        board=HARD_INITIAL_BOARD,
        num_ants=20,
        max_iterations=30000,
        rho=0.1,
        alpha=1.0,
        beta=1.0,
        initial_delay_ms=80,  # Плавний крок у 80 мс на старті
    )

    root.mainloop()