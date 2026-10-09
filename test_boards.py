from typing import Dict, List, Optional


def parse_sudoku_line(line: str) -> Optional[List[List[int]]]:
    """Конвертує рядок із 81 символу в матрицю 9x9."""
    line = line.strip()
    if len(line) < 81:
        return None

    board = []
    for r in range(9):
        row = []
        for c in range(9):
            char = line[r * 9 + c]
            row.append(int(char) if char.isdigit() and char != "0" else 0)
        board.append(row)
    return board


def load_benchmark_suite_from_file(
    filepath: str = "sudoku1465.txt", limit: int = 10
) -> Dict[str, List[List[int]]]:
    """Зчитує перші `limit` головоломок із файлу та формує словник для бенчмарку."""
    suite = {}

    with open(filepath, "r", encoding="utf-8") as f:
        idx = 1
        for line in f:
            board = parse_sudoku_line(line)
            if board is None:
                continue

            # Підраховуємо кількість початкових підказок
            hints_count = sum(cell != 0 for row in board for cell in row)
            key_name = f"Sudoku #{idx} ({hints_count} hints)"

            suite[key_name] = board
            idx += 1

            if len(suite) >= limit:
                break

    return suite


# Замість статичного словника:
# Зчитуємо, наприклад, 100 тестових задач із файлу
BENCHMARK_SUITE = load_benchmark_suite_from_file("top1465.txt", limit=100)