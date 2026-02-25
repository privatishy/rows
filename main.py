import numpy as np
import math
import json
import re

# Константы
ROW_COUNT = 5
COLUMN_COUNT = 6
PLAYER = 1
AI = 2
EMPTY = 0
WINDOW_LENGTH = 4


# Глобальное состояние
game_mode = "play" # "play", "editor", "puzzle"
current_puzzle_piece = PLAYER # Для режима головоломки: чей ход анализируем

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    return None
        
def print_board(board):
    print("\n " + " ".join(f"{c}" for c in range(COLUMN_COUNT)))
    print(" " + "-" * (COLUMN_COUNT * 2 - 1))
    for r in reversed(range(ROW_COUNT)):
        row_display = []
        for c in range(COLUMN_COUNT):
            cell = board[r][c]
            if cell == PLAYER:
                row_display.append("🔴")  # Красный (игрок)
            elif cell == AI:
                row_display.append("🟡")  # Жёлтый (AI)
            else:
                row_display.append("⚪")  # Пусто
        print(f"{r} " + " ".join(row_display))
    print()

def winning_move(board, piece):
    # Горизонталь
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    # Вертикаль
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    # Диагональ ↗
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    # Диагональ ↖
    for c in range(3, COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if all(board[r+i][c-i] == piece for i in range(4)):
                return True
    return False

def evaluate_window(window, piece):
    score = 0
    opp_piece = AI if piece == PLAYER else PLAYER
    
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4
    return score

def score_position(board, piece):
    score = 0
    # Центр
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    score += center_array.count(piece) * 3
    
    # Все направления
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            score += evaluate_window(row_array[c:c+4], piece)
    
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            score += evaluate_window(col_array[r:r+4], piece)
    
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            # ↗
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window, piece)
            # ↖
            window = [board[r+i][c+3-i] for i in range(4)]
            score += evaluate_window(window, piece)
    return score

def get_valid_locations(board):
    return [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]

def is_terminal_node(board):
    return winning_move(board, PLAYER) or winning_move(board, AI) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI):
                return (None, 10000000000)
            elif winning_move(board, PLAYER):
                return (None, -10000000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI))
    
    if maximizingPlayer:
        value = -math.inf
        column = valid_locations[len(valid_locations)//2] if valid_locations else 0
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI)
            new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = valid_locations[len(valid_locations)//2] if valid_locations else 0
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER)
            new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value
    
def get_best_move(board, piece, depth=5):
    """Универсальная функция для получения лучшего хода для любого игрока"""
    valid_locations = get_valid_locations(board)
    if not valid_locations:
        return None, 0

    # Если анализируем ход PLAYER, меняем логику minimax
    if piece == PLAYER:
        best_score = -math.inf
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, piece)
            # Запускаем minimax от имени противника
            _, score = minimax(b_copy, depth-1, -math.inf, math.inf, False)
            if score > best_score:
                best_score = score
                best_col = col
        return best_col, best_score
    else:
        return minimax(board, depth, -math.inf, math.inf, True)

# ========== РЕДАКТОР ДОСКИ ==========
def editor_mode(board):
    """Режим ручного редактирования доски"""
    print("\n🎨 РЕДАКТОР ДОСКИ")
    print("Команды:")
    print("  <col> <piece>  - поставить фишку (1=🔴 игрок, 2=🟡 AI, 0=очистить)")
    print("  clear          - очистить всю доску")
    print("  done           - закончить редактирование")
    print("  save <name>    - сохранить позицию")
    print("  load <name>    - загрузить позицию")
    print("  puzzle <1|2>   - перейти в режим головоломки (чей ход: 1 или 2)")

    positions = {} # Для сохранения позиций

    while True:
        print_board(board)
        cmd = input("Редактор > ").strip().split()
        if not cmd:
            continue

        if cmd[0] == "done":
            return board
        elif cmd[0] == "clear":
            board = create_board()
            print(" Доска очищена")
        elif cmd[0] == "save" and len(cmd) >= 2:
            positions[cmd[1]] = board.copy()
            print(" Позиция '{cmd[1]}' сохранена")
        elif cmd[0] == "load" and len(cmd) >= 2 and cmd[1] in positions:
            board = positions[cmd[1]].copy()
            print(" Позиция '{cmd[1]}' загружена")
        elif cmd[0] == "puzzle" and len(cmd) >= 2:
            try:
                global current_puzzle_piece
                current_puzzle_piece = int(cmd[1])
                if current_puzzle_piece not in [PLAYER, AI]:
                    print(" Используйте 1 (игрок) или 2 (AI)")
                    continue
                print(f" Режим головоломки: анализируем ход для {' Игрока' if current_puzzle_piece == PLAYER else ' AI'}")
                return board
            except ValueError:
                print(" Ошибка: укажите 1 или 2")
        elif len(cmd) == 2:
            try:
                col, piece = int(cmd[0]), int(cmd[1])
                if not (0 <= col < COLUMN_COUNT and piece in [0, 1, 2]):
                    print(" Колонка: 0-5, Фишка: 0=очистить, 1=🔴, 2=🟡")
                    continue
                if piece == 0:
                    # Очистить колонку сверху вниз
                    for r in range(ROW_COUNT):
                        if board[r][col] != EMPTY:
                            board[r][col] = EMPTY
                            break
                    
                else:
                    row = get_next_open_row(board, col)
                    if row is not None:
                        board[row][col] = piece
                    else:
                        print(" Колонка заполнена")
            except ValueError:
                print(" Формат: <колонка> <фишка>")
        else:
            print(" Неизвестная команда")

# ========== РЕЖИМ ГОЛОВОЛОМКИ ==========
def puzzle_mode(board, piece_to_analyze):
    """Режим анализа позиции: подсказки для любого игрока"""
    print(f"\n🧩 РЕЖИМ ГОЛОВОЛОМКИ")
    print(f"Анализируем лучший ход для: {'🔴 Игрока' if piece_to_analyze==PLAYER else '🟡 AI'}")
    print("Команды:")
    print("  hint           - получить подсказку")
    print("  analyze <col>  - оценить ход в колонку")
    print("  play           - продолжить игру с этой позиции")
    print("  edit           - вернуться в редактор")
    print("  <col>          - сделать ход (если хотите сыграть)")

    while True:
        print_board(board)

# --- ГЛАВНЫЙ ЦИКЛ ИГРЫ ---

board = create_board()
print_board(board)
game_over = False
turn = PLAYER # Ходит первым человек

def get_hint(board):
    """Вычисляет лучший ход для ИГРОКА на основе оценки позиции"""
    valid_locations = get_valid_locations(board)
    if not valid_locations:
        return None

    best_score = -math.inf
    best_col = valid_locations[0]

    for col in valid_locations:
        row = get_next_open_row(board, col)
        b_copy = board.copy()
        drop_piece(b_copy, row, col, PLAYER)

        # Оцениваем позицию для игрока
        score = score_position(b_copy, PLAYER)

        # Проверяем, не ведет ли ход к немедленному проигрышу (простая защита)
        # Для полноценной защиты нужно запускать minimax, но для подсказки хватит оценки
        if winning_move(b_copy, PLAYER):
            score += 100000 # Приоритет победе

        if score > best_score:
            best_score = score
            best_col = col
        
    return best_col

while not game_over:
    if turn == PLAYER:
        # Ход человека
        user_input = input("Введите номер колонки (0-5) или 'h' для подсказки: ")

        # --- Проверка на подсказку ---
        if user_input.lower() == 'h': 
            hint_col = get_hint(board)
            if hint_col is not None:
                print(f">>> ПОДСКАЗКА: Лучше всего сходить в колонку {hint_col}")
            else:
                print(">>> Нет доступных ходов!")
            continue # Пропускаем ход, игрок должен ввести число
        # -------------------------------------

        try:
            col = int(user_input)
        except ValueError:
            print("Пожалуйста, введите число или 'h'.")
            continue

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, PLAYER)

            if winning_move(board, PLAYER):
                print("Вы победили!")
                game_over = True
            turn += 1
            turn = turn % 2
    else:
        # Ход компьютера
        print("Компьютер думает...")
        # depth=5 достаточно для быстрой и сильной игры
        col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, AI)

            if winning_move(board, AI):
                print("Компьютер победил!")
                game_over = True

            print_board(board)
            turn += 1
            turn = turn % 2
