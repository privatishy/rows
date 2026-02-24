import numpy as np
import math

# Константы
ROW_COUNT = 5
COLUMN_COUNT = 6
PLAYER = 1
AI = 2
EMPTY = 0

WINDOW_LENGTH = 4

def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
        
def print_board(board):
    print(np.flip(board, 0)) # Переворачиваем, чтобы 0 ряд был внизу

def winning_move(board, piece):
    # Проверка горизонтали
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
            
    # Проверка вертикали
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    
    # Проверка диагонали (вправо-вверх)
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    
    # Проверка диагонали (влево-вверх)
    for c in range(3, COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c-1] == piece and board[r+2][c-2] == piece and board[r+3][c-3] == piece:
                return True
            
# --- ЭВРИСТИКА (Оценка позиции) ---

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER
    if piece == PLAYER:
        opp_piece = AI
    
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
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 3 # Центр важнее

    # Горизонталь
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Вертикаль
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Диагональ (вправо-вверх)    
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Диагональ (влево-вверх)
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+3-i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

def is_terminal_node(board):
    return winning_move(board, PLAYER) or winning_move(board, AI) or len(get_valid_locations(board)) == 0

# --- АЛГОРИТМ MINIMAX С АЛЬФА-БЕТА ОТСЕЧЕНИЕМ ---

def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    # Базовые случаи рекурсии
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI):
                return (None, 10000000000)
            elif winning_move(board, PLAYER):
                return (None, -10000000000)
            else: # Ничья
                return (None, 0)
        else: # Глубина 0
            return (None, score_position(board, AI))
    
    if maximizingPlayer:
        value = -math.inf
        # Сортировка ходов помогает альфа-бета отсечению работать быстрее (центр сначала)
        column = valid_locations[len(valid_locations)//2]
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
    
    else: # Minimazing Player (Человек)
        value = math.inf
        column = valid_locations[len(valid_locations)//2]
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
