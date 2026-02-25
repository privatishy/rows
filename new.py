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
game_mode = "play"  # "play", "editor", "puzzle"
current_puzzle_piece = PLAYER  # Для режима головоломки: чей ход анализируем

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
    print("\n  " + " ".join(f"{c}" for c in range(COLUMN_COUNT)))
    print("  " + "-" * (COLUMN_COUNT * 2 - 1))
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
    
    positions = {}  # Для сохранения позиций
    
    while True:
        print_board(board)
        cmd = input("Редактор > ").strip().split()
        if not cmd:
            continue
            
        if cmd[0] == "done":
            return board
        elif cmd[0] == "clear":
            board = create_board()
            print("✓ Доска очищена")
        elif cmd[0] == "save" and len(cmd) >= 2:
            positions[cmd[1]] = board.copy()
            print(f"✓ Позиция '{cmd[1]}' сохранена")
        elif cmd[0] == "load" and len(cmd) >= 2 and cmd[1] in positions:
            board = positions[cmd[1]].copy()
            print(f"✓ Позиция '{cmd[1]}' загружена")
        elif cmd[0] == "puzzle" and len(cmd) >= 2:
            try:
                global current_puzzle_piece
                current_puzzle_piece = int(cmd[1])
                if current_puzzle_piece not in [PLAYER, AI]:
                    print("✗ Используйте 1 (игрок) или 2 (AI)")
                    continue
                print(f"✓ Режим головоломки: анализируем ход для {'🔴 Игрока' if current_puzzle_piece==PLAYER else '🟡 AI'}")
                return board
            except ValueError:
                print("✗ Ошибка: укажите 1 или 2")
        elif len(cmd) == 2:
            try:
                col, piece = int(cmd[0]), int(cmd[1])
                if not (0 <= col < COLUMN_COUNT and piece in [0, 1, 2]):
                    print("✗ Колонка: 0-5, Фишка: 0=очистить, 1=🔴, 2=🟡")
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
                        print("✗ Колонка заполнена")
            except ValueError:
                print("✗ Формат: <колонка> <фишка>")
        else:
            print("✗ Неизвестная команда")

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
        
        # Проверка окончания
        if winning_move(board, PLAYER):
            print("🎉 🔴 Игрок победил!")
            return board, "play"
        if winning_move(board, AI):
            print("🎉 🟡 AI победил!")
            return board, "play"
        if not get_valid_locations(board):
            print("🤝 Ничья!")
            return board, "play"
        
        cmd = input("Головоломка > ").strip()
        if not cmd:
            continue
            
        if cmd == "hint":
            col, score = get_best_move(board, piece_to_analyze, depth=6)
            if col is not None:
                symbol = "🔴" if piece_to_analyze == PLAYER else "🟡"
                print(f"\n💡 ПОДСКАЗКА: {symbol} Лучше всего сходить в колонку {col} (оценка: {score})")
                # Показать альтернативы
                print("Альтернативы:")
                for c in get_valid_locations(board):
                    row = get_next_open_row(board, c)
                    b_copy = board.copy()
                    drop_piece(b_copy, row, c, piece_to_analyze)
                    alt_score = score_position(b_copy, piece_to_analyze)
                    marker = "⭐" if c == col else "  "
                    print(f"  {marker} Колонка {c}: оценка {alt_score}")
            else:
                print("✗ Нет доступных ходов")
                
        elif cmd.startswith("analyze"):
            parts = cmd.split()
            if len(parts) < 2:
                print("✗ Формат: analyze <колонка>")
                continue
            try:
                col = int(parts[1])
                if col not in get_valid_locations(board):
                    print("✗ Недоступная колонка")
                    continue
                row = get_next_open_row(board, col)
                b_copy = board.copy()
                drop_piece(b_copy, row, col, piece_to_analyze)
                
                # Оценка после хода
                score = score_position(b_copy, piece_to_analyze)
                wins = winning_move(b_copy, piece_to_analyze)
                
                print(f"\n📊 Анализ хода в колонку {col}:")
                print(f"   Оценка позиции: {score}")
                print(f"   Немедленная победа: {'✅ ДА!' if wins else '❌ Нет'}")
                
                # Проверка, не подставляет ли ход
                opp = AI if piece_to_analyze == PLAYER else PLAYER
                for next_col in get_valid_locations(b_copy):
                    next_row = get_next_open_row(b_copy, next_col)
                    b_next = b_copy.copy()
                    drop_piece(b_next, next_row, next_col, opp)
                    if winning_move(b_next, opp):
                        print(f"   ⚠️  Внимание: противник может выиграть ответным ходом в колонку {next_col}")
                        break
                        
            except ValueError:
                print("✗ Укажите номер колонки")
                
        elif cmd == "play":
            print("✓ Возврат в обычный режим игры")
            return board, "play"
        elif cmd == "edit":
            print("✓ Возврат в редактор")
            return board, "editor"
        else:
            # Попытка сделать ход
            try:
                col = int(cmd)
                if col not in get_valid_locations(board):
                    print("✗ Недоступная колонка")
                    continue
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, piece_to_analyze)
                print(f"✓ Ход сделан: колонка {col}")
            except ValueError:
                print("✗ Неизвестная команда. Введите 'hint', 'analyze <col>', 'play' или номер колонки")

# ========== УЛУЧШЕННАЯ ПОДСКАЗКА ==========
def get_hint_advanced(board, piece, show_details=True):
    """Расширенная подсказка с анализом"""
    valid_locations = get_valid_locations(board)
    if not valid_locations:
        return None
    
    best_score = -math.inf
    best_col = valid_locations[0]
    analysis = []
    
    for col in valid_locations:
        row = get_next_open_row(board, col)
        b_copy = board.copy()
        drop_piece(b_copy, row, col, piece)
        
        score = score_position(b_copy, piece)
        if winning_move(b_copy, piece):
            score += 100000
        
        # Проверка на опасность
        opp = AI if piece == PLAYER else PLAYER
        danger = False
        for opp_col in get_valid_locations(b_copy):
            opp_row = get_next_open_row(b_copy, opp_col)
            b_opp = b_copy.copy()
            drop_piece(b_opp, opp_row, opp_col, opp)
            if winning_move(b_opp, opp):
                danger = True
                score -= 50  # Штраф за опасный ход
                break
        
        analysis.append((col, score, danger))
        if score > best_score:
            best_score = score
            best_col = col
    
    if show_details:
        print(f"\n📈 Анализ ходов для {'🔴' if piece==PLAYER else '🟡'}:")
        analysis.sort(key=lambda x: x[1], reverse=True)
        for col, score, danger in analysis[:5]:  # Топ-5
            danger_mark = "⚠️" if danger else "✓"
            star = "⭐" if col == best_col else "  "
            print(f"  {star}{danger_mark} Колонка {col}: оценка {score}")
    
    return best_col

# ========== ГЛАВНЫЙ ЦИКЛ ==========
def main():
    board = create_board()
    global game_mode, current_puzzle_piece
    
    print("🎮 CONNECT 4 - Enhanced Edition")
    print("Режимы: play (обычная игра), editor (редактор), puzzle (головоломка)")
    
    while True:
        # Меню выбора режима
        if game_mode == "play":
            print("\n" + "="*40)
            print("🎮 РЕЖИМ ИГРЫ")
            print("Команды: <0-5> - ход, 'h' - подсказка, 'e' - редактор, 'p' - головоломка, 'q' - выход")
            print("="*40)
            print_board(board)
            
            user_input = input("Ваш ход > ").strip().lower()
            
            if user_input == 'q':
                break
            elif user_input == 'e':
                board = editor_mode(board)
                game_mode = "editor"
                continue
            elif user_input == 'p':
                current_puzzle_piece = PLAYER
                board = editor_mode(board)
                game_mode = "puzzle"
                continue
            elif user_input == 'h':
                hint_col = get_hint_advanced(board, PLAYER)
                if hint_col is not None:
                    print(f">>> 💡 ПОДСКАЗКА: Лучше всего сходить в колонку {hint_col}")
                continue
            
            try:
                col = int(user_input)
                if not is_valid_location(board, col):
                    print("✗ Недоступная колонка")
                    continue
                    
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER)
                
                if winning_move(board, PLAYER):
                    print_board(board)
                    print("🎉 Вы победили!")
                    break
                    
                # Ход компьютера
                print("\n🤖 Компьютер думает...")
                col_ai, _ = minimax(board, 5, -math.inf, math.inf, True)
                if col_ai is not None and is_valid_location(board, col_ai):
                    row_ai = get_next_open_row(board, col_ai)
                    drop_piece(board, row_ai, col_ai, AI)
                    print(f"→ Компьютер сходил в колонку {col_ai}")
                    
                    if winning_move(board, AI):
                        print_board(board)
                        print("🤖 Компьютер победил!")
                        break
                        
                print_board(board)
                
            except ValueError:
                print("✗ Введите число 0-5 или команду")
                
        elif game_mode == "editor":
            board = editor_mode(board)
            game_mode = "play"  # После редактора возвращаемся в игру
            
        elif game_mode == "puzzle":
            result = puzzle_mode(board, current_puzzle_piece)
            if isinstance(result, tuple):
                board, next_mode = result
                if next_mode == "play":
                    game_mode = "play"
                elif next_mode == "editor":
                    game_mode = "editor"

if __name__ == "__main__":
    main()
