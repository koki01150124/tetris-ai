import pygame
import random
import time

# 初期化
pygame.init()

# 定数
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
BOARD_X = 50
BOARD_Y = 50
# 2プレイヤー用に画面幅を調整
SCREEN_WIDTH = (BOARD_WIDTH * CELL_SIZE + 250) * 2
SCREEN_HEIGHT = BOARD_HEIGHT * CELL_SIZE + 100

# 色定義 (モダンなテーマ)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
GRAY = (128, 128, 128)

BACKGROUND_COLOR = (26, 26, 26)
BOARD_COLOR = (42, 42, 42)
GRID_COLOR = (68, 68, 68)
TEXT_COLOR = (240, 240, 240)

# テトリミノの形状定義
TETROMINOS = {
    'I': [['.....',
           '..#..',
           '..#..',
           '..#..',
           '..#..']],
    'O': [['.....',
           '.....',
           '.##..',
           '.##..',
           '.....']],
    'T': [['.....',
           '.....',
           '.#...',
           '###..',
           '.....'],
          ['.....',
           '.....',
           '.#...',
           '.##..',
           '.#...'],
          ['.....',
           '.....',
           '.....',
           '###..',
           '.#...'],
          ['.....',
           '.....',
           '.#...',
           '##...',
           '.#...']],
    'S': [['.....',
           '.....',
           '.##..',
           '##...',
           '.....'],
          ['.....',
           '.#...',
           '.##..',
           '..#..',
           '.....']],
    'Z': [['.....',
           '.....',
           '##...',
           '.##..',
           '.....'],
          ['.....',
           '..#..',
           '.##..',
           '.#...',
           '.....']],
    'J': [['.....',
           '.#...',
           '.#...',
           '##...',
           '.....'],
          ['.....',
           '.....',
           '#....',
           '###..',
           '.....'],
          ['.....',
           '.##..',
           '.#...',
           '.#...',
           '.....'],
          ['.....',
           '.....',
           '###..',
           '..#..',
           '.....']],
    'L': [['.....',
           '..#..',
           '..#..',
           '.##..',
           '.....'],
          ['.....',
           '.....',
           '###..',
           '#....',
           '.....'],
          ['.....',
           '##...',
           '.#...',
           '.#...',
           '.....'],
          ['.....',
           '.....',
           '..#..',
           '###..',
           '.....']]
}

TETROMINO_COLORS = {
    'I': CYAN,
    'O': YELLOW,
    'T': PURPLE,
    'S': GREEN,
    'Z': RED,
    'J': BLUE,
    'L': ORANGE,
    'G': GRAY  # おじゃまライン用の色
}

class Tetromino:
    def __init__(self, shape_type):
        self.shape_type = shape_type
        self.shape = TETROMINOS[shape_type]
        self.color = TETROMINO_COLORS[shape_type]
        self.x = BOARD_WIDTH // 2 - 2
        self.y = 0
        self.rotation = 0

    def get_rotated_shape(self, rotation=None):
        if rotation is None:
            rotation = self.rotation
        return self.shape[rotation % len(self.shape)]

class TetrisGame:
    def __init__(self, player_id):
        self.player_id = player_id
        self.board = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.garbage_to_receive = 0
        self.spawn_new_piece()

    def add_garbage(self, count):
        self.garbage_to_receive += count

    def receive_garbage(self):
        if self.garbage_to_receive == 0:
            return

        hole_x = random.randint(0, BOARD_WIDTH - 1)
        garbage_line = ['G'] * BOARD_WIDTH
        garbage_line[hole_x] = 0

        for _ in range(self.garbage_to_receive):
            self.board.pop(0)
            self.board.append(list(garbage_line))

        self.garbage_to_receive = 0

    def spawn_new_piece(self):
        if self.next_piece is None:
            self.next_piece = Tetromino(random.choice(list(TETROMINOS.keys())))
        
        self.current_piece = self.next_piece
        self.next_piece = Tetromino(random.choice(list(TETROMINOS.keys())))
        
        if not self.is_valid_position(self.current_piece, self.current_piece.x, self.current_piece.y):
            self.game_over = True

    def is_valid_position(self, piece, x, y, rotation=None):
        shape = piece.get_rotated_shape(rotation)
        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell == '#':
                    nx, ny = x + px, y + py
                    if nx < 0 or nx >= BOARD_WIDTH or ny >= BOARD_HEIGHT:
                        return False
                    if ny >= 0 and self.board[ny][nx]:
                        return False
        return True

    def place_piece(self, piece):
        shape = piece.get_rotated_shape()
        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell == '#':
                    nx, ny = piece.x + px, piece.y + py
                    if ny >= 0:
                        self.board[ny][nx] = piece.shape_type

    def clear_lines(self):
        lines_to_clear = []
        for y in range(BOARD_HEIGHT):
            if all(self.board[y]):
                lines_to_clear.append(y)

        for y in lines_to_clear:
            del self.board[y]
            self.board.insert(0, [0 for _ in range(BOARD_WIDTH)])

        lines_cleared = len(lines_to_clear)
        self.lines_cleared += lines_cleared
        self.score += lines_cleared ** 2 * 100

        return lines_cleared

    def move_piece(self, dx, dy):
        if self.current_piece and self.is_valid_position(self.current_piece, 
                                                       self.current_piece.x + dx, 
                                                       self.current_piece.y + dy):
            self.current_piece.x += dx
            self.current_piece.y += dy
            return True
        return False

    def rotate_piece(self, direction=1):
        if self.current_piece:
            new_rotation = (self.current_piece.rotation + direction) % len(self.current_piece.shape)
            if self.is_valid_position(self.current_piece, 
                                    self.current_piece.x, 
                                    self.current_piece.y, 
                                    new_rotation):
                self.current_piece.rotation = new_rotation
                return True
        return False

    def get_ghost_y(self):
        if not self.current_piece:
            return 0
        
        ghost_y = self.current_piece.y
        while self.is_valid_position(self.current_piece, self.current_piece.x, ghost_y + 1):
            ghost_y += 1
        return ghost_y

class TetrisAI:
    def __init__(self, god_mode=False):
        self.god_mode = god_mode
        self.weights_normal = {
            'completed_lines': 5000,
            'holes': -100000,
            'bumpiness': -2500,
            'aggregate_height': -5000,
            'max_height': -10000,
            'wells': -20000,
            'row_transitions': -2500,
            'col_transitions': -2500,
            't_spin': 1000000
        }
        self.weights_crisis = {
            'completed_lines': 10000,
            'holes': -200000,
            'bumpiness': -10000,
            'aggregate_height': -20000,
            'max_height': -40000,
            'wells': -40000,
            'row_transitions': -5000,
            'col_transitions': -5000,
            't_spin': -10000000
        }
        self.weights_god = {
            'completed_lines': 500000,
            'holes': -10000000,
            'bumpiness': -50000,
            'aggregate_height': -100000,
            'max_height': -200000,
            'wells': -500000,
            'row_transitions': -30000,
            'col_transitions': -30000,
            't_spin': 5000000
        }
        self.crisis_threshold = BOARD_HEIGHT / 2

    def get_current_weights(self, board):
        if self.god_mode:
            return self.weights_god

        heights = [0] * BOARD_WIDTH
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if board[y][x]:
                    heights[x] = BOARD_HEIGHT - y
                    break
        
        avg_height = sum(heights) / BOARD_WIDTH if BOARD_WIDTH > 0 else 0
        if avg_height > self.crisis_threshold:
            return self.weights_crisis
        return self.weights_normal

    def evaluate_board(self, board, lines_cleared, t_spin_reward, weights):
        heights = [0] * BOARD_WIDTH
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if board[y][x]:
                    heights[x] = BOARD_HEIGHT - y
                    break

        aggregate_height = sum(heights)
        max_height = max(heights) if heights else 0
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))

        holes = 0
        for x in range(BOARD_WIDTH):
            col_has_block = False
            for y in range(BOARD_HEIGHT):
                if board[y][x]:
                    col_has_block = True
                elif col_has_block and not board[y][x]:
                    holes += 1

        wells = 0
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if not board[y][x]:
                    left_wall = (x == 0) or (board[y][x-1])
                    right_wall = (x == BOARD_WIDTH - 1) or (board[y][x+1])
                    if left_wall and right_wall:
                        depth = 0
                        for y2 in range(y, BOARD_HEIGHT):
                            if not board[y2][x]:
                                depth += 1
                            else:
                                break
                        wells += depth

        row_transitions = 0
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH - 1):
                if (board[y][x] == 0 and board[y][x+1] != 0) or \
                   (board[y][x] != 0 and board[y][x+1] == 0):
                    row_transitions += 1

        col_transitions = 0
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT - 1):
                if (board[y][x] == 0 and board[y+1][x] != 0):
                    col_transitions += 1

        score = 0
        score += weights['completed_lines'] * (lines_cleared ** 2)
        score += weights['aggregate_height'] * aggregate_height
        score += weights['max_height'] * max_height
        score += weights['bumpiness'] * bumpiness
        score += weights['holes'] * holes
        score += weights['wells'] * wells
        score += weights['row_transitions'] * row_transitions
        score += weights['col_transitions'] * col_transitions
        score += weights['t_spin'] * t_spin_reward
        return score

    def is_t_spin(self, board, piece, x, y, rotation):
        if piece.shape_type != 'T':
            return False

        # Tスピンの判定には、Tミノの中心周りの4つの角のうち3つ以上が埋まっている必要がある
        # Tミノの形状データにおける中心はインデックス(2,1)あたりにあると仮定
        # 正確な中心は形状と回転に依存する
        # ここでは簡略化のため、ピースのバウンディングボックスの中心を基準にする
        center_x, center_y = x + 2, y + 1 # Tミノの回転軸に近い点を中心とする

        corners = [(center_y - 1, center_x - 1), (center_y - 1, center_x + 1), 
                   (center_y + 1, center_x - 1), (center_y + 1, center_x + 1)]
        
        filled_corners = 0
        for r, c in corners:
            # 盤面外は壁とみなす
            if not (0 <= r < BOARD_HEIGHT and 0 <= c < BOARD_WIDTH) or board[r][c]:
                filled_corners += 1
        
        # 3つ以上の角が埋まっているか、または特定の難しいTスピンの配置（例：壁際のTスピン）
        # ここでは基本的な3コーナーのルールのみを実装
        return filled_corners >= 3

    def simulate_drop(self, board, piece, x, rotation):
        temp_piece = Tetromino(piece.shape_type)
        temp_piece.rotation = rotation
        temp_piece.x = x
        temp_piece.y = 0

        # ピースが初期位置で既に無効な場合はNoneを返す
        if not self.is_valid_position_on_board(board, temp_piece, temp_piece.x, 0):
            return None, -1, 0

        # ピースを一番下まで落とす
        last_valid_y = 0
        while self.is_valid_position_on_board(board, temp_piece, temp_piece.x, temp_piece.y + 1):
            temp_piece.y += 1
        last_valid_y = temp_piece.y

        # 盤面にピースを配置
        temp_board = [row[:] for row in board]
        shape = temp_piece.get_rotated_shape()
        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell == '#':
                    nx, ny = temp_piece.x + px, temp_piece.y + py
                    if 0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT:
                        temp_board[ny][nx] = temp_piece.shape_type

        # Tスピンの報酬を計算
        t_spin_reward = 0
        if self.is_t_spin(board, temp_piece, temp_piece.x, last_valid_y, rotation):
            t_spin_reward = 1

        # ラインクリアを計算
        lines_cleared = 0
        y = BOARD_HEIGHT - 1
        while y >= 0:
            if all(temp_board[y]):
                del temp_board[y]
                temp_board.insert(0, [0 for _ in range(BOARD_WIDTH)])
                lines_cleared += 1
            else:
                y -= 1
        
        return temp_board, lines_cleared, t_spin_reward

    def is_valid_position_on_board(self, board, piece, x, y, rotation=None):
        shape = piece.get_rotated_shape(rotation)
        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell == '#':
                    nx, ny = x + px, y + py
                    if nx < 0 or nx >= BOARD_WIDTH or ny >= BOARD_HEIGHT:
                        return False
                    if ny >= 0 and board[ny][nx]:
                        return False
        return True

    def get_possible_moves(self, board, piece):
        possible_moves = []
        for rotation in range(len(piece.shape)):
            for x in range(-2, BOARD_WIDTH + 2):
                # ピースをそのxとrotationで落とせるかチェック
                temp_piece = Tetromino(piece.shape_type)
                temp_piece.rotation = rotation
                temp_piece.x = x
                temp_piece.y = 0
                
                # 落下開始地点で有効かチェック
                if self.is_valid_position_on_board(board, temp_piece, x, 0):
                    # 実際に落下させてみる
                    y = 0
                    while self.is_valid_position_on_board(board, temp_piece, x, y + 1):
                        y += 1
                    possible_moves.append((x, rotation))

        return list(set(possible_moves)) # 重複を除外

    def get_best_move(self, game):
        if not game.current_piece or not game.next_piece:
            return None

        best_score = float('-inf')
        best_move = None

        weights = self.get_current_weights(game.board)
        current_piece_moves = self.get_possible_moves(game.board, game.current_piece)

        if not current_piece_moves:
            return None

        for move in current_piece_moves:
            x, rotation = move
            
            board_after_first_move, lines_cleared1, t_spin1 = self.simulate_drop(game.board, game.current_piece, x, rotation)
            if board_after_first_move is None:
                continue

            # ゲームオーバーになるような手は避ける
            if any(board_after_first_move[0]):
                continue

            score1 = self.evaluate_board(board_after_first_move, lines_cleared1, t_spin1, weights)

            best_future_score = float('-inf')
            
            next_weights = self.get_current_weights(board_after_first_move)
            next_piece_moves = self.get_possible_moves(board_after_first_move, game.next_piece)
            
            if not next_piece_moves:
                # 次のピースが置けない場合（非常に稀）、現在の手のスコアのみで評価
                final_score = score1
            else:
                for next_move in next_piece_moves:
                    next_x, next_rotation = next_move
                    
                    board_after_second_move, lines_cleared2, t_spin2 = self.simulate_drop(board_after_first_move, game.next_piece, next_x, next_rotation)
                    if board_after_second_move is None:
                        continue

                    score2 = self.evaluate_board(board_after_second_move, lines_cleared2, t_spin2, next_weights)

                    if score2 > best_future_score:
                        best_future_score = score2
                
                if best_future_score == float('-inf'):
                    # 次の有効な手が一つも見つからなかった場合
                    final_score = score1
                else:
                    # 現在の手のスコアと、次の手の最善スコアを組み合わせて評価
                    final_score = score1 + best_future_score * 0.8 # 将来のスコアを少し割り引く

            if final_score > best_score:
                best_score = final_score
                best_move = move

        # 最善手が見つからなかった場合、とりあえず最初に見つかった有効な手を返す
        if best_move is None:
            return current_piece_moves[0]

        return best_move

def draw_block(surface, color, rect):
    """Draws a tetris block with a 3D-ish effect."""
    shadow_color = (max(0, color[0] - 60), max(0, color[1] - 60), max(0, color[2] - 60))
    highlight_color = (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60))

    pygame.draw.rect(surface, color, rect)
    pygame.draw.rect(surface, shadow_color, rect, 2)
    pygame.draw.line(surface, highlight_color, rect.topleft, (rect.right - 2, rect.top))
    pygame.draw.line(surface, highlight_color, rect.topleft, (rect.left, rect.bottom - 2))

def draw_game(screen, game, fonts, x_offset, player_name):
    board_x = BOARD_X + x_offset
    board_y = BOARD_Y

    # Draw player name
    player_text = fonts['title'].render(player_name, True, TEXT_COLOR)
    screen.blit(player_text, (board_x, board_y - 40))

    # Draw info panel background
    panel_rect = pygame.Rect(board_x + BOARD_WIDTH * CELL_SIZE + 10, board_y, 200, BOARD_HEIGHT * CELL_SIZE)
    pygame.draw.rect(screen, BACKGROUND_COLOR, panel_rect)

    # Draw board background
    board_bg_rect = pygame.Rect(board_x, board_y, BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE)
    pygame.draw.rect(screen, BOARD_COLOR, board_bg_rect)

    # Draw grid
    for x in range(BOARD_WIDTH + 1):
        pygame.draw.line(screen, GRID_COLOR, (board_x + x * CELL_SIZE, board_y), (board_x + x * CELL_SIZE, board_y + BOARD_HEIGHT * CELL_SIZE))
    for y in range(BOARD_HEIGHT + 1):
        pygame.draw.line(screen, GRID_COLOR, (board_x, board_y + y * CELL_SIZE), (board_x + BOARD_WIDTH * CELL_SIZE, board_y + y * CELL_SIZE))

    # Draw placed blocks
    for y in range(BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if game.board[y][x]:
                color = TETROMINO_COLORS.get(game.board[y][x], GRAY)
                rect = pygame.Rect(board_x + x * CELL_SIZE, board_y + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                draw_block(screen, color, rect)

    # Draw current piece and ghost
    if game.current_piece:
        ghost_y = game.get_ghost_y()
        shape = game.current_piece.get_rotated_shape()
        ghost_color = (80, 80, 80)
        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell == '#':
                    rect = pygame.Rect(board_x + (game.current_piece.x + px) * CELL_SIZE, board_y + (ghost_y + py) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, ghost_color, rect, 2)

        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell == '#':
                    rect = pygame.Rect(board_x + (game.current_piece.x + px) * CELL_SIZE, board_y + (game.current_piece.y + py) * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    draw_block(screen, game.current_piece.color, rect)

    # Draw info panel content
    info_x = board_x + BOARD_WIDTH * CELL_SIZE + 50
    
    title_score = fonts['title'].render("SCORE", True, TEXT_COLOR)
    screen.blit(title_score, (info_x, board_y + 30))
    text_score = fonts['main'].render(str(game.score), True, WHITE)
    screen.blit(text_score, (info_x, board_y + 70))

    title_lines = fonts['title'].render("LINES", True, TEXT_COLOR)
    screen.blit(title_lines, (info_x, board_y + 130))
    text_lines = fonts['main'].render(str(game.lines_cleared), True, WHITE)
    screen.blit(text_lines, (info_x, board_y + 170))

    title_next = fonts['title'].render("NEXT", True, TEXT_COLOR)
    screen.blit(title_next, (info_x, board_y + 230))
    if game.next_piece:
        shape = game.next_piece.get_rotated_shape()
        for py, row in enumerate(shape):
            for px, cell in enumerate(row):
                if cell == '#':
                    rect = pygame.Rect(info_x + 20 + px * (CELL_SIZE - 5), board_y + 280 + py * (CELL_SIZE - 5), CELL_SIZE - 5, CELL_SIZE - 5)
                    draw_block(screen, game.next_piece.color, rect)

    if game.game_over:
        overlay = pygame.Surface((BOARD_WIDTH * CELL_SIZE, BOARD_HEIGHT * CELL_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (board_x, board_y))
        
        game_over_text = fonts['game_over'].render("LOSE", True, RED)
        text_rect = game_over_text.get_rect(center=(board_x + BOARD_WIDTH * CELL_SIZE / 2, board_y + BOARD_HEIGHT * CELL_SIZE / 2))
        screen.blit(game_over_text, text_rect)

def draw_menu(screen, fonts):
    screen.fill(BACKGROUND_COLOR)
    width = screen.get_width()
    height = screen.get_height()

    title_text = fonts['game_over'].render("TETRIS BATTLE", True, YELLOW)
    title_rect = title_text.get_rect(center=(width / 2, height / 4))
    screen.blit(title_text, title_rect)

    # AI vs Human button
    pvai_button = pygame.Rect(width / 2 - 150, height / 2 - 50, 300, 60)
    pygame.draw.rect(screen, GRAY, pvai_button)
    pvai_text = fonts['title'].render("Human vs AI", True, TEXT_COLOR)
    pvai_text_rect = pvai_text.get_rect(center=pvai_button.center)
    screen.blit(pvai_text, pvai_text_rect)

    # AI vs AI button
    aivai_button = pygame.Rect(width / 2 - 150, height / 2 + 50, 300, 60)
    pygame.draw.rect(screen, GRAY, aivai_button)
    aivai_text = fonts['title'].render("AI vs AI", True, TEXT_COLOR)
    aivai_text_rect = aivai_text.get_rect(center=aivai_button.center)
    screen.blit(aivai_text, aivai_text_rect)
    
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if pvai_button.collidepoint(mouse_pos):
                    return 'human_vs_ai'
                if aivai_button.collidepoint(mouse_pos):
                    return 'ai_vs_ai'

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris AI Battle")
    clock = pygame.time.Clock()
    
    try:
        font_main = pygame.font.SysFont('consolas', 24)
        font_title = pygame.font.SysFont('consolas', 30, bold=True)
        font_game_over = pygame.font.SysFont('consolas', 50, bold=True)
    except pygame.error:
        font_main = pygame.font.Font(None, 24)
        font_title = pygame.font.Font(None, 36)
        font_game_over = pygame.font.Font(None, 60)

    fonts = {'main': font_main, 'title': font_title, 'game_over': font_game_over}
    
    game_mode = draw_menu(screen, fonts)
    if not game_mode:
        pygame.quit()
        return

    game1 = TetrisGame(1)
    game2 = TetrisGame(2)
    
    player1_name = ""
    player2_name = ""
    player2_is_ai = False

    ai1 = TetrisAI(god_mode=True)
    ai2 = None

    if game_mode == 'human_vs_ai':
        player1_name = "AI"
        player2_name = "PLAYER"
    elif game_mode == 'ai_vs_ai':
        ai2 = TetrisAI(god_mode=True)
        player1_name = "AI 1"
        player2_name = "AI 2"
        player2_is_ai = True

    running = True
    winner = None
    
    last_ai1_move_time = time.time()
    last_ai2_move_time = time.time()
    ai_move_delay = 0.1

    player_fall_time = 0
    fall_speed = 0.5  # seconds

    while running:
        dt = clock.tick(60)
        if not player2_is_ai:
            player_fall_time += dt / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    main() # Restart the whole game
                    return
                
                if not player2_is_ai and not game2.game_over and winner is None:
                    if event.key == pygame.K_LEFT:
                        game2.move_piece(-1, 0)
                    elif event.key == pygame.K_RIGHT:
                        game2.move_piece(1, 0)
                    elif event.key == pygame.K_DOWN:
                        if game2.move_piece(0, 1):
                            player_fall_time = 0
                    elif event.key == pygame.K_UP:
                        game2.rotate_piece()
                    elif event.key == pygame.K_SPACE:
                        while game2.move_piece(0, 1):
                            pass
                        game2.place_piece(game2.current_piece)
                        lines_cleared = game2.clear_lines()
                        if lines_cleared >= 2:
                            garbage_map = {2: 1, 3: 2, 4: 4}
                            game1.add_garbage(garbage_map.get(lines_cleared, 0))
                        game2.receive_garbage()
                        game2.spawn_new_piece()
                        player_fall_time = 0

        current_time = time.time()
        
        # AI 1 move
        if winner is None and current_time - last_ai1_move_time > ai_move_delay:
            if not game1.game_over:
                best_move = ai1.get_best_move(game1)
                if best_move:
                    game1.current_piece.x, game1.current_piece.rotation = best_move
                    while game1.move_piece(0, 1): pass
                    game1.place_piece(game1.current_piece)
                    lines_cleared = game1.clear_lines()
                    if lines_cleared >= 2:
                        garbage_map = {2: 1, 3: 2, 4: 4}
                        game2.add_garbage(garbage_map.get(lines_cleared, 0))
                    game1.receive_garbage()
                    game1.spawn_new_piece()
            last_ai1_move_time = current_time

        # AI 2 move
        if player2_is_ai and winner is None and current_time - last_ai2_move_time > ai_move_delay:
            if not game2.game_over:
                best_move = ai2.get_best_move(game2)
                if best_move:
                    game2.current_piece.x, game2.current_piece.rotation = best_move
                    while game2.move_piece(0, 1): pass
                    game2.place_piece(game2.current_piece)
                    lines_cleared = game2.clear_lines()
                    if lines_cleared >= 2:
                        garbage_map = {2: 1, 3: 2, 4: 4}
                        game1.add_garbage(garbage_map.get(lines_cleared, 0))
                    game2.receive_garbage()
                    game2.spawn_new_piece()
            last_ai2_move_time = current_time

        # Player 2 (human) piece fall
        if not player2_is_ai and not game2.game_over and winner is None and player_fall_time > fall_speed:
            if not game2.move_piece(0, 1):
                game2.place_piece(game2.current_piece)
                lines_cleared = game2.clear_lines()
                if lines_cleared >= 2:
                    garbage_map = {2: 1, 3: 2, 4: 4}
                    game1.add_garbage(garbage_map.get(lines_cleared, 0))
                game2.receive_garbage()
                game2.spawn_new_piece()
            player_fall_time = 0

        if winner is None:
            if game1.game_over:
                winner = player2_name
            elif game2.game_over:
                winner = player1_name
        
        screen.fill(BACKGROUND_COLOR)
        draw_game(screen, game1, fonts, 0, player1_name)
        draw_game(screen, game2, fonts, (BOARD_WIDTH * CELL_SIZE + 250), player2_name)

        if winner:
            winner_text = fonts['game_over'].render(f"{winner} WINS!", True, YELLOW)
            text_rect = winner_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 30))
            screen.blit(winner_text, text_rect)

            restart_text = fonts['main'].render("Press R to Restart", True, WHITE)
            text_rect = restart_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 30))
            screen.blit(restart_text, text_rect)

        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()
