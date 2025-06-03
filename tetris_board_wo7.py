import pygame
import random
import numpy as np
from time import sleep

# Tetris game class using pygame for rendering
class Tetris:
    '''Tetris game class'''

    # BOARD constants
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    # Tetromino definitions
    TETROMINOS = {
        0: {  # I
            0: np.array([(0,0),(1,0),(2,0),(3,0)]),
            90: np.array([(1,0),(1,1),(1,2),(1,3)]),
            180: np.array([(3,0),(2,0),(1,0),(0,0)]),
            270: np.array([(1,3),(1,2),(1,1),(1,0)]),
        },
        1: {  # T
            0: np.array([(1,0),(0,1),(1,1),(2,1)]),
            90: np.array([(0,1),(1,2),(1,1),(1,0)]),
            180: np.array([(1,2),(2,1),(1,1),(0,1)]),
            270: np.array([(2,1),(1,0),(1,1),(1,2)]),
        },
        2: {  # L
            0: np.array([(1,0),(1,1),(1,2),(2,2)]),
            90: np.array([(0,1),(1,1),(2,1),(2,0)]),
            180: np.array([(1,2),(1,1),(1,0),(0,0)]),
            270: np.array([(2,1),(1,1),(0,1),(0,2)]),
        },
        3: {  # J
            0: np.array([(1,0),(1,1),(1,2),(0,2)]),
            90: np.array([(0,1),(1,1),(2,1),(2,2)]),
            180: np.array([(1,2),(1,1),(1,0),(2,0)]),
            270: np.array([(2,1),(1,1),(0,1),(0,0)]),
        },
        4: {  # Z
            0: np.array([(0,0),(1,0),(1,1),(2,1)]),
            90: np.array([(0,2),(0,1),(1,1),(1,0)]),
            180: np.array([(2,1),(1,1),(1,0),(0,0)]),
            270: np.array([(1,0),(1,1),(0,1),(0,2)]),
        },
        5: {  # S
            0: np.array([(2,0),(1,0),(1,1),(0,1)]),
            90: np.array([(0,0),(0,1),(1,1),(1,2)]),
            180: np.array([(0,1),(1,1),(1,0),(2,0)]),
            270: np.array([(1,2),(1,1),(0,1),(0,0)]),
        },
        6: {  # O
            0: np.array([(1,0),(2,0),(1,1),(2,1)]),
            90: np.array([(1,0),(2,0),(1,1),(2,1)]),
            180: np.array([(1,0),(2,0),(1,1),(2,1)]),
            270: np.array([(1,0),(2,0),(1,1),(2,1)]),
        }
    }

    # 각 회전 상태별로 최소 Y 좌표(깊이)를 저장
    TETROMINOS_MIN_Y = {
        0: {0: np.array([1,1,1,1]), 90: np.array([4]), 180: np.array([1,1,1,1]), 270: np.array([4])},
        1: {0: np.array([2,2,2]), 90: np.array([2,3]), 180: np.array([2,3,2]), 270: np.array([3,2])},
        2: {0: np.array([3,3]), 90: np.array([2,2,2]), 180: np.array([1,3]), 270: np.array([3,2,2])},
        3: {0: np.array([3,3]), 90: np.array([2,2,3]), 180: np.array([3,1]), 270: np.array([2,2,2])},
        4: {0: np.array([1,2,2]), 90: np.array([3,2]), 180: np.array([1,2,2]), 270: np.array([3,2])},
        5: {0: np.array([2,2,1]), 90: np.array([2,3]), 180: np.array([2,2,1]), 270: np.array([2,3])},
        6: {0: np.array([2,2]), 90: np.array([2,2]), 180: np.array([2,2]), 270: np.array([2,2])},
    }

    COLORS = {
        0: (0, 0, 0),    # 빈칸: 검정
        1: (255, 0, 0),  # I 조각: 빨간
        2: (0, 255, 0),  # T 조각: 초록
        3: (255, 0, 255),# L 조각: 자홍
        4: (255, 255, 0),# J 조각: 노랑
        5: (255, 128, 0),# Z 조각: 주황
        6: (0, 255, 255),# S 조각: 시안
        7: (0, 0, 255),  # O 조각: 파랑
        9: (255, 255, 255),# 텍스트용 흰색
    }

    def __init__(self):
        self.reset()

    def reset(self):
        # 보드를 초기화
        self.board = np.zeros((Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH), dtype=int)
        self.column_heights = np.zeros(Tetris.BOARD_WIDTH, dtype=int)
        self.game_over = False

        # 7-bag 대신 단순 랜덤으로 조각 뽑기
        self.current_piece = random.randint(0, len(Tetris.TETROMINOS) - 1)
        self.current_rotation = 0
        self.current_pos = [3, 0]  # 스폰 위치: x=3, y=0
        self.next_piece = random.randint(0, len(Tetris.TETROMINOS) - 1)

        self.score = 0
        # 현재 스폰된 블록을 포함한 전체 보드 반환
        return self._get_complete_board()

    def _get_rotated_piece(self):
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        """
        현재 보드(고정 블록) 위에, 낙하 중인 블록(piece_id+1)을 오버레이하여 (20×10) 배열로 반환
        """
        piece_coords = self._get_rotated_piece() + self.current_pos  # shape (4,2)
        complete = self.board.copy()
        for (px, py) in piece_coords:
            if 0 <= py < Tetris.BOARD_HEIGHT and 0 <= px < Tetris.BOARD_WIDTH:
                complete[py, px] = self.current_piece + 1
        return complete

    def get_game_score(self):
        return self.score

    def _new_round(self):
        """
        매 라운드마다 호출: next_piece → current_piece 로 옮기고,
        다음 next_piece 는 단순 랜덤으로 선택
        """
        # 다음 조각을 현재 조각으로 설정
        self.current_piece = self.next_piece
        # 새로운 next_piece 을 랜덤으로 선택
        self.next_piece = random.randint(0, len(Tetris.TETROMINOS) - 1)

        # 회전 초기화 및 스폰 위치 설정
        self.current_rotation = 0
        self.current_pos = [3, 0]

        # 스폰 시점에 충돌 발생하면 게임 오버
        piece = self._get_rotated_piece()
        min_x = piece[:, 0].min()
        max_x = piece[:, 0].max()
        min_y = Tetris.TETROMINOS_MIN_Y[self.current_piece][0]  # 회전 상태 0
        slice_heights = self.column_heights[self.current_pos[0] + min_x : self.current_pos[0] + max_x + 1]
        if (Tetris.BOARD_HEIGHT - (slice_heights + min_y).max()) < 0:
            self.game_over = True

    def _check_collision(self, piece, pos):
        for x, y in piece:
            xi = x + pos[0]
            yi = y + pos[1]
            if xi < 0 or xi >= Tetris.BOARD_WIDTH or yi < 0 or yi >= Tetris.BOARD_HEIGHT:
                return True
            if self.board[yi][xi] > 0:
                return True
        return False

    def _add_piece_to_board(self, piece, pos):
        """
        현재 조각을 보드에 고정(piece_id+1)하고, 새로운 보드 상태 반환
        """
        new_board = self.board.copy()
        for x, y in piece:
            xi = x + pos[0]
            yi = y + pos[1]
            new_board[yi, xi] = self.current_piece + 1
        return new_board

    def _clear_lines(self, board):
        """
        꽉 찬 줄을 지우고 위의 블록을 아래로 내림
        """
        filled = (board > 0).sum(axis=1) == Tetris.BOARD_WIDTH
        lines_to_keep = np.where(filled == False)[0]
        num_cleared = Tetris.BOARD_HEIGHT - len(lines_to_keep)
        new_board = np.zeros_like(board)
        new_board[-len(lines_to_keep):] = board[lines_to_keep]
        return num_cleared, new_board

    def _number_of_holes(self, board, column_heights):
        return sum(column_heights) - (board > 0).sum()

    def _bumpiness(self, board, column_heights):
        diffs = np.abs(column_heights[:-1] - column_heights[1:])
        return sum(diffs), max(diffs)

    def _height(self, board, column_heights):
        return sum(column_heights), max(column_heights), min(column_heights)

    def _column_heights(self, board):
        return np.sum(np.cumsum(board > 0, axis=0) >= 1, axis=0)

    def get_state_size(self):
        # 20×10 = 200 반환
        return Tetris.BOARD_HEIGHT * Tetris.BOARD_WIDTH

    def set_board(self, board):
        self.board = board
        self.column_heights = self._column_heights(board)

    def get_next_states(self):
        """
        가능한 모든 (x, rotation) 조합에 대해,
        보드를 고정한 뒤의 (20×10) 배열 반환
        """
        states = {}
        pid = self.current_piece
        cols = self.column_heights

        # 회전 가능한 목록 설정
        if pid == 6:
            rotations = [0]
        elif pid in (0, 4, 5):
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        for rot in rotations:
            piece = Tetris.TETROMINOS[pid][rot]
            min_x = piece[:, 0].min()
            max_x = piece[:, 0].max()
            min_y = Tetris.TETROMINOS_MIN_Y[pid][rot]
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                new_heights = cols.copy()
                new_heights[x + min_x : x + max_x + 1] += min_y
                drop_y = Tetris.BOARD_HEIGHT - new_heights.max()
                if drop_y >= 0:
                    board_after = self._add_piece_to_board(piece, [x, drop_y])
                    num_cleared, board_cleared = self._clear_lines(board_after)

                    new_heights2 = self._column_heights(board_cleared)
                    states[(x, rot)] = board_cleared.copy()

        return states

    def play(self, x, rotation, render=True, screen=None, scale=25, clock=None):
        """
        x: 떨어뜨릴 컬럼 인덱스 (왼쪽 끝 기준)
        rotation: 0, 90, 180, 270
        render: 애니메이션 여부
        screen, scale, clock: pygame 렌더용
        """
        self.current_rotation = rotation
        piece = self._get_rotated_piece()
        min_x = piece[:, 0].min()
        max_x = piece[:, 0].max()
        min_y = Tetris.TETROMINOS_MIN_Y[self.current_piece][rotation]
        col_slice = self.column_heights[x + min_x : x + max_x + 1]
        drop_y = Tetris.BOARD_HEIGHT - (col_slice + min_y).max()

        # 애니메이션: 한 칸씩 내려가는 모습
        if render and screen is not None and clock is not None:
            for y_anim in range(0, drop_y + 1):
                self.current_pos = [x, y_anim]
                screen.fill((255, 255, 255))
                self.render(scale, screen)
                pygame.display.flip()
                clock.tick(60)

        # 최종 위치에 고정
        self.current_pos = [x, drop_y]
        new_board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, board_cleared = self._clear_lines(new_board)
        self.set_board(board_cleared)

        gained = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self.score += gained

        # 다음 라운드 준비
        self._new_round()
        if self.game_over:
            gained -= 5  # 게임 오버 시 패널티

        # play()가 더 이상 4개 특성이 아니라 '전체 보드 행렬'을 반환
        return gained, self.game_over, self._get_complete_board()

    def render(self, scale, screen):
        """
        Pygame으로 실제 화면을 그리는 부분
        """
        board_to_draw = self._get_complete_board()
        for y in range(Tetris.BOARD_HEIGHT):
            for x in range(Tetris.BOARD_WIDTH):
                val = board_to_draw[y, x]
                color = Tetris.COLORS[val]
                pygame.draw.rect(screen, color, (x * scale, y * scale, scale, scale))

        # 점수 출력
        font = pygame.font.SysFont(None, 24)
        txt = font.render(f"Score: {self.score}", True, Tetris.COLORS[9])
        screen.blit(txt, (5, 5))


def main():
    # ------------------ pygame 초기화 및 화면 생성 ------------------
    pygame.init()
    scale = 30
    screen = pygame.display.set_mode((Tetris.BOARD_WIDTH * scale,
                                      Tetris.BOARD_HEIGHT * scale))
    pygame.display.set_caption("Tetris Colored Blocks")
    clock = pygame.time.Clock()

    game = Tetris()
    running = True

    # 예시: 자동 랜덤 드롭 루프
    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        states = game.get_next_states()
        (x, rot), board_next = random.choice(list(states.items()))

        # play() 호출 시 render=True, 화면 관련 인자 넘김
        gained, done, board_now = game.play(x, rot, render=True,
                                            screen=screen, scale=scale, clock=clock)

    pygame.quit()


if __name__ == "__main__":
    main()
