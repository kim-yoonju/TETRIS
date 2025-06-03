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
        0: (0, 0, 0),      # 빈칸: 검정
        1: (255, 0, 0),    # I 조각: 빨간
        2: (0, 255, 0),    # T 조각: 초록
        3: (255, 0, 255),  # L 조각: 자홍
        4: (255, 255, 0),  # J 조각: 노랑
        5: (255, 128, 0),  # Z 조각: 주황
        6: (0, 255, 255),  # S 조각: 시안
        7: (0, 0, 255),    # O 조각: 파랑
        9: (255, 255, 255) # 텍스트용 흰색
    }

    def __init__(self):
        self.reset()

    def reset(self):
        # board[y][x] = 0(빈칸) or 1~7(고정된 블록: piece_id+1)
        self.board = np.zeros((Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH), dtype=int)
        self.column_heights = np.zeros(Tetris.BOARD_WIDTH, dtype=int)
        self.game_over = False

        # bag 기법으로 랜덤하게 모든 조각이 한 번씩 나오도록
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()

        self._new_round()
        self.score = 0

        # reset()은 이제 전체 보드를 반환
        return self._get_complete_board()

    def _get_rotated_piece(self):
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        """
        현재 보드(고정 블록) 위에, 낙하 중인 블록을 piece_id+1로 오버레이하여
        (20×10) 배열 형태로 반환.
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
        # bag이 비면 리필 후 셔플
        if not self.bag:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]  # x=3, y=0에서 스폰
        self.current_rotation = 0

        # 스폰 시점에 충돌 발생하면 게임 오버 처리
        min_x = self._get_rotated_piece()[:,0].min()
        max_x = self._get_rotated_piece()[:,0].max()
        min_y = Tetris.TETROMINOS_MIN_Y[self.current_piece][0]
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
        고정 블록을 board에 기록. piece_id+1 값을 사용.
        """
        new_board = self.board.copy()
        for x, y in piece:
            xi = x + pos[0]
            yi = y + pos[1]
            new_board[yi, xi] = self.current_piece + 1
        return new_board

    def _clear_lines(self, board):
        """
        꽉 찬 줄을 지우고 위의 블록을 아래로 내린다.
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

    # get_state_size() → 20×10 = 200 반환
    def get_state_size(self):
        return Tetris.BOARD_HEIGHT * Tetris.BOARD_WIDTH

    def set_board(self, board):
        self.board = board
        self.column_heights = self._column_heights(board)

    def get_next_states(self):
        """
        가능한 모든 (x, rotation) 조합에 대해,
        그 결과 생기는 상태를 “전체 보드 행렬” (20×10) 형태로 반환.
        """
        states = {}
        pid = self.current_piece
        cols = self.column_heights

        # 회전 가능한 목록 설정
        if pid == 6:
            rotations = [0]
        elif pid in (0,4,5):
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        for rot in rotations:
            piece = Tetris.TETROMINOS[pid][rot]
            min_x = piece[:,0].min()
            max_x = piece[:,0].max()
            min_y = Tetris.TETROMINOS_MIN_Y[pid][rot]
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                new_heights = cols.copy()
                new_heights[x + min_x : x + max_x + 1] += min_y
                drop_y = Tetris.BOARD_HEIGHT - new_heights.max()
                if drop_y >= 0:
                    # 해당 위치에 조각을 고정한 뒤의 보드를 구함
                    board_after = self._add_piece_to_board(piece, [x, drop_y])
                    num_cleared, board_cleared = self._clear_lines(board_after)

                    # 줄이 지워지면 높이도 업데이트해야 함
                    new_heights2 = self._column_heights(board_cleared)

                    # 반환값을 전체 보드 행렬(20×10)로 대체
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
        min_x = piece[:,0].min()
        max_x = piece[:,0].max()
        min_y = Tetris.TETROMINOS_MIN_Y[self.current_piece][rotation]
        col_slice = self.column_heights[x + min_x : x + max_x + 1]
        drop_y = Tetris.BOARD_HEIGHT - (col_slice + min_y).max()

        # 애니메이션: 한 칸씩 내려가는 모습
        if render and screen is not None and clock is not None:
            for y_anim in range(0, drop_y + 1):
                self.current_pos = [x, y_anim]
                screen.fill((255, 255, 255))
                self.render(scale, screen)
                # 다음 블록 미리보기도 함께 그림
                preview_offset_x = Tetris.BOARD_WIDTH * scale + 20
                preview_offset_y = 50
                self.render_next_piece(scale, screen, offset_x=preview_offset_x, offset_y=preview_offset_y)
                pygame.display.flip()
                clock.tick(60)

        # 최종 위치에 고정
        self.current_pos = [x, drop_y]
        new_board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, board_cleared = self._clear_lines(new_board)
        self.set_board(board_cleared)

        gained = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        self.score += gained

        # 다음 턴 준비
        self._new_round()
        if self.game_over:
            gained -= 5  # 게임 오버 시 패널티

        # play()가 반환하는 세 번째 값은 현재 보드(고정+낙하중 블록 포함)
        return gained, self.game_over, self._get_complete_board()

    def render(self, scale, screen):
        """
        Pygame으로 실제 화면을 그리는 부분 (게임 보드 + 점수)
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

    def render_next_piece(self, scale, screen, offset_x, offset_y):
        """
        다음에 나올 블록을 미리보기 영역에 그려주는 함수
        - scale: 한 칸의 픽셀 크기
        - screen: pygame 화면 객체
        - offset_x, offset_y: 미리보기 영역의 좌상단 기준 좌표 (픽셀 단위)
        """
        pid = self.next_piece
        # 항상 회전 상태 0번으로 미리보기
        shape = Tetris.TETROMINOS[pid][0]
        # shape의 최소 좌표를 (0,0)에 맞추기 위해 정규화
        min_x = shape[:,0].min()
        min_y = shape[:,1].min()
        normalized = shape - np.array([min_x, min_y])

        # (선택 사항) 미리보기 영역 배경 그리기
        # preview_w = 4 * scale
        # preview_h = 4 * scale
        # pygame.draw.rect(screen, (200, 200, 200), (offset_x, offset_y, preview_w, preview_h))

        # 정규화된 모양을 그리기
        for (x_cell, y_cell) in normalized:
            color = Tetris.COLORS[pid + 1]
            pygame.draw.rect(
                screen,
                color,
                (offset_x + x_cell * scale, offset_y + y_cell * scale, scale, scale),
                border_radius=3
            )


def main():
    # ------------------ pygame 초기화 및 화면 생성 ------------------
    pygame.init()
    scale = 30

    # 미리보기 영역을 위한 여유 공간
    preview_margin = 20
    preview_cell_width = 4  # 가로로 최대 4셀 들어가는 영역을 잡는다

    # 실제 창 크기: (보드 너비*scale) + preview_margin + (preview_cell_width*scale)
    screen_width = Tetris.BOARD_WIDTH * scale + preview_margin + preview_cell_width * scale
    screen_height = Tetris.BOARD_HEIGHT * scale

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Tetris Colored Blocks with Next-Piece Preview")
    clock = pygame.time.Clock()

    game = Tetris()
    running = True

    # 예시: 자동 랜덤 드롭 루프
    while running and not game.game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 랜덤으로 가능한 상태 중 하나를 선택
        states = game.get_next_states()
        (x, rot), board_next = random.choice(list(states.items()))

        # play() 호출: 내부에서 render()와 render_next_piece()를 수행함
        gained, done, board_now = game.play(
            x, rot, render=True, screen=screen, scale=scale, clock=clock
        )

        # 애니메이션 후에도 마지막 프레임을 한 번 더 그려준다
        if not game.game_over:
            screen.fill((255, 255, 255))
            game.render(scale, screen)

            preview_offset_x = Tetris.BOARD_WIDTH * scale + preview_margin
            preview_offset_y = 50
            game.render_next_piece(scale, screen,
                                   offset_x=preview_offset_x,
                                   offset_y=preview_offset_y)

            pygame.display.flip()
            clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
