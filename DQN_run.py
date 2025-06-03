import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm
import pygame

from DQN_agent4 import DQNAgent
from tetris_board import Tetris
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

def experience(env, agent, max_steps=200, screen=None, scale=25, clock=None):
    """
    한 에피소드 동안:
    - 현재 상태(20×10)에서 가능한 다음 상태(20×10)를 구하고,
    - agent.best_state()를 통해 다음 상태를 선택,
    - play()를 호출하여 실제 보드를 업데이트,
    - 리플레이 메모리에 (state, next_state, reward, done) 저장
    """
    state = env.reset()  # 이제 shape=(20,10)
    done = False
    steps = 0

    while not done and steps < max_steps:
        next_states = env.get_next_states()  # dict: {(x,rot): (20×10) 배열}
        # best_state 자체가 “(20×10) 배열” 형태로 받아짐
        best_board = agent.best_state(next_states.values(), exploration=True)

        # best_board와 매칭되는 action (x, rot)를 찾아야 함
        # (딕셔너리 키, 값 짝 맞추기)
        action = next(a for a, board in next_states.items() if np.array_equal(board, best_board))
        x, rot = action

        # 실제 game.play() 호출 → (점수, game_over, 현재전체보드) 반환
        gained, done, board_now = env.play(x, rot, render=True,
                                           screen=screen, scale=scale, clock=clock)

        # 메모리에 (이전 state, 다음 state, 보상, 종료여부) 저장
        agent.add_to_memory(state, best_board, gained, done)

        # 상태 업데이트
        state = best_board.copy()
        steps += 1

def validate(env, agent, num_reps=5, max_steps=200,
             render_first=True, screen=None, scale=30, clock=None):
    """
    검증용 루프:
    - exploration=False (항상 예측된 best_state 사용)
    """
    scores, steps_list = [], []

    for i in range(num_reps):
        state = env.reset()  # (20,10)
        done = False
        step = 0
        render = (render_first and i == 0)

        while not done and step < max_steps:
            next_states = env.get_next_states()
            # exploration=False
            best_board = agent.best_state(next_states.values(), exploration=False)
            action = next(a for a, board in next_states.items() if np.array_equal(board, best_board))
            x, rot = action

            gained, done, board_now = env.play(x, rot, render=render,
                                               screen=screen, scale=scale, clock=clock)
            if screen is not None and render:
                screen.fill((255, 255, 255))
                env.render(scale, screen)
                pygame.display.flip()
                clock.tick(5)

            state = best_board.copy()
            step += 1

        scores.append(env.get_game_score())
        steps_list.append(step)

    print(f"\nValidation\n"
          f"Max Score: {max(scores)}\t"
          f"Mean Score: {np.mean(scores):.2f}\t"
          f"Mean Steps: {np.mean(steps_list):.2f}\n")
    return scores, steps_list


def train(env, agent, screen=None, scale=30, clock=None,
          episodes=100, train_every=1, validate_every=25,
          num_val_reps=5, max_steps=200,
          memory_batch_size=512, training_batch_size=512, epochs=1):

    scores, val_scores, loss_log = [], [], []

    for ep in tqdm(range(1, episodes + 1)):
        # ─ play one episode & fill replay buffer ─
        experience(env, agent, max_steps, screen, scale, clock)
        scores.append(env.get_game_score())

        # ─ training ─
        loss = None
        if ep % train_every == 0:
            loss = agent.train(memory_batch_size,
                               training_batch_size,
                               epochs)
        loss_log.append(loss)

        # ─ validation ─
        if ep % validate_every == 0:
            v_scores, _ = validate(env, agent, num_val_reps, max_steps,
                                   render_first=True,
                                   screen=screen, scale=scale, clock=clock)
            val_scores.append(mean(v_scores))

    # ─── Score 그래프 ───
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, len(scores) + 1), scores,
             label='Train Score', color='tab:blue')
    if val_scores:
        val_x = list(range(validate_every, episodes + 1, validate_every))
        ax1.plot(val_x, val_scores,
                 label='Validation Score', color='tab:green')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Score'); ax1.set_title('Score')
    ax1.legend(); plt.tight_layout(); plt.show()

    # ─── Loss 그래프 ───
    xs = [i for i, l in enumerate(loss_log, 1) if l is not None]
    ys = [l for l in loss_log if l is not None]

    if ys:  # 학습이 실제로 발생했을 때만 그림
        fig2, ax2 = plt.subplots()
        ax2.plot(xs, ys, label='Training Loss', color='tab:red')
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Loss')
        ax2.set_title('DQN Training Loss')
        ax2.legend(); plt.tight_layout(); plt.show()



def main():
    pygame.init()
    scale = 30
    screen = pygame.display.set_mode((Tetris.BOARD_WIDTH * scale,
                                      Tetris.BOARD_HEIGHT * scale))
    pygame.display.set_caption("Tetris RL")
    clock = pygame.time.Clock()

    # 하이퍼파라미터
    episodes = 100
    max_steps = 1000
    train_every = 1
    validate_every = 25
    num_val_reps = 5
    memory_batch_size = 512
    training_batch_size = 512
    epochs = 1

    env = Tetris()
    # get_state_size() = 200
    agent = DQNAgent(
        env.get_state_size(),
        epsilon=1, epsilon_min=0.1,
        epsilon_stop_episode=1500,
        use_target_model=False,
        update_target_every=None,
        mem_size=10000,
        discount=0.95,
        replay_start_size=500
    )

    train(env=env,
          agent=agent,
          screen=screen,
          scale=scale,
          clock=clock,
          episodes=episodes,
          train_every=train_every,
          validate_every=validate_every,
          num_val_reps=num_val_reps,
          max_steps=max_steps,
          memory_batch_size=memory_batch_size,
          training_batch_size=training_batch_size,
          epochs=epochs
          )

    pygame.quit()

if __name__ == '__main__':
    main()
