import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm
import pygame

from REINFORCE_agent import REINFORCEAgent
from tetris_board import Tetris
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

def experience(env, agent, max_steps=200, screen=None, scale=25, clock=None):
    state = env.reset()
    done  = False
    steps = 0

    while not done and steps < max_steps:
        next_states = env.get_next_states()
        best_board, _ = agent.best_state(next_states.values(), exploration=True)

        # 액션 추출
        action = next(a for a, board in next_states.items()
                      if np.array_equal(board, best_board))
        x, rot = action

        reward, done, _ = env.play(
            x, rot, render=True, screen=screen, scale=scale, clock=clock)

        # ⬇️  log_prob 대신 board(=state) 저장
        agent.add_to_memory(best_board, reward)

        state = best_board.copy()
        steps += 1



def validate(env, agent, num_reps=5, max_steps=200,
             render_first=True, screen=None, scale=30, clock=None):
    """
    exploration=False 로 정책을 평가하는 함수
    """
    scores, steps_list = [], []

    for i in range(num_reps):
        state  = env.reset()
        done   = False
        step   = 0
        render = (render_first and i == 0)

        while not done and step < max_steps:
            next_states = env.get_next_states()

            # ──────────────────────────────────────────────
            # ✔ best_state 가 (board, log_prob) 튜플을 주므로 두 값 모두 받기
            # ──────────────────────────────────────────────
            best_board, _ = agent.best_state(next_states.values(), exploration=False)

            action = next(a for a, board in next_states.items()
                          if np.array_equal(board, best_board))
            x, rot = action

            gained, done, board_now = env.play(
                x, rot, render=render,
                screen=screen, scale=scale, clock=clock
            )

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


def train(env,
          agent,
          screen=None,
          scale=30,
          clock=None,
          episodes=1000,
          train_every=1,
          validate_every=25,
          num_val_reps=5,
          max_steps=200,
          memory_batch_size=512,
          training_batch_size=512,
          epochs=1):
    """
    DQN 학습 루프:
    - 매 에피소드마다 experience()로 상태-행동-보상 데이터를 모으고,
    - train_every마다 agent.train() 호출
    - validate_every마다 validate() 호출
    """
    scores, val_scores = [], []

    for ep in tqdm(range(1, episodes + 1)):
        experience(env, agent, max_steps, screen, scale, clock)
        scores.append(env.get_game_score())

        if ep % train_every == 0:
            agent.train()

        if ep % validate_every == 0:
            v_scores, _ = validate(env, agent, num_val_reps, max_steps,
                                   render_first=True,
                                   screen=screen, scale=scale, clock=clock)
            val_scores.append(mean(v_scores))

    # 학습 곡선 출력
    plt.plot(range(1, len(scores) + 1), scores, label='Train')
    if val_scores:
        x = list(range(validate_every, episodes + 1, validate_every))
        plt.plot(x, val_scores, label='Validation')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

def main():
    pygame.init()
    scale = 30
    screen = pygame.display.set_mode((Tetris.BOARD_WIDTH * scale,
                                      Tetris.BOARD_HEIGHT * scale))
    pygame.display.set_caption("Tetris RL")
    clock = pygame.time.Clock()

    # 하이퍼파라미터
    episodes = 1000
    max_steps = 1000
    train_every = 1
    validate_every = 25
    num_val_reps = 5
    memory_batch_size = 512
    training_batch_size = 512
    epochs = 1

    env = Tetris()
    # get_state_size() = 200
    agent = REINFORCEAgent(
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
