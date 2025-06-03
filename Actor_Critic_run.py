# run_actor_critic.py

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm
import pygame

from Actor_Critic_agent import ActorCriticAgent
from tetris_board import Tetris
import torch

# GPU 사용 가능 여부 확인
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


def experience(env, agent, max_steps=200, screen=None, scale=25, clock=None):
    """
    한 에피소드 동안의 경험 수집.
    1) state 초기화
    2) 매 스텝마다:
       - env.get_next_states() 로 {(x,rot): board} 딕셔너리 획득
       - model.predict() → 액션 선택
       - env.play(x, rot, …) 호출 → reward, done
       - agent.record_reward(reward)
       - if done or steps>=max_steps: 종료
    """
    state = env.reset()  # state는 (20,10) numpy array
    done = False
    steps = 0

    while not done and steps < max_steps:
        # ① 가능한 다음 상태(보드)들 얻기: {(x,rot): board}
        next_dict = env.get_next_states()
        action_list = list(next_dict.keys())           # [(x1,rot1), (x2,rot2), ...]
        cand_boards = list(next_dict.values())          # [board1, board2, ...]

        # ② Agent가 행동 인덱스를 선택
        idx = agent.select_action(cand_boards, greedy=False)

        # ③ 해당 인덱스의 (x,rot) 행동 추출
        x, rot = action_list[idx]

        # ④ 실제 한 스텝 진행 (시각화 포함)
        reward, done, _ = env.play(x, rot,
                                   render=True,
                                   screen=screen, scale=scale, clock=clock)

        # ⑤ 보상 기록
        agent.record_reward(reward)

        # ⑥ 다음 스텝을 위해 상태 갱신
        state = next_dict[(x, rot)]  # 실제로는 dispaly에는 render된 board 사용
        steps += 1


def validate(env, agent, num_reps=5, max_steps=200,
             render_first=True, screen=None, scale=30, clock=None):
    """
    검증용 루프 (exploration=False):
    매 반복마다:
     1) env.reset() → 초기 상태
     2) 매 스텝:
       - env.get_next_states() → next_dict
       - agent.select_action(cand_boards, greedy=True)
       - env.play(x, rot) → reward만 무시, done 체크
       - (render_first인 경우 첫 번째 반복에는 화면 그리기)
     3) 에피소드 종료 후 점수 기록
     4) num_reps만큼 반복 → 평균/최대 스코어 출력
    """
    scores = []
    steps_list = []

    for i in range(num_reps):
        state = env.reset()  # (20,10)
        done = False
        step = 0
        render = (render_first and i == 0)

        while not done and step < max_steps:
            next_dict = env.get_next_states()
            action_list = list(next_dict.keys())
            cand_boards = list(next_dict.values())

            # greedy=True 로 행동 선택
            idx = agent.select_action(cand_boards, greedy=True)
            x, rot = action_list[idx]

            # render 여부에 따라 화면 그리기
            gained, done, _ = env.play(x, rot,
                                       render=render,
                                       screen=screen, scale=scale, clock=clock)
            if screen is not None and render:
                screen.fill((255, 255, 255))
                env.render(scale, screen)
                pygame.display.flip()
                clock.tick(5)

            state = next_dict[(x, rot)].copy()
            step += 1

        scores.append(env.get_game_score())
        steps_list.append(step)

    print(f"\nValidation\n"
          f"Max Score: {np.max(scores)}\t"
          f"Mean Score: {np.mean(scores):.2f}\t"
          f"Mean Steps: {np.mean(steps_list):.2f}\n")
    return scores, steps_list


def train(env,
          agent,
          screen=None,
          scale=30,
          clock=None,
          episodes=1000,
          validate_every=25,
          num_val_reps=5,
          max_steps=200):
    """
    A2C 학습 루프:
      · 매 에피소드 experience()로 데이터를 모으고
      · agent.train_episode() 호출해서 loss 반환받기
      · 일정 간격마다 validate()로 점수 확인
      · 학습 곡선(Train/Validation 스코어 + Loss) 출력
    """
    train_scores = []     # 에피소드별 점수 (env.get_game_score())
    val_scores = []       # validation 시 평균 점수
    loss_history = []     # 에피소드별 학습 손실

    for ep in range(1, episodes + 1):
        # 1) 경험 수집
        experience(env, agent, max_steps, screen, scale, clock)

        # 2) 에피소드가 끝나면 학습 업데이트 → loss 리턴
        loss_value = agent.train_episode()
        loss_history.append(loss_value)

        # 3) 현재 에피소드 최종 점수 기록
        epi_score = env.get_game_score()
        train_scores.append(epi_score)

        # 4) 콘솔에 진행 상황 출력 (every 10 에피소드마다)
        if ep % 10 == 0:
            print(f"Episode {ep}/{episodes} → Train Score: {epi_score} \tLoss: {loss_value:.4f}")

        # 5) 일정 간격마다 검증 수행
        if ep % validate_every == 0:
            v_scores, _ = validate(env, agent, num_val_reps, max_steps,
                                   render_first=True,
                                   screen=screen, scale=scale, clock=clock)
            val_scores.append(np.mean(v_scores))

    # ────────────────────────────────────────────────────────────────
    #  학습이 끝난 뒤: Reward(Train vs Validation)곡선 & Loss 곡선 그리기
    # ────────────────────────────────────────────────────────────────

    # 학습 곡선 출력
    plt.plot(range(1, len(train_scores) + 1), train_scores, label='Train')
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
    pygame.display.set_caption("Tetris A2C")
    clock = pygame.time.Clock()

    # 하이퍼파라미터
    episodes = 1000
    max_steps = 1000
    validate_every = 25
    num_val_reps = 5

    env = Tetris()
    agent = ActorCriticAgent(
        state_shape=(20, 10),
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_stop_episode=1500,
        add_batch_norm=False
    )

    train(env=env,
          agent=agent,
          screen=screen,
          scale=scale,
          clock=clock,
          episodes=episodes,
          validate_every=validate_every,
          num_val_reps=num_val_reps,
          max_steps=max_steps)

    pygame.quit()


if __name__ == '__main__':
    main()
