import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import pygame

from PPO_agent import PPOAgent
from tetris_board import Tetris
from tensorflow.keras import backend as K

# ─── XLA JIT 활성화 (안정적으로 한 번 더) ───────────────────────────────────
tf.config.optimizer.set_jit(True)
# ───────────────────────────────────────────────────────────────────────────

def greedy_index(agent: PPOAgent, boards: np.ndarray) -> int:
    """
    검증 시에 사용: logits이 가장 큰 index 반환
    boards: np.ndarray of shape (K,20,10), dtype=np.float32
    """
    boards_tf = tf.convert_to_tensor(boards, dtype=tf.float32)
    lo, _ = agent.model(boards_tf, training=False)     # lo: (K,1)
    lo = tf.squeeze(lo, axis=-1)                       # (K,)
    # TensorFlow 연산만으로 argmax 구하기
    idx_tf = tf.argmax(lo, axis=0)                     # int64 tensor

    try:
        idx = int(idx_tf.numpy())
    except Exception:
        idx = int(K.get_value(idx_tf))

    return idx

def rollout(env: Tetris, agent: PPOAgent, max_steps=500,
            render=False, screen=None, scale=30, clock=None):
    """
    한 에피소드(롤아웃) 동안:
      1) 다음 가능한 상태(boards) 추출
      2) agent.select_action(boards) 호출
      3) env.play(x,rot) → 보상·done 얻고 버퍼 저장
      4) ROLLOUT_STEPS 이상이면 agent.train() 호출
    """
    score, step, done = 0, 0, False
    env.reset()

    while not done and step < max_steps:
        next_dict = env.get_next_states()  # dict {(x,rot): next_board}
        boards = np.array(list(next_dict.values()), dtype=np.float32)  # (K,20,10)

        idx, logp, val = agent.select_action(boards)
        (x, rot) = list(next_dict.keys())[idx]

        reward, done, _ = env.play(x, rot,
                                   render=render,
                                   screen=screen,
                                   scale=scale,
                                   clock=clock)
        agent.store(boards, idx, logp, val, reward, done)

        score = env.get_game_score()
        step += 1

        if agent.ready():
            final_board = env._get_complete_board().astype(np.float32)
            final_board_tf = tf.expand_dims(final_board, axis=0)  # (1,20,10)
            _, last_val_tf = agent.model(final_board_tf, training=False)
            last_val = float(tf.squeeze(last_val_tf, axis=-1).numpy())
            agent.train(last_value=last_val)

    return score, step

def validate(env: Tetris, agent: PPOAgent, reps=5, max_steps=500,
             render_first=True, screen=None, scale=30, clock=None):
    """
    greedy policy로 검증 에피소드 수행
    """
    scores = []
    for i in range(reps):
        env.reset()
        done, step = False, 0
        render = render_first and (i == 0)

        while not done and step < max_steps:
            next_dict = env.get_next_states()
            boards = np.array(list(next_dict.values()), dtype=np.float32)
            idx = greedy_index(agent, boards)
            (x, rot) = list(next_dict.keys())[idx]
            _, done, _ = env.play(x, rot,
                                  render=render,
                                  screen=screen,
                                  scale=scale,
                                  clock=clock)
            step += 1

        scores.append(env.get_game_score())

    return np.mean(scores), np.max(scores)

def train_ppo(episodes=1000,
              validate_every=25,
              num_val_reps=5,
              max_steps=500):
    """
    전체 학습 루프
    """
    pygame.init()
    scale  = 30
    screen = pygame.display.set_mode((Tetris.BOARD_WIDTH * scale,
                                      Tetris.BOARD_HEIGHT * scale))
    clock  = pygame.time.Clock()

    env   = Tetris()
    agent = PPOAgent()

    train_scores, val_means = [], []
    for ep in tqdm(range(1, episodes+1)):
        tr_score, _ = rollout(env, agent,
                              max_steps,
                              render=False,
                              screen=screen,
                              scale=scale,
                              clock=clock)
        train_scores.append(tr_score)

        if ep % validate_every == 0:
            mean_val, max_val = validate(env, agent,
                                         num_val_reps,
                                         max_steps,
                                         render_first=True,
                                         screen=screen,
                                         scale=scale,
                                         clock=clock)
            val_means.append(mean_val)
            print(f"\n[Val {ep}] mean {mean_val:.1f} / max {max_val}\n")

    # 학습 곡선 시각화
    plt.plot(train_scores, label="Train score")
    if val_means:
        xs = list(range(validate_every, episodes+1, validate_every))
        plt.plot(xs, val_means, label="Val mean")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.show()
    pygame.quit()

if __name__ == "__main__":
    tf.config.optimizer.set_jit(True)
    train_ppo(episodes=100)
