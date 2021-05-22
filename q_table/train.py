import gym 
from q_learning import QLearningAgent


def run_episode(env, agent, render=False):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0
    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）

    while True:
        action = agent.sample(obs) # 根据算法选择一个动作
        next_obs, reward, done, _ = env.step(action) # 与环境进行一个交互
        # 训练 Q-learning算法
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps

def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        # time.sleep(0.5)
        # env.render()
        if done:
            break
    return total_reward

if __name__ == '__main__':
    # 使用gym创建悬崖环境
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left

    obs_n=env.observation_space.n
    # 创建一个agent实例，输入超参数
    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=0.1,
        gamma=0.9,
        e_greed=0.1)


    # 训练500个episode，打印每个episode的分数
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, False)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
