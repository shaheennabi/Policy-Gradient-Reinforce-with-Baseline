import torch
import gymnasium as gym
from networks import PolicyNet, ValueNet
import torch.optim as optim
from config import POLICY_LR, VALUE_LR
from torch.distributions import Categorical
from losses import policy_loss, value_loss
from returns import returns




def training_reinforce_with_baseline(num_episodes):
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(state_dim, action_dim)
    value_net = ValueNet(state_dim)

    policy_optim = optim.Adam(policy.parameters(), lr=POLICY_LR)
    value_optim = optim.Adam(value_net.parameters(), lr=VALUE_LR)

    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False


        log_probs = []
        values = []
        rewards = []

        while not done: 
            state_t = torch.tensor(state, dtype=torch.float32)
            probabilities = policy(state_t)
            dist = Categorical(probabilities)
            action = dist.sample()
            ##log prob
            log_probs.append(dist.log_prob(action))

            ## values
            v = value_net(state_t)
            values.append(v)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            rewards.append(reward)
            state = next_state

        actual_returns = returns(rewards=rewards)
        actual_returns = torch.tensor(actual_returns, dtype=torch.float32)
        values = torch.cat(values).squeeze()

        ## advantage
        advantage = actual_returns - values.detach()

        ## policy update
        policy_loss(log_probs=log_probs, advantage=advantage,  policy_optim=policy_optim)

        ## value update
        value_loss(values=values, actual_returns=actual_returns, value_optim=value_optim)

        # Log progress
        total_reward = sum(rewards)
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f"Episode {episode+1}/{num_episodes}, total reward: {total_reward}")


        





