import time
import torch
import asyncio
import threading
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import matplotlib.pyplot as plt
import torch.distributions as distributions
from torch.utils.data import DataLoader, TensorDataset

import environment 

class BackboneNetwork(nn.Module):
    def __init__(self, in_features, hidden_dimensions, out_features, dropout):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dimensions)
        self.layer2 = nn.Linear(hidden_dimensions, hidden_dimensions)
        self.layer3 = nn.Linear(hidden_dimensions, out_features)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = f.relu(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x
    
class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
    
class ppo():


    def save_model(self, agent, filename):
        torch.save(agent.state_dict(), filename)
        print(f'Model saved to {filename}')

    def load_model(self, agent, filename):
        agent.load_state_dict(torch.load(filename))
        # agent.eval()  # Set the model to evaluation mode
        print(f'Model loaded from {filename}')

    def create_agent(self, hidden_dimensions, dropout):
        INPUT_FEATURES = 5
        HIDDEN_DIMENSIONS = hidden_dimensions
        ACTOR_OUTPUT_FEATURES = 6
        CRITIC_OUTPUT_FEATURES = 1
        DROPOUT = dropout
        actor = BackboneNetwork(
                INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT)
        critic = BackboneNetwork(
                INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT)
        agent = ActorCritic(actor, critic)
        return agent

    def calculate_returns(self, rewards, discount_factor):
        print("Rewards: " + str(rewards))
        print("DiscountFactor: " + str(discount_factor))
        returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + cumulative_reward * discount_factor
            returns.insert(0, cumulative_reward)
        
        returns = torch.tensor(returns)
        print("returns(pre normal): " + str(returns))
        
        # Only normalize if there are enough elements
        if returns.numel() > 1:
            std_dev = returns.std()
            if std_dev > 0:
                returns = (returns - returns.mean()) / std_dev
            else:
                print("Standard deviation is zero, skipping normalization.")
        else:
            print("Not enough elements to normalize, returning raw returns.")
        
        print("returns: " + str(returns))
        return returns

    def calculate_advantages(self, returns, values):
        advantages = returns - values
        # Normalize the advantage
        advantages = (advantages - advantages.mean()) / advantages.std()
        print("advantages: " + str(advantages))
        return torch.tensor(advantages)

    def calculate_surrogate_loss(self, 
            actions_log_probability_old,
            actions_log_probability_new,
            epsilon,
            advantages):
        advantages = advantages.detach()
        policy_ratio = (
                actions_log_probability_new - actions_log_probability_old
                ).exp()
        surrogate_loss_1 = policy_ratio * advantages
        surrogate_loss_2 = torch.clamp(
                policy_ratio, min=1.0-epsilon, max=1.0+epsilon
                ) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss

    def calculate_losses(self, 
            surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
        entropy_bonus = entropy_coefficient * entropy
        policy_loss = -(surrogate_loss + entropy_bonus).sum()
        value_loss = f.smooth_l1_loss(returns, value_pred).sum()
        return policy_loss, value_loss

    def init_training(self, ):
        states = []
        actions = []
        actions_log_probability = []
        values = []
        rewards = []
        done = False
        episode_reward = 0
        return states, actions, actions_log_probability, values, rewards, done, episode_reward

    ## TODO: modifiy enviroment.py to act as a good enviroment for this chunk of code
    def forward_pass(self, env, agent, optimizer, discount_factor):
        states, actions, actions_log_probability, values, rewards, done, episode_reward = self.init_training()
        
        state = env.reset() # reset the environment and get initial state
        print("State: " + str(state))
        agent.train()

        while not done:
            # convert the data struct of state into tensor
            state = torch.FloatTensor(state).unsqueeze(0)
            states.append(state)

            # predict the observation state after taking the action
            action_pred, value_pred = agent(state)
            print("action pred: " + str(action_pred))
            print("value_pred: " + str(value_pred))
            action_prob = f.softmax(action_pred, dim=-1)
            print("action_prob: " + str(action_prob))
            dist = distributions.Categorical(action_prob)
            print("dist: " + str(dist))
            action = dist.sample()
            print("action: " + str(action))
            log_prob_action = dist.log_prob(action)
            print("log_prob_action: " + str(log_prob_action))

            # send the action to the environment and get new information from it 
            state, reward, done, _ = env.step(action.item())
            print("State: " + str(state))
            print("Reward: " + str(reward))
            print("Done: " + str(done))

            # collect all the data
            actions.append(action)
            actions_log_probability.append(log_prob_action)
            values.append(value_pred)
            rewards.append(reward)
            episode_reward += reward
            print("loop")

        # post loop concatenation of data
        states = torch.cat(states)
        actions = torch.cat(actions)
        actions_log_probability = torch.cat(actions_log_probability)
        values = torch.cat(values).squeeze(-1)
        returns = self.calculate_returns(rewards, discount_factor)
        advantages = self.calculate_advantages(returns, values)

        return episode_reward, states, actions, actions_log_probability, advantages, returns

    def update_policy(self, 
            agent,
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns,
            optimizer,
            ppo_steps,
            epsilon,
            entropy_coefficient):
        BATCH_SIZE = 128
        total_policy_loss = 0
        total_value_loss = 0
        actions_log_probability_old = actions_log_probability_old.detach()
        actions = actions.detach()
        training_results_dataset = TensorDataset(
                states,
                actions,
                actions_log_probability_old,
                advantages,
                returns)
        batch_dataset = DataLoader(
                training_results_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False)
        for _ in range(ppo_steps):
            for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
                # get new log prob of actions for all input states
                action_pred, value_pred = agent(states)
                value_pred = value_pred.squeeze(-1)
                action_prob = f.softmax(action_pred, dim=-1)
                print("Action Probabilities:", str(action_prob))
                probability_distribution_new = distributions.Categorical(
                        action_prob)
                entropy = probability_distribution_new.entropy()
                # estimate new log probabilities using old actions
                actions_log_probability_new = probability_distribution_new.log_prob(actions)
                surrogate_loss = self.calculate_surrogate_loss(
                        actions_log_probability_old,
                        actions_log_probability_new,
                        epsilon,
                        advantages)
                policy_loss, value_loss = self.calculate_losses(
                        surrogate_loss,
                        entropy,
                        entropy_coefficient,
                        returns,
                        value_pred)
                optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                optimizer.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
        return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


    def evaluate(self, env, agent):
        agent.eval()
        rewards = []
        done = False
        episode_reward = 0
        state = env.reset()
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_pred, _ = agent(state)
                action_prob = f.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _ = env.step(action.item())
            episode_reward += reward
        return episode_reward


    def plot_train_rewards(self, train_rewards, reward_threshold):
        plt.figure(figsize=(12, 8))
        plt.plot(train_rewards, label='Training Reward')
        plt.xlabel('Episode', fontsize=20)
        plt.ylabel('Training Reward', fontsize=20)
        plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def plot_test_rewards(self, test_rewards, reward_threshold):
        plt.figure(figsize=(12, 8))
        plt.plot(test_rewards, label='Testing Reward')
        plt.xlabel('Episode', fontsize=20)
        plt.ylabel('Testing Reward', fontsize=20)
        plt.hlines(reward_threshold, 0, len(test_rewards), color='y')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def plot_losses(self, policy_losses, value_losses):
        plt.figure(figsize=(12, 8))
        plt.plot(value_losses, label='Value Losses')
        plt.plot(policy_losses, label='Policy Losses')
        plt.xlabel('Episode', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def __init__(self, seed=11111111, currentEpisode=1, maxEpisodes=500):
        self.env_train = environment.IoaiNavEnv(headless=False, seed=seed)
        self.env_test = environment.IoaiNavEnv(headless=True, seed=seed)
        self.currentEpisode = currentEpisode
        self.MAX_EPISODES = maxEpisodes
        self.ppo_done = False

    def run_ppo(self, name="", load=False):
        DISCOUNT_FACTOR = 0.99
        REWARD_THRESHOLD = 475
        PRINT_INTERVAL = 10
        PPO_STEPS = 8
        N_TRIALS = 100
        EPSILON = 0.2
        ENTROPY_COEFFICIENT = 0.01
        HIDDEN_DIMENSIONS = 64
        DROPOUT = 0.2
        LEARNING_RATE = 0.001
        train_rewards = []
        test_rewards = []
        policy_losses = []
        value_losses = []
        agent = self.create_agent(HIDDEN_DIMENSIONS, DROPOUT)
        if (load):
            self.load_model(agent, "./models/model-" + str(name) + ".pth")
        optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)
        for episode in range(self.currentEpisode, self.MAX_EPISODES+1):
            train_reward, states, actions, actions_log_probability, advantages, returns = self.forward_pass(
                    self.env_train,
                    agent,
                    optimizer,
                    DISCOUNT_FACTOR)
            policy_loss, value_loss = self.update_policy(
                    agent,
                    states,
                    actions,
                    actions_log_probability,
                    advantages,
                    returns,
                    optimizer,
                    PPO_STEPS,
                    EPSILON,
                    ENTROPY_COEFFICIENT)
            test_reward = self.evaluate(self.env_test, agent)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
            mean_train_rewards = np.mean(train_rewards[-N_TRIALS:])
            mean_test_rewards = np.mean(test_rewards[-N_TRIALS:])
            mean_abs_policy_loss = np.mean(np.abs(policy_losses[-N_TRIALS:]))
            mean_abs_value_loss = np.mean(np.abs(value_losses[-N_TRIALS:]))
            if episode % PRINT_INTERVAL == 0:
                print(f'Episode: {episode:3} | \
                    Mean Train Rewards: {mean_train_rewards:3.1f} \
                    | Mean Test Rewards: {mean_test_rewards:3.1f} \
                    | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                    | Mean Abs Value Loss: {mean_abs_value_loss:2.2f}')
            if mean_test_rewards >= REWARD_THRESHOLD:
                print(f'Reached reward threshold in {episode} episodes')
                break
            self.save_model(agent, "./models/model-" + str(name) + ".pth")
            self.currentEpisode += 1
        self.ppo_done = True
        self.plot_train_rewards(train_rewards, REWARD_THRESHOLD)
        self.plot_test_rewards(test_rewards, REWARD_THRESHOLD)
        self.plot_losses(policy_losses, value_losses)


# if __name__ == "__main__":
#     env = ppo()
#     #env.__init__()
#     env.run_ppo()
