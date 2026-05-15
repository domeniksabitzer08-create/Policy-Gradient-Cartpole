# env
import gymnasium as gym
# torch
import torch
# others
from tqdm.auto import tqdm

env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1", render_mode = "human")
state, _ = env.reset()
print(f"state: {state}")
print(f"shape: {state.shape}")
print(f"action space: {env.action_space.n}")

### NETWORK ###
class PolicyNet(torch.nn.Module):
    def __init__(self, input_features=4, output_features=2, hidden_units=64):
        super(PolicyNet, self).__init__()
        self.hidden_units = hidden_units
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(input_features, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, output_features),
            torch.nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.layer_stack(x)

### AGENT ###
class Agent:
    def __init__(self, gamma, lr):
        self.gamma = gamma
        self.policy_net = PolicyNet()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        probs = self.policy_net(state)
        dist = torch.distributions.Categorical(probs)
        action = torch.distributions.categorical.Categorical(probs=probs).sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def train(self, episodes: int):
        sprint_score = 0
        for episode in tqdm(range(episodes)):
            episode_score = 0
            log_probs = []
            rewards = []
            losses = []
            state, _ = env.reset()
            is_done = False
            while not is_done:
                action, log_prob = self.select_action(torch.tensor(state))
                state, reward, terminal, truncated, info = env.step(action)
                is_done = terminal or truncated
                log_probs.append(log_prob)
                rewards.append(reward)
            # Calculate returns
            G = 0
            returns = []
            for reward in reversed(rewards):
                G = reward + G * self.gamma
                returns.insert(0, G)
            # turn returns to tensor
            returns = torch.tensor(returns, dtype=torch.float32)
            # Calculate the loss
            loss = 0
            for log_prob, G in zip(log_probs, returns):
                loss += -log_prob * G
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ### DATA ###
            sprint_score += len(rewards)
            avg_score = sprint_score / (episode+1)
            if episode % 100 == 0:
                print(f"Episode: {episode} | avg Score: {avg_score:.3f}")
                sprint_score = 0




agent = Agent(0.99, 0.001)
agent.train(1000)


