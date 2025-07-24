import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import random
import logging
import time
import pickle
from datetime import datetime, timedelta, timezone
from collections import deque
from ta.momentum import RSIIndicator
from ta.trend import MACD

# Configure logging
logging.basicConfig(
    filename='realtime_trading.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
T = 15
batch_size = 32
actor_lr = 1e-3
critic_lr = 5e-4
gamma = 0.99
tau = 0.005
initial_balance = 10_000_000
checkpoint_dir = './realtime_checkpoints'
min_buffer_size = 29
num_base_models = 3
max_trades_per_day = 1000
max_shares = 100.0

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device)
        )

    def __len__(self):
        return len(self.buffer)

class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class ALSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        weights = torch.softmax(self.attention(out).squeeze(-1), dim=1)
        context = torch.sum(out * weights.unsqueeze(-1), dim=1)
        return self.fc(context)

class TransformerModel(nn.Module):
    def __init__(self, input_size=5, d_model=64, output_size=1, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h = self.transformer(x)
        return self.fc(h.mean(dim=1))

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

class RealTimeStockEnv:
    def __init__(self, window_size):
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        self.data_buffer = deque(maxlen=window_size*2)
        self.last_timestamp = None
        self.buffer_checkpoints = {29: False}
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = initial_balance
        self.shares = 0.0
        self.total_assets = [initial_balance]
        self.trade_log = []
        self.buffer_checkpoints = {29: False}
        return None

    def update(self, new_data, timestamp):
        timestamp_minute = timestamp.replace(second=0, microsecond=0)
        if self.last_timestamp and timestamp_minute <= self.last_timestamp:
            logging.debug(f"Skipping duplicate or old timestamp: {timestamp}")
            return None
        
        current_buffer = len(self.data_buffer)
        for checkpoint in self.buffer_checkpoints:
            if current_buffer == checkpoint and not self.buffer_checkpoints[checkpoint]:
                logging.info(f"Reached {current_buffer}/{min_buffer_size} points")
                print(f"Reached {current_buffer}/{min_buffer_size} points")
                self.buffer_checkpoints[checkpoint] = True
                logging.info(f"Continued buffering at {current_buffer} points")
                print(f"Continued buffering at {current_buffer} points")
                break

        self.data_buffer.append(new_data)
        logging.debug(f"Buffer updated, size: {len(self.data_buffer)}")
        
        if len(self.data_buffer) < min_buffer_size:
            remaining = min_buffer_size - len(self.data_buffer)
            wait_time = remaining * 60
            wait_str = f"{wait_time//3600:.0f}h {wait_time%3600//60:.0f}m" if wait_time >= 3600 else f"{wait_time//60:.0f}m"
            logging.info(f"Buffering: {len(self.data_buffer)}/{min_buffer_size} points, ~{wait_str} remaining")
            print(f"Buffering: {len(self.data_buffer)}/{min_buffer_size} points, ~{wait_str} remaining")
            return None
        if len(self.data_buffer) >= min_buffer_size and not hasattr(self.scaler, 'data_min_'):
            self.scaler.fit(np.array(self.data_buffer))
            logging.info(f"Scaler fitted with {len(self.data_buffer)} points, trading can begin")
            print(f"Scaler fitted with {len(self.data_buffer)} points, trading can begin")
            self.buffer_checkpoints = {29: False}
        scaled_data = self.scaler.transform([new_data])[0]
        return scaled_data

    def get_state(self):
        if len(self.data_buffer) >= T and hasattr(self.scaler, 'data_min_'):
            scaled_data = self.scaler.transform(np.array(self.data_buffer))
            return scaled_data[-T:]
        return None

def compute_technical_indicators(data_buffer):
    df = pd.DataFrame(list(data_buffer), columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    rsi = RSIIndicator(df['Close'], window=14).rsi().iloc[-1] / 100 if len(df) >= 14 else 0
    macd = MACD(df['Close']).macd_diff().iloc[-1] / df['Close'].iloc[-1] if len(df) >= 26 else 0
    roc = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] if len(df) >= 5 else 0
    return np.array([rsi, macd, roc])

def gumbel_softmax(logits, temperature=10.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-6) + 1e-6)
    y = logits + gumbel_noise
    return torch.softmax(y / temperature, dim=-1)

def sample_dirichlet_action_probs(num_actions):
    alpha = np.ones(num_actions) * 0.5  # Low concentration for diverse probabilities
    probs = np.random.dirichlet(alpha)
    return torch.FloatTensor(probs).to(device)

def save_replay_buffer(replay, path):
    try:
        with open(path, 'wb') as f:
            pickle.dump(replay.buffer, f)
        logging.info(f"Saved replay buffer to {path}")
    except Exception as e:
        logging.error(f"Error saving replay buffer: {e}")

def load_replay_buffer(path, capacity):
    replay = ReplayBuffer(capacity)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                replay.buffer = pickle.load(f)
                replay.position = len(replay.buffer) % capacity
            logging.info(f"Loaded replay buffer from {path} with {len(replay)} experiences")
            print(f"Loaded replay buffer from {path} with {len(replay)} experiences")
        except Exception as e:
            logging.error(f"Error loading replay buffer: {e}")
    return replay

def initialize_env_with_history(ticker, env, initial_points=29, max_retries=5):
    tickers = [ticker, 'XAUUSD=X', 'SPY']
    for t in tickers:
        for attempt in range(max_retries):
            try:
                points_fetched = 0
                df = yf.download(t, period='7d', interval='1m', auto_adjust=False, progress=False)
                if df.empty:
                    logging.warning(f"Attempt {attempt+1}/{max_retries} for {t}: Empty historical data")
                    print(f"Attempt {attempt+1}/{max_retries} for {t}: Empty historical data")
                    time.sleep(15 * (2 ** attempt))
                    continue
                for timestamp, row in df.iterrows():
                    if points_fetched >= initial_points:
                        break
                    data = row[['Open', 'High', 'Low', 'Close', 'Volume']].values
                    env.update(data, timestamp)
                    points_fetched += 1
                date_range = f"{df.index[0]} to {df.index[-1]}" if not df.empty else "N/A"
                logging.info(f"Ticker {t}, Attempt {attempt+1}: Pre-populated {points_fetched} historical points ({date_range})")
                print(f"Ticker {t}, Attempt {attempt+1}: Pre-populated {points_fetched} historical points ({date_range})")
                if len(env.data_buffer) >= initial_points:
                    return t
            except Exception as e:
                logging.error(f"Ticker {t}, Attempt {attempt+1}/{max_retries}: Failed to fetch historical data: {str(e)}")
                print(f"Ticker {t}, Attempt {attempt+1}/{max_retries}: Failed to fetch historical data: {str(e)}")
                time.sleep(15 * (2 ** attempt))
    logging.error(f"Failed to fetch historical data for all tickers after {max_retries} attempts")
    print(f"Failed to fetch historical data for all tickers after {max_retries} attempts")
    return ticker

def initialize_models(input_size=5):
    models = {
        'GRU': GRUModel().to(device),
        'ALSTM': ALSTMModel().to(device),
        'Transformer': TransformerModel().to(device)
    }
    state_dim = T * 5 + num_base_models * 2 + 3
    actor = Actor(state_dim, num_base_models).to(device)  # Always initialize new actor
    critic = Critic(state_dim, num_base_models).to(device)
    target_actor = Actor(state_dim, num_base_models).to(device)
    target_critic = Critic(state_dim, num_base_models).to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    for model_name in models:
        model_path = os.path.join(checkpoint_dir, f'{model_name}.pth')
        if os.path.exists(model_path):
            try:
                models[model_name].load_state_dict(torch.load(model_path))
                logging.info(f"Loaded {model_name} checkpoint")
            except Exception as e:
                logging.error(f"Error loading {model_name} checkpoint: {e}")
    critic_path = os.path.join(checkpoint_dir, 'critic.pth')
    if os.path.exists(critic_path):
        try:
            critic.load_state_dict(torch.load(critic_path))
            logging.info("Loaded critic checkpoint")
        except Exception as e:
            logging.error(f"Error loading critic checkpoint: {e}")

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    return models, actor, critic, target_actor, target_critic

def real_time_training_loop(ticker='GC=F'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    replay_path = os.path.join(checkpoint_dir, 'replay_buffer.pkl')
    

    # Initialize components
    env = RealTimeStockEnv(T)
    ticker = initialize_env_with_history(ticker, env, initial_points=29)
    base_models, actor, critic, target_actor, target_critic = initialize_models()
    replay = load_replay_buffer(replay_path, 500000)
    error_windows = {name: deque(maxlen=100) for name in base_models}

    # Initialize optimizers
    base_optimizers = {name: optim.Adam(model.parameters(), lr=1e-4) for name, model in base_models.items()}
    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    criterion = nn.MSELoss()

    prev_action = 'HOLD'
    drl_training_started = False
    max_stall_attempts = 3
    stall_count = 0
    last_buffer_size = 0
    trades_today = 0
    last_trade_date = datetime.now().date()
    exploration_steps = 0
    trade_attempts = 0

    while True:
        try:
            # Reset trades at new day
            if datetime.now().date() != last_trade_date:
                trades_today = 0
                last_trade_date = datetime.now().date()
                logging.info("Reset daily trade count for new day")

            # Fetch real-time 1-minute data
            logging.debug(f"Attempting to fetch data for {ticker}")
            df = yf.download(ticker, period='1d', interval='1m', auto_adjust=False, progress=False)
            if df.empty:
                logging.warning(f"No real-time data received for {ticker}")
                print(f"No real-time data received for {ticker}")
                stall_count += 1
                if stall_count >= max_stall_attempts:
                    logging.info("Switching to fallback ticker XAUUSD=X after 180-second break")
                    print("Switching to fallback ticker XAUUSD=X after 180-second break")
                    time.sleep(180)
                    ticker = 'XAUUSD=X'
                    yf.Ticker(ticker).session.close()
                    stall_count = 0
                else:
                    time.sleep(15 * (2 ** stall_count))
                continue

            timestamp = df.index[-1]
            timestamp_ist = timestamp.astimezone(timezone(timedelta(hours=5.5)))
            latest_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].values
            current_price = latest_data[3]
            logging.debug(f"Fetched data for {ticker} at {timestamp}: {latest_data}")

            # Update environment
            scaled = env.update(latest_data, timestamp)
            if scaled is None:
                current_buffer = len(env.data_buffer)
                if current_buffer == last_buffer_size:
                    stall_count += 1
                    if stall_count >= max_stall_attempts:
                        logging.info("Stalled at buffer limit, retrying after 180-second break")
                        print("Stalled at buffer limit, retrying after 180-second break")
                        time.sleep(180)
                        stall_count = 0
                        ticker = 'XAUUSD=X'
                        logging.info("Switching to fallback ticker XAUUSD=X")
                        print("Switching to fallback ticker XAUUSD=X")
                        yf.Ticker(ticker).session.close()
                else:
                    stall_count = 0
                last_buffer_size = current_buffer
                time.sleep(60)
                continue

            # Reset stall counter
            stall_count = 0
            last_buffer_size = len(env.data_buffer)

            # Train base models
            state = env.get_state()
            if state is not None and len(env.data_buffer) > T+1:
                X = np.array([state[:-1]])
                y = scaled[3]
                for name, model in base_models.items():
                    pred = model(torch.FloatTensor(X).to(device))
                    loss = criterion(pred, torch.FloatTensor([[y]]).to(device))
                    base_optimizers[name].zero_grad()
                    loss.backward()
                    base_optimizers[name].step()
                    error = abs(pred.item() - y) / y
                    error_windows[name].append(error)

            # Generate trading signal
            if state is not None:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                preds = {name: model(state_tensor).item() for name, model in base_models.items()}
                errors = {name: np.mean(err) if err else 1.0 for name, err in error_windows.items()}
                inv_errors = np.array([1/(e+1e-6) for e in errors.values()])
                model_weights = inv_errors / inv_errors.sum()
                tech_indicators = compute_technical_indicators(env.data_buffer)
                state_vector = np.concatenate([state.flatten(), list(preds.values()), model_weights, tech_indicators])
                state_vector += np.random.normal(0, 0.05, state_vector.shape)
                with torch.no_grad():
                    logits = actor(torch.FloatTensor(state_vector).to(device))
                    logits += torch.normal(0, 1.0, logits.shape).to(device)
                    action_probs = sample_dirichlet_action_probs(num_base_models)
                signal = action_probs.mean().item()
                signal_std = action_probs.std().item()
                exploration_steps += 1
                trade_attempts += 1

                # Normalize signal
                signal = 0.05 + 0.9 * (signal - action_probs.min().item()) / (action_probs.max().item() - action_probs.min().item() + 1e-6)

                # Probabilistic trade trigger
                action_probs_random = np.array([0.45, 0.45, 0.1])  # BUY, SELL, HOLD
                action_taken = np.random.choice(['BUY', 'SELL', 'HOLD'], p=action_probs_random)
                qty = 0
                if action_taken == 'BUY' and trades_today < max_trades_per_day:
                    qty = max(1.0, min(env.balance / (current_price * 1), max_shares - env.shares, signal * 1.0))
                    if qty > 0:
                        env.balance -= qty * current_price * 1
                        env.shares += qty
                        trades_today += 1
                        env.trade_log.append(f"{timestamp_ist} BUY @ ${current_price:.2f} × {qty:.4f} contracts")
                elif action_taken == 'SELL' and env.shares > 0 and trades_today < max_trades_per_day:
                    sell_qty = max(1.0, min(env.shares, (1.0 - signal) * 1.0))
                    env.balance += sell_qty * current_price * 1
                    env.shares -= sell_qty
                    qty = sell_qty
                    trades_today += 1
                    env.trade_log.append(f"{timestamp_ist} SELL @ ${current_price:.2f} × {sell_qty:.4f} contracts")
                elif env.shares > max_shares:
                    sell_qty = env.shares - max_shares
                    env.balance += sell_qty * current_price * 1
                    env.shares -= sell_qty
                    action_taken = 'SELL'
                    qty = sell_qty
                    trades_today += 1
                    env.trade_log.append(f"{timestamp_ist} SELL @ ${current_price:.2f} × {sell_qty:.4f} contracts")

                # Calculate reward
                new_value = env.balance + env.shares * current_price
                returns = np.diff(env.total_assets[-100:]) / env.total_assets[-100:-1]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-6) if len(returns) > 1 else 0
                trade_size_factor = qty / max_shares if action_taken != 'HOLD' else 0
                reward = (new_value - env.total_assets[-1]) / env.total_assets[-1] + 0.001 * sharpe + 1.0 * (action_taken != prev_action) + 0.2 * trade_size_factor
                env.total_assets.append(new_value)

                # Store experience
                next_state = env.get_state()
                if next_state is not None:
                    next_vector = np.concatenate([next_state.flatten(), list(preds.values()), model_weights, tech_indicators])
                    replay.push(state_vector, action_probs.cpu().detach().numpy(), reward, next_vector)
                    if exploration_steps % 10 == 0:
                        save_replay_buffer(replay, replay_path)

                # Train DRL models
                if len(replay) >= batch_size and not drl_training_started:
                    logging.info(f"Replay buffer filled with {len(replay)} experiences, DRL training started")
                    print(f"Replay buffer filled with {len(replay)} experiences, DRL training started")
                    drl_training_started = True
                if len(replay) >= batch_size:
                    states, actions, rewards, next_states = replay.sample(batch_size)
                    with torch.no_grad():
                        target_actions = target_actor(next_states)
                        target_q = rewards + gamma * target_critic(next_states, target_actions)
                    current_q = critic(states, actions)
                    critic_loss = criterion(current_q, target_q)
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_optim.step()
                    action_probs = actor(states)
                    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-6), dim=-1).mean()
                    actor_loss = -critic(states, action_probs).mean() - 1.0 * entropy
                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_optim.step()
                    for t_param, param in zip(target_actor.parameters(), actor.parameters()):
                        t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)
                    for t_param, param in zip(target_critic.parameters(), critic.parameters()):
                        t_param.data.copy_(tau * param.data + (1 - tau) * t_param.data)

                # Log signal and action
                log_message = (
                    f"{timestamp_ist} | Price: ${current_price:.2f} | Signal: {signal:.3f} | "
                    f"Signal Std: {signal_std:.3f} | Action: {action_taken} | "
                    f"Portfolio: ${new_value:.2f} | Trades Today: {trades_today}/{max_trades_per_day}"
                )
                logging.info(log_message)
                print(log_message)
                logging.debug(f"Raw logits: {logits.cpu().detach().numpy().tolist()}")
                logging.debug(f"Raw action probabilities: {action_probs.cpu().detach().numpy().tolist()}")
                logging.debug(f"Trade attempt: {trade_attempts}")

                # Update previous action
                prev_action = action_taken

                # Save models periodically
                if datetime.now().minute % 5 == 0:
                    try:
                        for name, model in base_models.items():
                            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{name}.pth'))
                        torch.save(actor.state_dict(), os.path.join(checkpoint_dir, 'actor.pth'))
                        torch.save(critic.state_dict(), os.path.join(checkpoint_dir, 'critic.pth'))
                        logging.info("Saved model checkpoints")
                    except Exception as e:
                        logging.error(f"Error saving checkpoints: {e}")

            time.sleep(60)

        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            print(f"Error in main loop: {str(e)}")
            time.sleep(15 * (2 ** stall_count))

if __name__ == "__main__":
    real_time_training_loop()