import streamlit as st
import numpy as np
import random

# Page configuration
st.set_page_config(page_title="RL Tic-Tac-Toe Dashboard", page_icon="🤖", layout="centered")

st.title("🤖 RL-Powered Tic-Tac-Toe Dashboard")
st.markdown("Play against a **Q-Learning Agent** trained using Markov Decision Processes and reinforcement learning principles (SPPU ML Unit V)[cite: 20]!")

# --- 1. Tic-Tac-Toe Environment ---
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int) # 0: empty, 1: X (Agent), -1: O (Opponent)
        self.winner = None
        self.game_over = False

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.winner = None
        self.game_over = False
        return self.get_state()

    def get_state(self):
        return str(self.board.flatten())

    def available_actions(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

    def make_move(self, action, player):
        r, c = action
        if self.board[r, c] == 0:
            self.board[r, c] = player
            self.check_game_over()
            return True
        return False

    def check_game_over(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                self.winner = self.board[i, 0]
                self.game_over = True
                return
            if abs(sum(self.board[:, i])) == 3:
                self.winner = self.board[0, i]
                self.game_over = True
                return
        if abs(self.board[0,0] + self.board[1,1] + self.board[2,2]) == 3:
            self.winner = self.board[1,1]
            self.game_over = True
            return
        if abs(self.board[0,2] + self.board[1,1] + self.board[2,0]) == 3:
            self.winner = self.board[1,1]
            self.game_over = True
            return
        if len(self.available_actions()) == 0:
            self.winner = 0 # Tie
            self.game_over = True

# --- 2. Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.05):
        self.q_table = {}
        self.alpha = alpha     
        self.gamma = gamma     
        self.epsilon = epsilon # Low epsilon for playing (exploitation mode)

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [available_actions[i] for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, state, action, reward, next_state, next_available):
        current_q = self.get_q(state, action)
        max_next_q = max([self.get_q(next_state, a) for a in next_available]) if next_available else 0.0
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

# --- 3. Cache the Training Process ---
@st.cache_resource
def train_agent():
    env = TicTacToe()
    agent = QLearningAgent()
    episodes = 30000
    
    for episode in range(episodes):
        state = env.reset()
        while not env.game_over:
            actions = env.available_actions()
            action = agent.choose_action(state, actions)
            
            env.make_move(action, 1)
            next_state = env.get_state()
            
            if env.game_over:
                if env.winner == 1:
                    agent.update_q(state, action, 1.0, next_state, [])
                elif env.winner == 0:
                    agent.update_q(state, action, 0.5, next_state, [])
                break
                
            # Opponent random move during training
            opp_actions = env.available_actions()
            if opp_actions:
                opp_action = random.choice(opp_actions)
                env.make_move(opp_action, -1)
                next_state = env.get_state()
                
                if env.game_over:
                    if env.winner == -1:
                        agent.update_q(state, action, -1.0, next_state, [])
                        break
                    elif env.winner == 0:
                        agent.update_q(state, action, 0.5, next_state, [])
                        break
                        
            next_actions = env.available_actions()
            agent.update_q(state, action, 0.0, next_state, next_actions)
            state = next_state
            
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.999
            
    return agent

# Load or train the agent once
with st.spinner("Training RL Agent in the background... Please wait."):
    trained_agent = train_agent()

# --- 4. Streamlit Game State Session ---
if "board" not in st.session_state:
    st.session_state.board = np.zeros((3, 3), dtype=int)
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None
if "turn" not in st.session_state:
    st.session_state.turn = -1  # -1 for Human (O), 1 for AI (X)

def ai_turn():
    if st.session_state.game_over:
        return
    state = str(st.session_state.board.flatten())
    actions = [(r, c) for r in range(3) for c in range(3) if st.session_state.board[r, c] == 0]
    if actions:
        action = trained_agent.choose_action(state, actions)
        st.session_state.board[action[0], action[1]] = 1
        
        # Check game result after AI move
        env_temp = TicTacToe()
        env_temp.board = st.session_state.board.copy()
        env_temp.check_game_over()
        if env_temp.game_over:
            st.session_state.game_over = True
            st.session_state.winner = env_temp.winner
        else:
            st.session_state.turn = -1

# --- 5. Sidebar Layout ---
st.sidebar.header("Game Controls")
if st.sidebar.button("🔄 Reset Game"):
    st.session_state.board = np.zeros((3, 3), dtype=int)
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.turn = -1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("**Legend:**\n* ❌ **AI Agent** (Player 1)\n* ⭕ **You** (Player -1)")

# --- 6. Main Interactive Board UI ---
st.subheader("Interactive Match Grid")
cols = st.columns(3)

for r in range(3):
    for c in range(3):
        with cols[c]:
            val = st.session_state.board[r, c]
            label = " "
            if val == 1:
                label = "❌"
            elif val == -1:
                label = "⭕"
                
            if st.button(label, key=f"cell_{r}_{c}", use_container_width=True):
                if not st.session_state.game_over and st.session_state.turn == -1 and val == 0:
                    # Human Move
                    st.session_state.board[r, c] = -1
                    
                    # Check game result after human move
                    env_temp = TicTacToe()
                    env_temp.board = st.session_state.board.copy()
                    env_temp.check_game_over()
                    
                    if env_temp.game_over:
                        st.session_state.game_over = True
                        st.session_state.winner = env_temp.winner
                    else:
                        st.session_state.turn = 1
                        ai_turn()
                    st.rerun()

# --- 7. Status Banner ---
st.markdown("---")
if st.session_state.game_over:
    if st.session_state.winner == 1:
        st.error("🎉 **Game Over! The AI Agent (❌) Wins!**")
    elif st.session_state.winner == -1:
        st.success("🏆 **Congratulations! You (⭕) Beat the AI!**")
    else:
        st.warning("🤝 **It's a Tie! Well played.**")
else:
    if st.session_state.turn == -1:
        st.info("👉 **Your turn!** Click an empty square to make your move (⭕).")
    else:
        st.info("⏳ **AI is thinking...**")
