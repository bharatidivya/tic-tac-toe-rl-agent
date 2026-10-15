import streamlit as st
import numpy as np
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="RL Tic-Tac-Toe Arena",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom Styling for Interactive Look ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        margin-bottom: 30px;
    }
    .stButton>button {
        font-size: 32px !important;
        font-weight: bold !important;
        height: 90px !important;
        border-radius: 12px !important;
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        border-color: #FF4B4B !important;
        color: #FF4B4B !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<p class="main-title">🎮 RL Tic-Tac-Toe Arena</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Human vs. Q-Agent: A Reinforcement Learning Game Arena</p>', unsafe_allow_html=True)

# --- 1. Tic-Tac-Toe Environment ---
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.winner = None
        self.game_over = False

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.winner = None
        self.game_over = False
        return str(self.board.flatten())

    def available_actions(self):
        return [(r, c) for r in range(3) for c in range(3) if self.board[r, c] == 0]

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
            self.winner = 0 
            self.game_over = True

# --- 2. Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.q_table = {}
        self.alpha = alpha     
        self.gamma = gamma     

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions, epsilon=0.0):
        if random.uniform(0, 1) < epsilon:
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

# --- 3. Background Training Cache ---
@st.cache_resource
def train_agent():
    env = TicTacToe()
    agent = QLearningAgent()
    epsilon = 1.0
    
    for episode in range(35000):
        state = env.reset()
        while not env.game_over:
            actions = env.available_actions()
            action = agent.choose_action(state, actions, epsilon)
            
            env.board[action[0], action[1]] = 1
            env.check_game_over()
            next_state = str(env.board.flatten())
            
            if env.game_over:
                if env.winner == 1:
                    agent.update_q(state, action, 1.0, next_state, [])
                elif env.winner == 0:
                    agent.update_q(state, action, 0.5, next_state, [])
                break
                
            opp_actions = env.available_actions()
            if opp_actions:
                opp_action = random.choice(opp_actions)
                env.board[opp_action[0], opp_action[1]] = -1
                env.check_game_over()
                next_state = str(env.board.flatten())
                
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
            
        if epsilon > 0.01:
            epsilon *= 0.999
            
    return agent

with st.spinner("🧠 Initializing and training the RL Agent in background..."):
    trained_agent = train_agent()

# --- 4. Session State Management ---
if "board" not in st.session_state:
    st.session_state.board = np.zeros((3, 3), dtype=int)
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None
if "turn" not in st.session_state:
    st.session_state.turn = -1  # -1: Human (⭕), 1: AI (❌)
if "scores" not in st.session_state:
    st.session_state.scores = {"Human": 0, "AI": 0, "Ties": 0}

# --- 5. Interactive Sidebar Controls ---
st.sidebar.header("⚙️ Game Settings & Stats")

# AI Difficulty Selector
ai_difficulty = st.sidebar.selectbox(
    "Choose AI Difficulty", 
    ["Master (Optimal)", "Casual (Exploratory)"],
    help="Master uses 0% random exploration. Casual introduces random moves."
)
epsilon_val = 0.0 if ai_difficulty == "Master (Optimal)" else 0.2

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Live Scoreboard")
st.sidebar.markdown(f"**👤 You (⭕):** {st.session_state.scores['Human']}")
st.sidebar.markdown(f"**🤖 AI (❌):** {st.session_state.scores['AI']}")
st.sidebar.markdown(f"**🤝 Ties:** {st.session_state.scores['Ties']}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Scoreboard"):
    st.session_state.scores = {"Human": 0, "AI": 0, "Ties": 0}
    st.rerun()

# --- 6. AI Turn Logic ---
def execute_ai_turn():
    if st.session_state.game_over:
        return
    state = str(st.session_state.board.flatten())
    actions = [(r, c) for r in range(3) for c in range(3) if st.session_state.board[r, c] == 0]
    if actions:
        action = trained_agent.choose_action(state, actions, epsilon=epsilon_val)
        st.session_state.board[action[0], action[1]] = 1
        
        env_temp = TicTacToe()
        env_temp.board = st.session_state.board.copy()
        env_temp.check_game_over()
        
        if env_temp.game_over:
            st.session_state.game_over = True
            st.session_state.winner = env_temp.winner
            if env_temp.winner == 1:
                st.session_state.scores["AI"] += 1
            elif env_temp.winner == 0:
                st.session_state.scores["Ties"] += 1
        else:
            st.session_state.turn = -1

# --- 7. Main Dashboard Interface ---

# Turn Banner Notification
if not st.session_state.game_over:
    if st.session_state.turn == -1:
        st.info("👉 **Your Turn!** Click any available square to place your ⭕.")
    else:
        st.info("⏳ **AI is calculating optimal move...**")
else:
    if st.session_state.winner == 1:
        st.error("🎉 **Round Over: AI Agent (❌) Wins!**")
    elif st.session_state.winner == -1:
        st.success("🏆 **Round Over: Amazing! You (⭕) Beat the AI!**")
    else:
        st.warning("🤝 **Round Over: It's a Tie!**")

# Render the 3x3 Grid
grid_container = st.container()
with grid_container:
    for r in range(3):
        cols = st.columns(3)
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
                        
                        env_temp = TicTacToe()
                        env_temp.board = st.session_state.board.copy()
                        env_temp.check_game_over()
                        
                        if env_temp.game_over:
                            st.session_state.game_over = True
                            st.session_state.winner = env_temp.winner
                            if env_temp.winner == -1:
                                st.session_state.scores["Human"] += 1
                            elif env_temp.winner == 0:
                                st.session_state.scores["Ties"] += 1
                        else:
                            st.session_state.turn = 1
                            execute_ai_turn()
                        st.rerun()

# Bottom Action Button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Play Next Round", use_container_width=True):
        st.session_state.board = np.zeros((3, 3), dtype=int)
        st.session_state.game_over = False
        st.session_state.winner = None
        st.session_state.turn = -1
        st.rerun()
