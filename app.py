import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(page_title="RL Tic-Tac-Toe Dashboard", page_icon="🤖", layout="centered")

st.title("🤖 RL-Powered Tic-Tac-Toe Dashboard")
st.markdown("Play against a **Q-Learning Agent** trained using Markov Decision Processes and reinforcement learning principles!")

# Initialize Session State variables for the game
if "board" not in st.session_state:
    st.session_state.board = np.zeros((3, 3), dtype=int)
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None
if "turn" not in st.session_state:
    st.session_state.turn = -1  # -1 for Human (O), 1 for AI (X)

# A mock/trained Q-table agent integration 
# (In your full project, you can load your pre-trained q_table dictionary here)
if "q_table" not in st.session_state:
    st.session_state.q_table = {}

def get_state(board):
    return str(board.flatten())

def available_actions(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r, c] == 0]

def check_game_over(board):
    for i in range(3):
        if abs(sum(board[i, :])) == 3:
            return board[i, 0]
        if abs(sum(board[:, i])) == 3:
            return board[0, i]
    if abs(board[0,0] + board[1,1] + board[2,2]) == 3:
        return board[1,1]
    if abs(board[0,2] + board[1,1] + board[2,0]) == 3:
        return board[1,1]
    if len(available_actions(board)) == 0:
        return 0 # Tie
    return None

# AI Move Logic using Q-table (falls back to random if state is new)
def ai_move():
    if st.session_state.game_over:
        return
    state = get_state(st.session_state.board)
    actions = available_actions(st.session_state.board)
    if actions:
        # Choose best action from Q-table or random explore
        q_values = [st.session_state.q_table.get((state, a), 0.0) for a in actions]
        max_q = max(q_values)
        best_actions = [actions[i] for i, q in enumerate(q_values) if q == max_q]
        chosen_action = np.random.choice(len(best_actions))
        r, c = best_actions[chosen_action]
        
        st.session_state.board[r, c] = 1  # AI is 1 (X)
        
        # Check if AI won
        res = check_game_over(st.session_state.board)
        if res is not None:
            st.session_state.game_over = True
            st.session_state.winner = res
        else:
            st.session_state.turn = -1 # Switch back to human

# Sidebar controls
st.sidebar.header("Game Settings")
if st.sidebar.button("🔄 Restart Game"):
    st.session_state.board = np.zeros((3, 3), dtype=int)
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.turn = -1
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("**Legend:**\n* ❌ **AI Agent** (Player 1)\n* ⭕ **You** (Player -1)")

# Main Game Grid UI
st.subheader("Game Board")
cols = st.columns(3)

for r in range(3):
    for c in range(3):
        with cols[c]:
            val = st.session_state.board[r, c]
            # Assign labels for display
            label = " "
            if val == 1:
                label = "❌"
            elif val == -1:
                label = "⭕"
                
            # Render grid button
            if st.button(label, key=f"btn_{r}_{c}", use_container_width=True):
                if not st.session_state.game_over and st.session_state.turn == -1 and val == 0:
                    # Human Move
                    st.session_state.board[r, c] = -1
                    
                    # Check if human won
                    res = check_game_over(st.session_state.board)
                    if res is not None:
                        st.session_state.game_over = True
                        st.session_state.winner = res
                    else:
                        st.session_state.turn = 1
                        ai_move()
                    st.rerun()

# Game Status Display
st.markdown("---")
if st.session_state.game_over:
    if st.session_state.winner == 1:
        st.error("🎉 **Game Over! The AI Agent (❌) Wins!**")
    elif st.session_state.winner == -1:
        st.success("🏆 **Congratulations! You (⭕) Win!**")
    else:
        st.warning("🤝 **It's a Tie!**")
else:
    if st.session_state.turn == -1:
        st.info("Your turn! Click an empty square to place your ⭕.")
    else:
        st.info("AI is thinking...")
