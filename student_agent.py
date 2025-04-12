# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
from collections import defaultdict

COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)
    
def get_afterstate(env, state, action):
    tmp_env = copy.deepcopy(env)
    tmp_env.board = state.copy()
    tmp_env.score = 0
    if action == 0:
        moved = tmp_env.move_up()
    elif action == 1:
        moved = tmp_env.move_down()
    elif action == 2:
        moved = tmp_env.move_left()
    elif action == 3:
        moved = tmp_env.move_right()
    return tmp_env.board, tmp_env.score
    
    
class MCTNode:
    def __init__(self, state, score, random_node = False, n_sample = 10, parent=None, action=None):
        self.state = state
        self.score = score
        self.random_node = random_node  # the state is the one before adding the new random tile if random_node
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        
        env = Game2048Env()
        env.board = state.copy()
        env.score = score
        self.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        
        if self.random_node:
            for i in range(n_sample):
                sim_env = copy.deepcopy(env)
                sim_env.board = state.copy()
                sim_env.score = score
                sim_env.add_random_tile()
                self.children[i] = MCTNode(sim_env.board.copy(), sim_env.score, parent=self, action=None)
                
    def fully_expanded(self):
        return len(self.untried_actions) == 0

class MCTS:
    def __init__(self, env, approximator, iterations=300, exploration_constant=1.41, rollout_depth=0, n_sim=10):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.n_sim = n_sim

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # Q + c * sqrt(log(parent_visits)/child_visits)
        cursor = node
        while cursor.children:
            if cursor.random_node:
                cursor = cursor.children[random.randint(0, len(cursor.children) - 1)]
            else:
                if cursor.fully_expanded():
                    ucb = np.zeros(4)
                    ucb.fill(-np.inf)
                    for action, child in cursor.children.items():
                        Q = child.total_reward / child.visits
                        ucb[action] = Q + self.c * np.sqrt(np.log(cursor.visits) / child.visits)
                    action_chosen = np.argmax(ucb)
                else:
                    action_chosen = cursor.untried_actions.pop(random.randint(0, len(cursor.untried_actions) - 1))
                cursor = cursor.children[action_chosen]
        return cursor

    def rollout(self, root, depth):
        total_score = 0
        for i in range(self.n_sim):
            node = root.children[i]
            sim_env = self.create_env_from_state(node.state, node.score)
            done = False
            score = node.score
            afterstate = root.state.copy()
            for __ in range(depth):
                legal_moves = [action for action in [0, 1, 2, 3] if sim_env.is_move_legal(action)]
                if not legal_moves:
                    done = True
                    break
                action = random.choice(legal_moves)
                afterstate, __ = get_afterstate(sim_env, sim_env.board, action)
                state, score, done, __ = sim_env.step(action)
                
            score += self.approximator.value(afterstate)
            total_score += score
            
        return total_score / self.n_sim

    def backpropagate(self, node, reward):
        cursor = node
        while cursor.parent is not None:
            cursor.visits += 1
            cursor.total_reward += reward - cursor.score
            cursor = cursor.parent
        cursor.visits += 1
        cursor.total_reward += reward - cursor.score
        return
    
    def expansion(self, node, sim_env):
        for action in node.untried_actions:
            state, score = get_afterstate(sim_env, node.state, action)
            node.children[action] = MCTNode(state.copy(), score, random_node=True, n_sample=self.n_sim, parent = node, action = action)
        action = node.untried_actions.pop(random.randint(0, len(node.untried_actions) - 1))
        node = node.children[action]
        return node

    def run_simulation(self, root):
        node = root

        node = self.select_child(node)
        sim_env = self.create_env_from_state(node.state, node.score)
        
        if not sim_env.is_game_over():    
            if node == root or node.visits > 4:
                node = self.expansion(node, sim_env)
            else:
                node = node.parent
                
            rollout_reward = self.rollout(node, self.rollout_depth)
        else:
            rollout_reward = node.score
            
        self.backpropagate(node, rollout_reward)
    
    def best_action(self, root):
        Q = np.zeros(4)
        Q.fill(-np.inf)
        for action, child in root.children.items():
            Q[action] = child.total_reward / child.visits
        best_action = np.argmax(Q)
        return best_action, Q

def rot90(pattern):
      return [(x, 3 - y) for (y, x) in pattern]

def ref(pattern):
  return [(y, 3 - x) for (y, x) in pattern]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)

    def generate_symmetries(self, pattern):
        syms = []
        for __ in range(4):
            syms.append(pattern)
            syms.append(ref(pattern))
            pattern = rot90(pattern)
        return syms

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple(self.tile_to_index(board[y][x]) for (y, x) in coords)

    def value(self, board):
        value = 0
        for (index, pattern) in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            value += self.weights[index // 8][feature]
        return value

    def update(self, board, delta, alpha):
        weight_update = alpha * delta
        for (index, pattern) in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            self.weights[index // 8][feature] += weight_update
        return

    def simulate_action(self, env, state, action, gamma):
        sim_env = copy.deepcopy(env)
        sim_env.board = state.copy()
        sim_env.score = 0
        afterstate, reward = get_afterstate(sim_env, state, action)
        value = self.value(afterstate)
        return reward + value * gamma

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99, epsilon=0.1):
    final_scores = []
    success_flags = []

    try:
        if path is not None:
            with open(path, "rb") as f:
                approximator.weights = pickle.load(f)
                print("Weights loaded from", path)
        
        for episode in range(num_episodes):
            state = env.reset().copy()
            trajectory = []
            previous_score = 0
            beforestate = np.zeros((4, 4), dtype=int)
            done = False
            max_tile = np.max(state)

            while not done:
                legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                if not legal_moves:
                    break
                Q_values = np.zeros(4)
                Q_values.fill(-np.inf)
                for action in legal_moves:
                    Q_values[action] = approximator.simulate_action(env, state, action, gamma)

                action = np.argmax(Q_values)

                # if np.random.rand() < epsilon:
                #     action = np.random.choice(legal_moves)

                afterstate, __ = get_afterstate(env, state, action)
                next_state, new_score, done, _ = env.step(action)
                next_state = next_state.copy()
                incremental_reward = new_score - previous_score
                previous_score = new_score
                max_tile = max(max_tile, np.max(next_state))

                trajectory.append((beforestate, action, incremental_reward, afterstate))

                state = next_state
                beforestate = afterstate
                
            # # trajectory = [(state, action, reward, next_state), ... ]
            # for i in reversed(range(len(trajectory))):
            #     value = approximator.value(trajectory[i][0])
            #     end = i + n_steps if i + n_steps < len(trajectory) else len(trajectory) - 1
            #     target = approximator.value(trajectory[end][3])
            #     for j in reversed(range(i, end + 1)):
            #         target = trajectory[j][2] + gamma * target
            #     delta = target - value
            #     approximator.update(trajectory[i][0], delta, alpha)
                
            for (beforestate, action, reward, afterstate) in reversed(trajectory):
                value = approximator.value(beforestate)
                target = reward + gamma * approximator.value(afterstate)
                delta = target - value
                approximator.update(beforestate, delta, alpha)

            final_scores.append(env.score)
            success_flags.append(1 if max_tile >= 2048 else 0)

            if (episode + 1) % print_interval == 0:
                avg_score = np.mean(final_scores[-print_interval:])
                success_rate = np.sum(success_flags[-print_interval:]) / 100
                print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
    except KeyboardInterrupt:
        print(f"Training interrupted at episode {episode + 1}")

    return final_scores

print_interval = 100
avg_interval = 100
n_episodes = 50000

path = "weights.pkl"

patterns = [[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)],
            [(1, 0), (1, 1), (2, 0), (2, 1), (1, 2), (1, 3)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
            [(0, 0), (0, 1), (1, 1), {1, 2}, (1, 3), (2, 2)],
            [(1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0)],
            [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
            [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]]

"""
the patterns chosen:
1.      2.      3.      4.
■■■■    □□□□    ■■■□    ■■□□
■■□□    ■■■■    ■■■□    □■■■
□□□□    ■■□□    □□□□    □□■□
□□□□    □□□□    □□□□    □□□□

5.      6.      7.      8.
□□□□    ■■□□    ■■□□    ■■■□
■□■□    □■□□    □■□□    ■□■□
■■■□    □■□□    ■■□□    □□■□
■□□□    □■■□    □■□□    □□□□

"""

approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()
if path is not None:
    with open(path, "rb") as f:
        approximator.weights = pickle.load(f)
        print("Weights loaded from", path)

mcts = MCTS(env, approximator)

def get_action(state, score):
    root = MCTNode(state, score)

    for _ in range(mcts.iterations):
        mcts.run_simulation(root)

    best_action, Q = mcts.best_action(root)
    return best_action

def cal_avg_score(scores):
    avg_scores = []
    for i in range(len(scores) - avg_interval + 1):
        avg = 0
        for j in range(avg_interval):
            avg += scores[i + j]
        avg /= avg_interval
        avg_scores.append(avg)
    return avg_scores

if __name__ == '__main__':
    scores = td_learning(env, approximator, num_episodes=n_episodes, alpha=0.005, gamma=0.99, epsilon=0.01)    
    avg_scores = cal_avg_score(scores)
    plt.plot(scores, label = 'return', color = 'lightblue')
    plt.plot([i + avg_interval - 1 for i in range(len(avg_scores))], avg_scores, label = 'avg_return')
    plt.legend()
    plt.xlabel('num_episodes')
    plt.savefig('learning_fig.png')
    plt.show()
    
    if(input("Save weights? (y/n): ").lower() == "y"):
        with open("weights.pkl", "wb") as f:
            pickle.dump(approximator.weights, f)
            print("Weights saved to weights.pkl")