import numpy as np
import pylab as pl
import networkx as nx

# 1. Define the Graph (Locations and Paths)
# CHANGE 1: Added a new edge (2, 7) to create a shortcut to the new goal
edges = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), 
         (1, 3), (9, 10), (2, 4), (0, 6), (6, 7),
         (8, 9), (7, 8), (1, 7), (3, 9), (2, 7)]

# CHANGE 2: Changed the Goal Node from 10 to 7
goal = 7 

# 2. Visualize the Graph
G = nx.Graph()
G.add_edges_from(edges)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos)
# CHANGE 3: Added a Title
pl.title("Graph Environment (Goal: Node 7)") 
pl.show()

# 3. Initialize Reward Matrix (M)
MATRIX_SIZE = 11
M = np.matrix(np.ones(shape =(MATRIX_SIZE, MATRIX_SIZE)))
M *= -1 # Initialize all paths as blocked (-1)

# Assign rewards to paths
for point in edges:
    # If path leads to goal, give 100 reward
    if point[1] == goal:
        M[point] = 100
    else:
        M[point] = 0

    if point[0] == goal:
        M[point[::-1]] = 100
    else:
        M[point[::-1]]= 0

# Goal to itself is max reward
M[goal, goal]= 100

# 4. Q-Learning Training
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))

gamma = 0.75 # Discount factor (future rewards are worth less)
initial_state = 1

def available_actions(state):
    current_state_row = M[state, ]
    available_action = np.where(current_state_row >= 0)[1]
    return available_action

available_action = available_actions(initial_state)

def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_action, 1))
    return next_action

action = sample_next_action(available_action)

def update(current_state, action, gamma):
  max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]
  if max_index.shape[0] > 1:
      max_index = int(np.random.choice(max_index, size = 1))
  else:
      max_index = int(max_index)
  max_value = Q[action, max_index]
  Q[current_state, action] = M[current_state, action] + gamma * max_value
  if (np.max(Q) > 0):
    return(np.sum(Q / np.max(Q)*100))
  else:
    return (0)

# Train for 1000 iterations
print("Training Agent...")
scores = []
for i in range(1000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_action = available_actions(current_state)
    action = sample_next_action(available_action)
    score = update(current_state, action, gamma)
    scores.append(score)

print("Training Complete!")

# 5. Testing the Learned Path
# Start from Node 0 and try to find the Goal
current_state = 0
steps = [current_state]

while current_state != goal:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size = 1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("\n------------------------------------------------")
print(f"Most efficient path to Goal (Node {goal}):")
print(steps)
print("------------------------------------------------")

# Plot Learning Curve
pl.plot(scores)
pl.xlabel('No of iterations')
pl.ylabel('Reward gained')
pl.title("Agent Learning Curve")
pl.show()