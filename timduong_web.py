import random
import math
from simpleai.search import SearchProblem, astar
import numpy as np
from PIL import Image
import streamlit as st
import time

# Define the cost of moving around the map
COSTS = {
    "up": 1.0,
    "down": 1.0,
    "left": 1.0,
    "right": 1.0,
    "up left": 1.7,
    "up right": 1.7,
    "down left": 1.7,
    "down right": 1.7,
}

# Function to generate a random maze
def generate_random_maze(rows, cols):
    maze = [['#' if random.random() < 0.3 else ' ' for _ in range(cols)] for _ in range(rows)]
    maze[0][0] = maze[rows-1][cols-1] = ' '  # Ensure start and end points are empty
    return maze

# Generate a random maze
M, N, W = 30, 50, 21  # Example dimensions
MAP = generate_random_maze(M, N)

# Create colors for the map
mau_xanh = np.full((W, W, 3), (0, 255, 0), dtype=np.uint8)
mau_trang = np.full((W, W, 3), (255, 255, 255), dtype=np.uint8)
mau_khung = np.full((W, W, 3), (0, 255, 0), dtype=np.uint8)

# Function to create an image from the maze
def create_maze_image(map_data):
    rows, cols = len(map_data), len(map_data[0])
    image = np.ones(((rows + 2) * W, (cols + 2) * W, 3), dtype=np.uint8) * 255

    # Add blue border
    image[:W, :] = (0, 0, 255)  # top border
    image[-W:, :] = (0, 0, 255)  # bottom border
    image[:, :W] = (0, 0, 255)  # left border
    image[:, -W:] = (0, 0, 255)  # right border

    for x in range(rows):
        for y in range(cols):
            if map_data[x][y] == '#':
                image[(x + 1) * W:(x + 2) * W, (y + 1) * W:(y + 2) * W] = mau_xanh
            else:
                image[(x + 1) * W:(x + 2) * W, (y + 1) * W:(y + 2) * W] = mau_trang

    return image

# Create the initial maze image
image = create_maze_image(MAP)
pil_image = Image.fromarray(image)

# Class to solve the maze
class MazeSolver(SearchProblem):
    def __init__(self, board):
        self.board = board
        self.goal = None
        self.initial = (0, 0)

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == 'o':
                    self.initial = (x, y)
                elif self.board[y][x].lower() == 'x':
                    self.goal = (x, y)  # Assign the goal coordinates here

        super().__init__(initial_state=self.initial)

    def actions(self, state):
        actions = []
        for action in COSTS:
            newx, newy = self.result(state, action)
            if self.is_valid_move(state, (newx, newy), action):
                actions.append(action)
        return actions

    def result(self, state, action):
        x, y = state
        if 'up' in action:
            y -= 1
        if 'down' in action:
            y += 1
        if 'left' in action:
            x -= 1
        if 'right' in action:
            x += 1
        return x, y

    def is_valid_move(self, state, new_state, action):
        x, y = state
        newx, newy = new_state

        # Ensure new state is within bounds and not a wall
        if not (0 <= newx < len(self.board[0]) and 0 <= newy < len(self.board)):
            return False
        if self.board[newy][newx] == '#':
            return False

        # Prevent diagonal moves through walls
        if action in ("up left", "up right", "down left", "down right"):
            if action == "up left" and (self.board[y-1][x] == '#' or self.board[y][x-1] == '#'):
                return False
            if action == "up right" and (self.board[y-1][x] == '#' or self.board[y][x+1] == '#'):
                return False
            if action == "down left" and (self.board[y+1][x] == '#' or self.board[y][x-1] == '#'):
                return False
            if action == "down right" and             (self.board[y+1][x] == '#' or self.board[y][x+1] == '#'):
                return False

        return True

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return COSTS[action]

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

# Streamlit app
def main():
    st.title("Maze Solver")

    # Display maze image
    st.image(pil_image, caption='Initial Maze', use_column_width=True)

    # Initialize variables for start and end points
    start_point = None
    end_point = None

    # User interaction to place start and end points
    st.write("Click on the maze to select the start and end points.")

    # Function to handle click events on the maze image
    click_coordinates = st.image_click(pil_image)

    # Check if click coordinates are available
    if click_coordinates:
        x, y = click_coordinates["x"], click_coordinates["y"]
        row = int(y // W) - 1  # Convert click coordinates to maze coordinates
        col = int(x // W) - 1

        if st.button("Set Start Point"):
            start_point = (col, row)
        elif st.button("Set End Point"):
            end_point = (col, row)

    if start_point:
        x, y = start_point
        MAP[y][x] = 'o'
        st.image(create_maze_image(MAP), caption='Maze with Start Point', use_column_width=True)
    if end_point:
        x, y = end_point
        MAP[y][x] = 'x'
        st.image(create_maze_image(MAP), caption='Maze with Start and End Points', use_column_width=True)

    # Solve maze button
    if start_point and end_point:
        if st.button("Solve Maze"):
            st.write("Solving the maze...")
            problem = MazeSolver(MAP)
            result = astar(problem, graph_search=True)
            if result is None or len(result.path()) == 0:
                st.error("No path found!")
                return

            # Display solved path
            solved_path = [x[1] for x in result.path()]
            for i, (x, y) in enumerate(solved_path):
                MAP[y][x] = 's'  # Mark path with 's'
                st.image(create_maze_image(MAP), caption=f'Step {i+1} - Solved Path', use_column_width=True)

                # Delay between steps
                time.sleep(0.1)

            # Reset maze after solving
            MAP = generate_random_maze(M, N)
            MAP[start_point[1]][start_point[0]] = 'o'
            MAP[end_point[1]][end_point[0]] = 'x'

if __name__ == "__main__":
    main()
