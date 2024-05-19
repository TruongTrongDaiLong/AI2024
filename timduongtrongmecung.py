import random
import math
from simpleai.search import SearchProblem, astar
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox  # Import the messagebox module
from PIL import Image, ImageTk
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
color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(color_converted)

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
            if action == "down right" and (self.board[y+1][x] == '#' or self.board[y][x+1] == '#'):
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

# Function to create a rounded button
def create_rounded_button(canvas, x, y, width, height, radius, text, command, bg="blue", fg="white", font=("Arial", 12, "bold")):
    if radius > min(width, height) // 2:
        radius = min(width, height) // 2

    points = [
        x + radius, y,
        x + width - radius, y,
        x + width - radius, y + radius,
        x + width, y + radius,
        x + width, y + height - radius,
        x + width - radius, y + height - radius,
        x + width - radius, y + height,
        x + radius, y + height,
        x + radius, y + height - radius,
        x, y + height - radius,
        x, y + radius,
        x + radius, y + radius
    ]

    button = canvas.create_polygon(points, smooth=True, outline=bg, fill=bg)
    text_item = canvas.create_text((x + width // 2, y + height // 2), text=text, fill=fg, font=font)

    def on_click(event):
        command()

    canvas.tag_bind(button, "<Button-1>", on_click)
    canvas.tag_bind(text_item, "<Button-1>", on_click)

    return button

# GUI application
class MazeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tìm Đường Trong Mê Cung")
        self.dem = 0

        self.canvas = tk.Canvas(self, width=(N + 2) * W, height=(M + 2) * W, relief=tk.SUNKEN, border=1)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.canvas.bind("<Button-1>", self.handle_mouse)

        self.menu_canvas = tk.Canvas(self, width=150, height=200)
        self.create_buttons()

        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        self.menu_canvas.grid(row=0, column=1, padx=5, pady=5, sticky=tk.N)

    def create_buttons(self):
        create_rounded_button(self.menu_canvas, 10, 10, 120, 40, 20, "Start", self.start_solver)
        create_rounded_button(self.menu_canvas, 10, 60, 120, 40, 20, "Reset", self.reset_solver)
        create_rounded_button(self.menu_canvas, 10, 110, 120, 40, 20, "Generate", self.generate_new_maze)

    def handle_mouse(self, event):
        if self.dem == 0:
            x, y = (event.x // W) - 1, (event.y // W) - 1
            if 0 <= x < N and 0 <= y < M and MAP[y][x] != '#':
                MAP[y][x] = 'o'
                self.canvas.create_oval((x + 1) * W + 2, (y + 1) * W + 2, (x + 2) * W - 2, (y + 2) * W - 2,
                                        outline='#FF0000', fill='#FF0000')
                self.dem += 1
        elif self.dem == 1:
            x, y = (event.x // W) - 1, (event.y // W) - 1
            if 0 <= x < N and 0 <= y < M and MAP[y][x] != '#':
                MAP[y][x] = 'x'
                self.canvas.create_rectangle((x + 1) * W + 2, (y + 1) * W + 2, (x + 2) * W - 2, (y + 2) * W - 2,
                                             outline='#FF0000', fill='#FF0000')
                self.dem += 1

    def start_solver(self):
        problem = MazeSolver(MAP)
        result = astar(problem, graph_search=True)
        if result is None or len(result.path()) == 0:
            messagebox.showinfo("Thông báo", "Không tìm được đường đi!")
            return

        path = [x[1] for x in result.path()]

        for i in range(1, len(path)-1):
            x, y = path[i]
            self.canvas.create_rectangle((x + 1) * W + 2, (y + 1) * W + 2, (x + 2) * W - 2, (y + 2) * W - 2,
                                         outline='#FFFF00', fill='#FFFF00')
            time.sleep(0.1)
            self.canvas.update()

    def reset_solver(self):
        self.canvas.delete(tk.ALL)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.dem = 0
        for x in range(M):
            for y in range(N):
                if MAP[x][y] in ('o', 'x'):
                    MAP[x][y] = ' '

    def generate_new_maze(self):
        global MAP, image, color_converted, pil_image
        MAP = generate_random_maze(M, N)
        image = create_maze_image(MAP)
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.reset_solver()

if __name__ == "__main__":
    app = MazeApp()
    app.mainloop()
