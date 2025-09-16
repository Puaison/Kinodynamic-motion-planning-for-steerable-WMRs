import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class OccupancyGridMap:
    def __init__(
        self,
        width,
        height,
        obstacle_positions=None,
        inflate=False,
        inflation_radius=1,
        obstacle_probability=0,
        resolution = 1, # resolution: number of meters occupied by a cell (e.g resolution=0.1 means that each cell occupies 0.1m^2 )
        seed = 1,
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.obstacle_positions = (
            obstacle_positions if obstacle_positions is not None else []
        )
        self.occupancy_grid = np.zeros((height, width), dtype=np.uint8)
        
        self.rng = np.random.default_rng(seed)

        # Set obstacle positions if given
        if obstacle_positions:
            for pos in obstacle_positions:
                x, y = pos
                if 0 <= x < height and 0 <= y < width:
                    self.occupancy_grid[x, y] = 1

        # Add random obstacles based on the probability
        if obstacle_probability > 0:
            self.add_random_obstacles(obstacle_probability)

        # Inflate obstacles if inflate is True
        if inflate:
            self.inflate_obstacles(inflation_radius)

    def inflate_obstacles(self, max_inflation_radius):
        """
        Each obstacle is inflated of a radius given by 'infaltion_radius'
        """
        inflated_grid = np.zeros((self.height, self.width), dtype=np.uint8)
        for x, y in self.obstacle_positions:
            inflation_radius = self.rng.integers(low=0, high=max_inflation_radius)

            for i in range(-inflation_radius, inflation_radius + 1):
                for j in range(-inflation_radius, inflation_radius + 1):
                    nx, ny = x + i, y + j
                    if 0 <= nx < self.height and 0 <= ny < self.width:
                        inflated_grid[nx, ny] = 1
        self.occupancy_grid = inflated_grid

    def add_random_obstacles(self, obstacle_probability):
        """
        Each cell has probability given by 'obstacle_probability' to be an obstacle
        """
        for x in range(self.height):
            for y in range(self.width):
                if self.rng.random() < obstacle_probability:
                    self.obstacle_positions.append((x, y))
                    self.occupancy_grid[x, y] = 1

    def create_base_plot_map(self):
        plt.close()
        plt.figure()
        plt.imshow(self.occupancy_grid, cmap="binary", interpolation="nearest")
        # plt.gca().set_xticks(
        #     np.arange(0.5, self.occupancy_grid.shape[1], 1), minor=False
        # )
        # plt.gca().set_yticks(
        #     np.arange(0.5, self.occupancy_grid.shape[0], 1), minor=False
        # )

        # plt.xticks([])
        # plt.yticks([])
        plt.grid(which="both", color="black", linestyle="-", linewidth=0)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().invert_yaxis()
        # plt.axis("off")
        # plt.show()

    def plot_map(self):
        plt.imshow(self.occupancy_grid, cmap="binary", interpolation="nearest")
        plt.gca().set_xticks(
            np.arange(0.5, self.occupancy_grid.shape[1], 1), minor=False
        )
        plt.gca().set_yticks(
            np.arange(0.5, self.occupancy_grid.shape[0], 1), minor=False
        )

        plt.xticks([])
        plt.yticks([])
        plt.grid(which="both", color="black", linestyle="-", linewidth=2)
        plt.gca().set_aspect("equal", adjustable="box")
        # plt.axis("off")
        plt.show()

    def save_map(self):
        plt.imshow(self.occupancy_grid, cmap="binary", interpolation="nearest")
        plt.gca().set_xticks(
            np.arange(0.5, self.occupancy_grid.shape[1], 1), minor=False
        )
        plt.gca().set_yticks(
            np.arange(0.5, self.occupancy_grid.shape[0], 1), minor=False
        )

        plt.xticks([])
        plt.yticks([])
        plt.grid(which="both", color="black", linestyle="-", linewidth=2)
        plt.gca().set_aspect("equal", adjustable="box")
        # plt.axis("off")
        plt.gca().invert_yaxis()
        plt.savefig("map.png")
        plt.show()

    def is_free(self, x, y):
        """
        Check if the (x,y) cell is free
        """
        return self.occupancy_grid[x, y] == 0

    def free_space(self):
        """
        Return the list of cells (xi,yi) that are free
        """
        free_space = []
        for x in range(self.height):
            for y in range(self.width):
                if self.is_free(x, y):
                    free_space.append((x, y))
        return free_space


def main():

    # Example usage with visualization:
    heigth = 100
    width = 100
    inflate = True  # Set to True to inflate obstacles
    inflation_radius = 4
    obstacle_probability = (
        0.005  # Probability for each cell of being an obstacle (before inflating)
    )

    grid_map = OccupancyGridMap(
        width, heigth, None, inflate, inflation_radius, obstacle_probability
    )
    grid_map.plot_map()


if __name__ == "__main__":

    main()
