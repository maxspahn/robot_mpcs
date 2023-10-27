import numpy as np
import matplotlib.pyplot as plt
from robotmpcs.global_planner.a_star import a_star
from robotmpcs.global_planner.gridmap import OccupancyGridMap

class GlobalPlanner(object):
    def __init__(self, dim_pixels, limits_low, limits_high, BOOL_PLOTTING=True, threshold=0.2):
        self.dim_pixels = dim_pixels
        self.limits_high = limits_high
        self.limits_low = limits_low
        self.dim_meters = -limits_low + limits_high
        self.cell_size_xyz = self.dim_meters/dim_pixels
        self.threshold = threshold

        # dimension of cell must be the same in x and y direction:
        if self.cell_size_xyz[0] == self.cell_size_xyz[1]:
            self.cell_size = self.cell_size_xyz[0]
        else:
            print("The voxels must have the same size [meter x meter] in the x, y direction! Please correct!!")
            self.cell_size = self.cell_size_xyz[0]

        self.BOOL_PLOTTING = BOOL_PLOTTING

    def get_occupancy_map(self, sensor, occupancy_map_3D):
        self.occupancy_map_2D = np.clip(np.sum(occupancy_map_3D, axis=2), 0, self.threshold)
        plt.imsave('occupancy_map.png', self.occupancy_map_2D)
        return sensor

    def plot_occupancy_map(self):
        """
        Plot occupancy map from numpy.array(N, M)
        """
        plt.style.use('_mpl-gallery-nogrid')

        # plot
        fig, ax = plt.subplots()

        ax.imshow(self.occupancy_map_2D)
        plt.grid(True)
        plt.show()

    def plot_occupancy_map_and_path(self, path, gmap, start_pos, goal_pos):
        # plot occupancy map (from struct)
        gmap.plot()

        # plot start and goal node
        start_node_px = gmap.get_index_from_coordinates(start_pos[0], start_pos[1])
        goal_node_px = gmap.get_index_from_coordinates(goal_pos[0], goal_pos[1])
        plt.plot(start_node_px[0], start_node_px[1], 'ro')
        plt.plot(goal_node_px[0], goal_node_px[1], 'go')

        # plot path (if exists)
        if path:
            path_arr = np.array(path)
            plt.plot(path_arr[:, 0], path_arr[:, 1], 'y')
        plt.grid(True)
        plt.show()

    def convert_meters(self, pos_meters):
        """
        For the Astar, you would like to have all xy-coordinates positive and in the frame of the image.
        The image has the (0, 0) frame at the top left and has the x and y axis flipped.
        """
        pos_meters_update = pos_meters - self.limits_low
        pos_meters_updated = [pos_meters_update[1], self.dim_meters[1] - pos_meters_update[0], pos_meters[2]]

        return pos_meters_updated

    def convert_meters_reversed(self, pos_meters):
        """
        Reversed coordinate transformation of function above (convert_meters)
        """
        if len(pos_meters) == 2:
            pos_meters = pos_meters + (0.0,)
        pos_meters_update = [self.dim_meters[1] - pos_meters[1], pos_meters[0], pos_meters[2]]
        pos_meters_updated = pos_meters_update + self.limits_low

        return pos_meters_updated

    def convert_path(self, path):
        path_converted = []
        for position in path:
            pos_converted = self.convert_meters_reversed(position)
            path_converted.append(pos_converted)
        return path_converted

    def add_path_to_env(self, path_length, env):
        """
        Add correct number of shapes to visual
        """
        for _ in range(path_length):
            env.add_visualization(size=[0.1, 0.1],
                                  rgba_color=[0.0, 1.0, 1.0, 0.3])

    def get_global_path_astar(self, start_pos, goal_pos):
        # load the map
        gmap = OccupancyGridMap.from_png('occupancy_map.png', cell_size=self.cell_size)

        # convert coordinates to make sure all (x, y) coordinates are positive and correct wrt the gmap:
        start_pos = self.convert_meters(start_pos)
        goal_pos = self.convert_meters(goal_pos)

        # compute path
        path, path_px = a_star(start_pos, goal_pos, gmap, movement='8N')

        #check if path is feasible
        if path:
            print("path is feasible")
        else:
            print('Goal is not reachable')

        #plot path if BOOL is satisfied:
        if self.BOOL_PLOTTING == 1:
            self.plot_occupancy_map_and_path(path=path_px, gmap=gmap, start_pos=start_pos, goal_pos=goal_pos)
        path_converted = self.convert_path(path)
        return path_converted, path_px
