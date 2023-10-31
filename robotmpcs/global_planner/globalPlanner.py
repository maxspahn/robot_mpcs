import numpy as np
import matplotlib.pyplot as plt
import copy
import cv2 as cv #imaging processing toolbox
from robotmpcs.global_planner.a_star import a_star
from robotmpcs.global_planner.gridmap import OccupancyGridMap

class GlobalPlanner(object):
    def __init__(self, dim_pixels, limits_low, limits_high,
                 BOOL_PLOTTING=True,
                 threshold=0.29,
                 convolution_blur = (5, 5),
                 enlarge_obstacles=True,
                 threshold_local_goal=1.3):
        self.dim_pixels = dim_pixels
        self.limits_high = limits_high
        self.limits_low = limits_low
        self.dim_meters = -limits_low + limits_high
        self.cell_size_xyz = self.dim_meters/dim_pixels
        self.threshold = threshold
        self.enlarge_obstacles = enlarge_obstacles
        self.convolution_blur = convolution_blur
        self.idx_local = 0
        self.threshold_local_goal = threshold_local_goal

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

    def get_enlarged_obstacles(self, size_robot=0.5):
        """
        Blurs image to get enlarged obstacles
        be ware that for images the max value is 255
        """
        # img = cv.imread('occupancy_map.png', cv.IMREAD_GRAYSCALE)
        # img_blurred = cv.blur(img, self.convolution_blur)  #todo: only need opencv for this, could be written by convolution
        #
        # image_enlarged_obst = np.clip(img_blurred/255, 0, self.threshold)
        # plt.imsave('occupancy_map_enlarged.png', image_enlarged_obst)

        gmap = OccupancyGridMap.from_png('occupancy_map.png', cell_size=self.cell_size)
        size_robot_pixels = int(np.ceil(size_robot/self.cell_size))
        self.kernel = np.ones((size_robot_pixels*2+1, size_robot_pixels*2+1))
        self.occupancy_map_convoluted = self.convolution_size_robot(occ_map=gmap.data, kernel=self.kernel)
        self.occupancy_map_enlarged = self.create_binary_map(self.occupancy_map_convoluted)
        return self.occupancy_map_enlarged

    def convolution_size_robot(self, occ_map, kernel):
        k = int((len(kernel)-1)/2)
        sum_kernel = np.sum(kernel)
        occ_map_convoluted = copy.deepcopy(occ_map)
        for i in range(k, occ_map.shape[0]-k):
            for j in range(k, occ_map.shape[1]-k):
                # for k in range(len(kernel)):
                subsample = kernel*occ_map[i-k:i+k+1, j-k:j+k+1]
                value_pixel = np.sum(subsample)/sum_kernel  #to avoid very large values
                occ_map_convoluted[i, j] = value_pixel
        return occ_map_convoluted

    def create_binary_map(self, occ_map):
        occ_map_enlarged = copy.deepcopy(occ_map)
        for i in range(occ_map.shape[0]):
            for j in range(occ_map.shape[1]):
                if occ_map[i, j]>self.threshold:
                    occ_map_enlarged[i, j] = 1
                else:
                    occ_map_enlarged[i, j] = 0
        return occ_map_enlarged

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
        # # enlarge obstacles in the map to fill gaps where the robot would not fit:
        # if self.enlarge_obstacles == True:
        #     self.get_enlarged_obstacles()

        # load the map
        gmap = OccupancyGridMap.from_png('occupancy_map.png', cell_size=self.cell_size)

        # enlarge obstacles in the map to fill gaps where the robot would not fit:
        if self.enlarge_obstacles == True:
            gmap.data = self.get_enlarged_obstacles()

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


    def get_distance_points(self, position1, position2):
        distance = np.sqrt((position2[0]-position1[0])**2 + (position2[1]-position1[1])**2)
        return distance

    def get_local_goal(self, position, path):
        """
        Gets the local goal on the global path that is closer then x meters
        - closer then x meters
        - not going backwards along the path
        Only update when the final node is not reached.
        Returns the (x, y) coordinates of the local path
        """
        distance_pos_path = self.get_distance_points(position, path[self.idx_local])

        if self.idx_local < len(path)-1 and len(path)>0: #Only update when not at final node
            if distance_pos_path <= self.threshold_local_goal:
                self.idx_local = self.idx_local + 1

        local_goal = path[self.idx_local]
        return local_goal

    # def check_local_goal_stuck(self, time, dt=):
    #     """
    #     It might be nice that if the local goal is not reachable, it tries the next one if that is
    #     """
