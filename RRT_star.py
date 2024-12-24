import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d import Axes3D


class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0


class RRTStar:
    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_distance=2.0, goal_sample_rate=5, max_iter=500):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_distance = expand_distance
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_distance)

            if not self.check_collision(new_node):
                continue

            near_inds = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_inds)
            self.node_list.append(new_node)
            self.rewire(new_node, near_inds)

            if self.calc_distance_to_goal(new_node.x, new_node.y, new_node.z) <= self.expand_distance:
                final_node = self.steer(new_node, self.goal, self.expand_distance)
                if self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # Path not found

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand)
            )
        else:
            rnd = Node(self.goal.x, self.goal.y, self.goal.z)
        return rnd

    def get_nearest_node_index(self, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 +
                 (node.z - rnd_node.z)**2 for node in self.node_list]
        minind = dlist.index(min(dlist))
        return minind

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, from_node.z)
        d, theta, phi = self.calc_distance_and_angle(new_node, to_node)

        extend_length = min(extend_length, d)
        new_node.x += extend_length * math.cos(theta) * math.sin(phi)
        new_node.y += extend_length * math.sin(theta) * math.sin(phi)
        new_node.z += extend_length * math.cos(phi)

        new_node.parent = from_node
        new_node.cost = from_node.cost + extend_length
        return new_node

    def check_collision(self, node):
        for (shape, ox, oy, oz, size) in self.obstacle_list:
            d = float('inf')
            if (shape == "sphere"):
                dx = ox - node.x
                dy = oy - node.y
                dz = oz - node.z
                d = dx * dx + dy * dy + dz * dz
            elif (shape == "cylinder"):
                if node.z <= oz:
                    dx = ox - node.x
                    dy = oy - node.y
                    d = dx * dx + dy * dy
            if d <= size**2:
                return False  # Collision
        return True  # Safe

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        r = self.expand_distance * math.sqrt((math.log(nnode) / nnode))
        dlist = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 +
                 (node.z - new_node.z)**2 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r**2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if self.check_collision(t_node):
                costs.append(near_node.cost + self.calc_distance_and_angle(near_node, new_node)[0])
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        min_ind = near_inds[costs.index(min_cost)]
        if min_cost == float("inf"):
            return new_node
        new_node.cost = min_cost
        new_node.parent = self.node_list[min_ind]
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not self.check_collision(edge_node):
                continue
            cost = new_node.cost + self.calc_distance_and_angle(new_node, near_node)[0]
            if cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = cost

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = math.sqrt(dx**2 + dy**2 + dz**2)
        theta = math.atan2(dy, dx)  
        phi = math.acos(dz / d) if d != 0 else 0 
        return d, theta, phi

    def calc_distance_to_goal(self, x, y, z):
        dx = x - self.goal.x
        dy = y - self.goal.y
        dz = z - self.goal.z
        return math.sqrt(dx**2 + dy**2 + dz**2)

    def generate_final_course(self, goal_ind):
        path = []
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])
        return path

    def draw_graph(self, rnd=None, path=None):
        ax = plt.figure().add_subplot(projection='3d')
        if rnd is not None:
            ax.scatter(rnd.x, rnd.y, rnd.z, c='k', marker='^')
        for node in self.node_list:
            if node.parent:
                ax.plot(
                    [node.x, node.parent.x],
                    [node.y, node.parent.y],
                    [node.z, node.parent.z],
                    "-g"
                )
        for (shape, ox, oy, oz, size) in self.obstacle_list:
            if shape == "sphere":
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = size * np.cos(u) * np.sin(v) + ox
                y = size * np.sin(u) * np.sin(v) + oy
                z = size * np.cos(v) + oz
            elif shape == "cylinder":
                u, v = np.mgrid[0:2*np.pi:20j, 0:oz:10j]
                x = size * np.cos(u) + ox
                y = size * np.sin(u) + oy
                z = v
            ax.plot_wireframe(x, y, z, color="r")
        ax.scatter(self.start.x, self.start.y, self.start.z, c='blue', marker='x')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, c='purple', marker='x')

        if path is not None:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color='black', linewidth=2)

        plt.grid(True)
        plt.show()

def main():
    print("Starting RRT* path planning in 3D...")
    # Define start and goal positions
    start = (0, 0, 0)
    goal = (50, 50, 50)
    # Define obstacles as (sphere, x, y, z, radius) or (cylinder, x, y, height, radius)
    # Shapes: Sphere, Cylinder
    obstacle_list = [
        ("cylinder", 25, 25, 50, 5),
        ("cylinder", 10, 20, 50, 10),
        ("cylinder", 30, 25, 50, 5),
    ]
    rand_area = [0, 60]
    rrt_star = RRTStar(start, goal, obstacle_list, rand_area)
    path = rrt_star.planning()

    if path is None:
        print("No path found!")
    else:
        print("Path found!")
        rrt_star.draw_graph(path=path)


if __name__ == '__main__':
    main()
