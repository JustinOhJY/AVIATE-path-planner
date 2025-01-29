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
    def __init__(self, 
                 start, 
                 goal, 
                 obstacle_list, 
                 space_bounds,   # [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
                 step_size=5.0,  
                 expand_distance=2.0, 
                 goal_sample_rate=5, 
                 max_iter=500,
                 smoothness_weight=10.0, 
                 max_turn_angle_deg=60.0,
                 vision_range=20.0     # agent can only see 20 units
                 ):
        """
        :param start: (x, y, z)
        :param goal: (x, y, z)
        :param obstacle_list: 
            List of obstacles (e.g., ("rectprism", cx, cy, cz, lx, ly, lz))
        :param space_bounds: Global bounding region: [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        :param step_size: discretization step for sampling
        :param expand_distance: distance for steering
        :param goal_sample_rate: % chance of sampling the goal directly
        :param max_iter: maximum number of RRT iterations
        :param smoothness_weight: angle penalty factor
        :param max_turn_angle_deg: maximum turn angle in degrees
        :param vision_range: how far we can see around the current aircraft node
        """
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacle_list = obstacle_list
        self.expand_distance = expand_distance
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.smoothness_weight = smoothness_weight
        self.vision_range = vision_range

        # Convert turn angle to radians
        self.max_turn_angle = math.radians(max_turn_angle_deg)

        # Store global bounding box
        (self.xmin, self.xmax) = space_bounds[0]
        (self.ymin, self.ymax) = space_bounds[1]
        (self.zmin, self.zmax) = space_bounds[2]

        # For uniform sampling, we still store these arrays
        self.x_vals = np.arange(self.xmin, self.xmax + step_size, step_size)
        self.y_vals = np.arange(self.ymin, self.ymax + step_size, step_size)
        self.z_vals = np.arange(self.zmin, self.zmax + step_size, step_size)

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
            if new_node is None:
                continue

            self.node_list.append(new_node)
            self.rewire(new_node, near_inds)

            # Check if we're near the goal
            if self.calc_distance_to_goal(new_node.x, new_node.y, new_node.z) <= self.expand_distance:
                final_node = self.steer(new_node, self.goal, self.expand_distance)
                if self.check_collision(final_node):
                    raw_path = self.generate_final_course(len(self.node_list) - 1)
                    # Smoothing
                    smoothed_path = self.smooth_path(raw_path, self.obstacle_list, iterations=100)
                    return smoothed_path

        return None  # Not found within max_iter

    #only sample around the new 20 unit limit from agent
    def get_random_node(self):
        # Chance to sample the goal
        if random.randint(0, 100) <= self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y, self.goal.z)

        # "Current node" = last node in the tree
        current_node = self.node_list[-1]

        # Local bounding box for sampling, clamped to global bounds
        x_min_local = max(self.xmin, current_node.x - self.vision_range)
        x_max_local = min(self.xmax, current_node.x + self.vision_range)
        y_min_local = max(self.ymin, current_node.y - self.vision_range)
        y_max_local = min(self.ymax, current_node.y + self.vision_range)
        z_min_local = max(self.zmin, current_node.z - self.vision_range)
        z_max_local = min(self.zmax, current_node.z + self.vision_range)

        # Sample uniformly within this local region
        x = random.uniform(x_min_local, x_max_local)
        y = random.uniform(y_min_local, y_max_local)
        z = random.uniform(z_min_local, z_max_local)

        return Node(x, y, z)

    def get_nearest_node_index(self, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 + (node.z - rnd_node.z)**2
                 for node in self.node_list]
        return dlist.index(min(dlist))

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
        for obs in self.obstacle_list:
            shape = obs[0]
            if shape == "rectprism":
                _, cx, cy, cz, lx, ly, lz = obs
                if (cx - lx <= node.x <= cx + lx and
                    cy - ly <= node.y <= cy + ly and
                    cz - lz <= node.z <= cz + lz):
                    return False
        return True

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        r = self.expand_distance * math.sqrt((math.log(nnode) / nnode))  
        dlist = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 + (node.z - new_node.z)**2
                 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r**2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node

        best_cost = float("inf")
        best_parent = None
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if not self.check_collision(t_node):
                continue
            if not self.check_turn_feasibility(near_node, t_node):
                continue

            dist, _, _ = self.calc_distance_and_angle(near_node, new_node)
            angle_pen = self.angle_penalty(near_node, t_node)
            total_cost = near_node.cost + dist + angle_pen

            if total_cost < best_cost:
                best_cost = total_cost
                best_parent = near_node

        if best_parent is None:
            return None

        final_node = self.steer(best_parent, new_node)
        final_node.cost = best_cost
        final_node.parent = best_parent
        return final_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if (self.check_collision(edge_node) and
                self.check_turn_feasibility(new_node, edge_node)):

                dist, _, _ = self.calc_distance_and_angle(new_node, near_node)
                angle_pen = self.angle_penalty(new_node, edge_node)
                new_cost = new_node.cost + dist + angle_pen

                if new_cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = new_cost

    def check_turn_feasibility(self, parent_node, new_node):
        if parent_node.parent is None:
            return True

        gp = parent_node.parent
        A = np.array([parent_node.x - gp.x, parent_node.y - gp.y, parent_node.z - gp.z])
        B = np.array([new_node.x - parent_node.x, new_node.y - parent_node.y, new_node.z - parent_node.z])
        normA = np.linalg.norm(A)
        normB = np.linalg.norm(B)
        if normA < 1e-9 or normB < 1e-9:
            return True

        cos_angle = np.dot(A, B) / (normA * normB)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)
        if angle > self.max_turn_angle:
            return False
        return True

    def angle_penalty(self, parent_node, new_node):
        if parent_node is None or parent_node.parent is None:
            return 0.0

        gp = parent_node.parent
        v1 = np.array([parent_node.x - gp.x, parent_node.y - gp.y, parent_node.z - gp.z])
        v2 = np.array([new_node.x - parent_node.x, new_node.y - parent_node.y, new_node.z - parent_node.z])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0

        cos_ang = np.dot(v1, v2) / (norm1 * norm2)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        angle = math.acos(cos_ang)
        return self.smoothness_weight * angle

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = math.sqrt(dx**2 + dy**2 + dz**2)
        theta = math.atan2(dy, dx) if d != 0 else 0
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
        return path[::-1]
    
    def discretize_path(self, path, step=2.0):
        """
        Given a piecewise-linear path (list of [x, y, z]),
        sample intermediate points so that consecutive waypoints
        are no more than 'step' units apart.

        :param path: list of [x, y, z] in order
        :param step: desired spacing in same units as path
        :return: new_path: a list of [x, y, z] with ~'step' spacing
        """
        if not path:
            return []

        new_path = []
        new_path.append(path[0])  # always include start
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)

            if dist < 1e-9:
                continue

            # number of segments between p1 and p2
            n_segments = int(dist // step)
            # step might not divide dist exactly, so we use integer division
            # and then the final segment might be shorter than 'step'

            # sample points
            for s in range(n_segments):
                ratio = (s * step) / dist
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                z = p1[2] + ratio * (p2[2] - p1[2])
                new_path.append([x, y, z])

            # add the exact endpoint (p2) of this segment if it's not the last
            # and if it hasn't already been added
            # We'll add p2 in next iteration or after loop if needed.
            # but we can do it here for clarity:
            new_path.append(p2)

        # Optionally, remove duplicates if they occur
        # (e.g., if multiple segments share the same endpoint)
        final_path = [new_path[0]]
        for i in range(1, len(new_path)):
            if (math.isclose(final_path[-1][0], new_path[i][0], abs_tol=1e-7) and
                math.isclose(final_path[-1][1], new_path[i][1], abs_tol=1e-7) and
                math.isclose(final_path[-1][2], new_path[i][2], abs_tol=1e-7)):
                continue  # skip duplicate
            final_path.append(new_path[i])

        return final_path

    def smooth_path(self, path, obstacle_list, iterations=100):
        if path is None or len(path) < 3:
            return path

        for _ in range(iterations):
            i = random.randint(0, len(path) - 2)
            j = random.randint(i + 1, len(path) - 1)
            if j - i < 2:
                continue

            p_i = path[i]
            p_j = path[j]
            if not self.check_line_collision(p_i, p_j, obstacle_list):
                continue
            if not self.check_smooth_turn(path, i, j):
                continue

            path = path[:i+1] + path[j:]
        return path

    def check_smooth_turn(self, path, i, j):
        if i > 0:
            if not self.check_three_point_turn(path[i-1], path[i], path[j]):
                return False
        if j < len(path) - 1:
            if not self.check_three_point_turn(path[i], path[j], path[j+1]):
                return False
        return True

    def check_three_point_turn(self, A, B, C):
        v1 = np.array([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
        v2 = np.array([C[0] - B[0], C[1] - B[1], C[2] - B[2]])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return True
        cos_ang = np.dot(v1, v2) / (norm1 * norm2)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        angle = math.acos(cos_ang)
        return (angle <= self.max_turn_angle)

    def check_line_collision(self, p1, p2, obstacle_list):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        if dist < 1e-9:
            return True

        step_size = 0.5
        steps = int(dist / step_size)
        for step in range(steps+1):
            t = float(step) / float(steps) if steps > 0 else 0
            x = x1 + (x2 - x1)*t
            y = y1 + (y2 - y1)*t
            z = z1 + (z2 - z1)*t
            if not self.check_point_collision(x, y, z, obstacle_list):
                return False
        return True

    def check_point_collision(self, x, y, z, obstacle_list):
        for obs in obstacle_list:
            if obs[0] == "rectprism":
                _, cx, cy, cz, lx, ly, lz = obs
                if (cx - lx <= x <= cx + lx and
                    cy - ly <= y <= cy + ly and
                    cz - lz <= z <= cz + lz):
                    return False
        return True

    def draw_graph(self, rnd=None, path=None):
        ax = plt.figure().add_subplot(projection='3d')
        if rnd is not None:
            ax.scatter(rnd.x, rnd.y, rnd.z, c='k', marker='^')
        for node in self.node_list:
            if node.parent:
                ax.plot([node.x, node.parent.x],
                        [node.y, node.parent.y],
                        [node.z, node.parent.z], "-g")

        for obs in self.obstacle_list:
            shape = obs[0]
            if shape == "rectprism":
                _, cx, cy, cz, lx, ly, lz = obs
                self.draw_rectangular_prism(ax, cx, cy, cz, lx, ly, lz)

        ax.scatter(self.start.x, self.start.y, self.start.z, c='blue', marker='x')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, c='purple', marker='x')
        if path is not None:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color='black', linewidth=2)

        plt.title("RRT* with Limited Vision & Min Turn Radius")
        plt.grid(True)
        plt.show()

    def draw_rectangular_prism(self, ax, cx, cy, cz, lx, ly, lz):
        corners = []
        for dx in [-lx, lx]:
            for dy in [-ly, ly]:
                for dz in [-lz, lz]:
                    corners.append([cx + dx, cy + dy, cz + dz])
        corners = np.array(corners)
        edges = [
            (0, 1), (0, 2), (0, 4),
            (3, 1), (3, 2), (3, 7),
            (5, 1), (5, 4), (5, 7),
            (6, 2), (6, 4), (6, 7)
        ]
        for (i, j) in edges:
            xs = [corners[i][0], corners[j][0]]
            ys = [corners[i][1], corners[j][1]]
            zs = [corners[i][2], corners[j][2]]
            ax.plot(xs, ys, zs, color='r')

    def print_waypoints(self, path):
        """
        Prints all waypoints in the given path.

        :param path: list of [x, y, z] waypoints
        """
        if not path:
            print("No path to print!")
            return

        for i, waypoint in enumerate(path):
            x, y, z = waypoint
            print(f"Waypoint {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")

def main():
    print("Starting RRT* with Limited Vision...")

    space_bounds = [
        (-20, 120),
        (-20, 120),
        (0, 120)
    ]

    # Start and Goal
    start = (0, 10, 0)
    goal = (80, 80, 40)

    # Example obstacles
    obstacle_list = [
        ("rectprism", 40, 40, 30, 10, 10, 30),
        ("rectprism", 60, 20, 20, 5, 5, 10),
        ("rectprism", 12.5, 40, 30, 10, 10, 30),
        ("rectprism", 62.5, 40, 30, 10, 10, 30)
    ]

    # Instantiate RRT* with a 20 unit vision range
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        space_bounds=space_bounds,
        step_size=5.0,
        expand_distance=2.0,
        goal_sample_rate=5,
        max_iter=5000,
        smoothness_weight=15.0,
        max_turn_angle_deg=60.0,
        vision_range=20.0
    )

    path = rrt_star.planning()
    if path is None:
        print("No path found!")
    else:
        print("Path found!")
        # 2) Discretize the path at 0.5-meter spacing 
        discrete_path = rrt_star.discretize_path(path, step=0.5)
        print("\nDiscretized Waypoints:")
        rrt_star.print_waypoints(discrete_path)

        # 3) Visualize the final path or the discrete path (both are valid to show)
        #    We'll show the discrete one:
        rrt_star.draw_graph(path=discrete_path)

if __name__ == "__main__":
    main()
