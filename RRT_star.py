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
                 step_size=5.0,  # spacing between discrete points
                 expand_distance=2.0, 
                 goal_sample_rate=5, 
                 max_iter=500,
                 smoothness_weight=10.0,  # weight for penalizing sharp angles
                 max_turn_angle_deg=60.0  # max allowable turn angle
                 ):
        """
        :param start: (x, y, z)
        :param goal: (x, y, z)
        :param obstacle_list: 
            List of obstacles. For rectangular prisms, each entry:
              ("rectprism", cx, cy, cz, lx, ly, lz)
        :param space_bounds: 3D bounding region for sampling: [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        :param step_size: discretization step for sampling
        :param expand_distance: distance used for steering
        :param goal_sample_rate: percentage chance of sampling the goal
        :param max_iter: maximum number of iterations
        :param smoothness_weight: factor for penalizing turning angles in cost
        :param max_turn_angle_deg: maximum allowed turn angle (in degrees)
        """

        self.start = Node(*start)
        self.goal = Node(*goal)
        self.obstacle_list = obstacle_list
        self.expand_distance = expand_distance
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.smoothness_weight = smoothness_weight

        # Convert turn angle from degrees to radians
        self.max_turn_angle = math.radians(max_turn_angle_deg)

        # Store 3D bounding box
        (xmin, xmax) = space_bounds[0]
        (ymin, ymax) = space_bounds[1]
        (zmin, zmax) = space_bounds[2]
        
        # Create discrete sets of x, y, z coordinates
        self.x_vals = np.arange(xmin, xmax + step_size, step_size)
        self.y_vals = np.arange(ymin, ymax + step_size, step_size)
        self.z_vals = np.arange(zmin, zmax + step_size, step_size)

        # Maintain a list of nodes
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_distance)

            # Check collision
            if not self.check_collision(new_node):
                continue

            # Find neighborhood
            near_inds = self.find_near_nodes(new_node)
            new_node = self.choose_parent(new_node, near_inds)
            if new_node is None:
                continue

            self.node_list.append(new_node)
            self.rewire(new_node, near_inds)

            # Check goal proximity
            if self.calc_distance_to_goal(new_node.x, new_node.y, new_node.z) <= self.expand_distance:
                final_node = self.steer(new_node, self.goal, self.expand_distance)
                if self.check_collision(final_node):
                    raw_path = self.generate_final_course(len(self.node_list) - 1)
                    # Smoothing (shortcut) with turn feasibility checks
                    smoothed_path = self.smooth_path(raw_path, self.obstacle_list, iterations=100)
                    return smoothed_path

        return None  # Not found within max_iter

    def get_random_node(self):
        # With some probability, sample the goal to increase connectivity
        if random.randint(0, 100) > self.goal_sample_rate:
            x = random.choice(self.x_vals)
            y = random.choice(self.y_vals)
            z = random.choice(self.z_vals)
            return Node(x, y, z)
        else:
            return Node(self.goal.x, self.goal.y, self.goal.z)

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
        # Cost so far: parent's cost + length of new edge
        new_node.cost = from_node.cost + extend_length
        return new_node

    def check_collision(self, node):
        """
        Checks whether 'node' is inside any rectangular prism in obstacle_list.
        """
        for obs in self.obstacle_list:
            shape = obs[0]
            if shape == "rectprism":
                _, cx, cy, cz, lx, ly, lz = obs
                if (cx - lx <= node.x <= cx + lx and
                    cy - ly <= node.y <= cy + ly and
                    cz - lz <= node.z <= cz + lz):
                    return False  # Collision
        return True  # Safe

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list)
        # Neighborhood radius
        r = self.expand_distance * math.sqrt((math.log(nnode) / nnode))
        dlist = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 + (node.z - new_node.z)**2
                 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r**2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        """
        Evaluate potential parents in near_inds, picking the one 
        with minimal cost + angle penalty, and that does NOT exceed the max turn angle.
        """
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

            # Compute cost with angle penalty
            dist, _, _ = self.calc_distance_and_angle(near_node, new_node)
            angle_pen = self.angle_penalty(near_node, t_node)
            total_cost = near_node.cost + dist + angle_pen

            if total_cost < best_cost:
                best_cost = total_cost
                best_parent = near_node

        if best_parent is None:
            # No feasible parent
            return None

        # Construct new_node with updated cost & parent
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
        """
        Ensures the turn from parent_node's parent -> parent_node -> new_node
        does not exceed self.max_turn_angle.
        If parent_node has no grandparent, there's no turn to check.
        """
        if parent_node.parent is None:
            return True  # no grandparent => no turn to check

        # Vector A = parent_node - grandparent
        gp = parent_node.parent
        Ax = parent_node.x - gp.x
        Ay = parent_node.y - gp.y
        Az = parent_node.z - gp.z
        A = np.array([Ax, Ay, Az])

        # Vector B = new_node - parent_node
        Bx = new_node.x - parent_node.x
        By = new_node.y - parent_node.y
        Bz = new_node.z - parent_node.z
        B = np.array([Bx, By, Bz])

        normA = np.linalg.norm(A)
        normB = np.linalg.norm(B)
        if normA < 1e-9 or normB < 1e-9:
            return True  # degenerate => no meaningful turn

        cos_angle = np.dot(A, B) / (normA * normB)
        # clamp floating errors
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.acos(cos_angle)

        # If angle is bigger than allowed => turn is too sharp
        if angle > self.max_turn_angle:
            return False
        return True

    def angle_penalty(self, parent_node, new_node):
        """
        Additional cost for sharp angles. 
        (We still also have a hard constraint from check_turn_feasibility above.)
        """
        if parent_node is None or parent_node.parent is None:
            return 0.0

        # Vector 1: parent_node - grandparent
        gp = parent_node.parent
        v1 = np.array([parent_node.x - gp.x, 
                       parent_node.y - gp.y, 
                       parent_node.z - gp.z])

        # Vector 2: new_node - parent_node
        v2 = np.array([new_node.x - parent_node.x, 
                       new_node.y - parent_node.y, 
                       new_node.z - parent_node.z])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return 0.0

        cos_ang = np.dot(v1, v2) / (norm1 * norm2)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        angle = math.acos(cos_ang)

        # Weighted penalty
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
        return path[::-1]  # reverse => start->goal

    def smooth_path(self, path, obstacle_list, iterations=100):
        """
        "Shortcut" smoothing:
        - Randomly pick two indices i < j in the path
        - Check if there's a straight line from path[i] to path[j] that is collision-free
        - Also ensure that connecting path[i] -> path[j] does not break min turn radius 
          at either i or j.
        - If feasible, remove intermediate points
        """
        if path is None or len(path) < 3:
            return path

        for _ in range(iterations):
            i = random.randint(0, len(path) - 2)
            j = random.randint(i + 1, len(path) - 1)
            if j - i < 2:
                # Not enough in-between points to skip
                continue

            p_i = path[i]   # [x_i, y_i, z_i]
            p_j = path[j]   # [x_j, y_j, z_j]

            # Collision check for direct line
            if not self.check_line_collision(p_i, p_j, obstacle_list):
                continue

            # Check turn feasibility at the "join":
            #   The turn at path[i] -> path[i+1 or i-1] -> path[j]
            #   and the turn at path[i] -> path[j] -> path[j+1 or j-1]
            # Because we remove everything in between, we treat i -> j as neighbors.
            if not self.check_smooth_turn(path, i, j):
                continue

            # Shortcut is feasible; remove intermediate nodes
            path = path[:i+1] + path[j:]
        return path

    def check_smooth_turn(self, path, i, j):
        """
        Ensure that connecting path[i] -> path[j] does not yield 
        a turn angle bigger than self.max_turn_angle at i or j 
        (relative to the neighbors on each side).
        """
        # Check the angle at path[i] with the node before it (if exists)
        if i > 0:
            # turn: path[i-1] -> path[i] -> path[j]
            if not self.check_three_point_turn(path[i-1], path[i], path[j]):
                return False

        # Check the angle at path[j] with the node after it (if exists)
        if j < len(path) - 1:
            # turn: path[i] -> path[j] -> path[j+1]
            if not self.check_three_point_turn(path[i], path[j], path[j+1]):
                return False

        return True

    def check_three_point_turn(self, A, B, C):
        """
        Check the turn angle formed by A->B->C 
        is not beyond self.max_turn_angle.
        A, B, C are [x, y, z].
        """
        v1 = np.array([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
        v2 = np.array([C[0] - B[0], C[1] - B[1], C[2] - B[2]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            return True  # degenerate, no real turn

        cos_ang = np.dot(v1, v2) / (norm1 * norm2)
        cos_ang = max(-1.0, min(1.0, cos_ang))
        angle = math.acos(cos_ang)

        return (angle <= self.max_turn_angle)

    def check_line_collision(self, p1, p2, obstacle_list):
        """
        Discretize the line p1->p2, check if any point is in collision.
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        if dist < 1e-9:
            return True

        step_size = 0.5
        steps = int(dist / step_size)
        for step in range(steps+1):
            t = float(step) / float(steps) if steps > 0 else 0
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            z = z1 + (z2 - z1) * t
            if not self.check_point_collision(x, y, z, obstacle_list):
                return False
        return True

    def check_point_collision(self, x, y, z, obstacle_list):
        """
        Check if (x,y,z) is inside any rectangular prism.
        """
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

        # Plot edges
        for node in self.node_list:
            if node.parent:
                ax.plot(
                    [node.x, node.parent.x],
                    [node.y, node.parent.y],
                    [node.z, node.parent.z],
                    "-g"
                )

        # Plot obstacles
        for obs in self.obstacle_list:
            shape = obs[0]
            if shape == "rectprism":
                _, cx, cy, cz, lx, ly, lz = obs
                self.draw_rectangular_prism(ax, cx, cy, cz, lx, ly, lz)

        # Start and Goal
        ax.scatter(self.start.x, self.start.y, self.start.z, c='blue', marker='x')
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, c='purple', marker='x')

        # Final path
        if path is not None:
            path = np.array(path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color='black', linewidth=2)

        plt.title("RRT* 3D with Min Turn Radius & Shortcut Smoothing")
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


def main():
    print("Starting RRT* with Min Turn Radius + Shortcut Smoothing...")

    # space bounding the grid
    space_bounds = [
        (-20, 120),  # X range
        (-20, 120),  # Y range
        (0, 120)     # Z range
    ]

    # Define start and goal
    start = (0, 10, 0)
    goal = (80, 80, 40)

    # Obstacles: "rectprism", center=(cx,cy,cz), half-lengths=(lx,ly,lz)
    obstacle_list = [
        ("rectprism", 40, 40, 30, 10, 10, 30),
        ("rectprism", 60, 20, 20, 5, 5, 10),
        ("rectprism", 12.5, 40, 30, 10, 10, 30),
        ("rectprism", 62.5, 40, 30, 10, 10, 30)
    ]

    rrt_star = RRTStar(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        space_bounds=space_bounds,
        step_size=5.0,
        expand_distance=2.0, 
        goal_sample_rate=5, 
        max_iter=10000,
        smoothness_weight=15.0,    # angle penalty weight
        max_turn_angle_deg=60.0    # e.g. 60-degree max turn
    )

    path = rrt_star.planning()
    if path is None:
        print("No path found!")
    else:
        print("Path found!")
        rrt_star.draw_graph(path=path)


if __name__ == '__main__':
    main()
