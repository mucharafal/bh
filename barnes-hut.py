import numpy as np
import pandas as pd
import matplotlib 
import sys

# Const
G = 6.6732
# 0 < Fi < 1
FI = 0.5

# To replace from cmd
particles_number = int(sys.argv[1])

# Simulation is in 3d
dimensions = 3

# For the same results with every run
np.random.seed(1)

# Generate particles parameters: position (0, 1, 2), velocity (3, 4, 5) and mass (6)
particles_parameters = np.random.rand(particles_number, (2*dimensions+1))
# particles_parameters = np.array([[0, 0, 0, 0, -1, 0, 1], [1, 0, 0, 0, 1, 0, 1]], np.dtype(np.float))

# Table for forces
forces = np.zeros((particles_number, dimensions))

# Also should be parametrized from cmd
iterations_number = int(sys.argv[2])

track = np.zeros((iterations_number, particles_number, dimensions), np.dtype(np.float))

# Time of one step
delta_time = 1e-3


def vec_distance(vec1, vec2):
    diff = vec1 - vec2
    return np.sqrt(np.sum(diff * diff))


class Node:
    def __init__(self, nodes, min_v, max_v):
        self.nodes = nodes
        self.min_v = min_v
        self.max_v = max_v

    def map_and_sum(self, function):
        return sum([function(node) for node in self.nodes])

    def mass_center(self):
        return self.map_and_sum(lambda x: x.mass() * x.mass_center()) / self.mass()

    def mass(self):
        return self.map_and_sum(lambda x: x.mass())

    def get_particles_number(self):
        return self.map_and_sum(lambda x: x.get_particles_number())

    def diameter(self):
        return vec_distance(self.min_v, self.max_v)

class Leaf(Node):
    def __init__(self, particle):
        self.particle = particle
    
    def mass_center(self):
        return self.particle[0:3]
    
    def mass(self):
        return self.particle[6]

    def get_particles_number(self):
        return 1

class Empty(Leaf):
    def __init__(self):
        pass

    def mass(self):
        return 0

    def mass_center(self):
        return np.array([0, 0, 0])

    def get_particles_number(self):
        return 0


def find_boundaries(particles_parameters):
    min_v = np.full(dimensions, np.inf)
    max_v = np.full(dimensions, -np.inf)
    for particle in particles_parameters:
        min_v = np.minimum(min_v, particle[0:3])
        max_v = np.maximum(max_v, particle[0:3])
    return (min_v, max_v)


def is_in_part(particle, middle_point, x_less, y_less, z_less):
    is_x_less = particle[0] <= middle_point[0]
    is_y_less = particle[1] <= middle_point[1]
    is_z_less = particle[2] <= middle_point[2]
    return is_x_less == x_less and is_y_less == y_less and is_z_less == z_less


def partition(particles_parameters):
    (min_v, max_v) = find_boundaries(particles_parameters)
    middle = (min_v + max_v) / 2
    partitions = []
    for x_less in [True, False]:
        for y_less in [True, False]:
            for z_less in [True, False]:
                partitions.append([particle for particle in particles_parameters if is_in_part(particle, middle, x_less, y_less, z_less)])
    return (partitions, min_v, max_v)


def createTree(particles_parameters):
    if(len(particles_parameters) > 1):
        (partitions, min_v, max_v) = partition(particles_parameters)
        nodes = [createTree(partition) for partition in partitions]
        return Node(nodes, min_v, max_v)
    elif len(particles_parameters) == 1:
        return Leaf(particles_parameters[0])
    else:
        return Empty()


def calculate_force(particle, other_particle):
    delta_position = particle[0:3] - other_particle[0:3]
    # sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2))
    r2 = np.sum(delta_position * delta_position)
    r = np.sqrt(r2)
    
    particle_mass = particle[6]
    other_particle_mass = other_particle[6]

    if particle_mass == 0 or other_particle_mass == 0 or r2 == 0:
        return 0

    f = G * particle_mass * other_particle_mass / r2
    fp = -f * delta_position / r
    # print(fp)
    return fp


def traverse_graph(particle, tree, diameter):
    if isinstance(tree, Empty):
        return 0
    if isinstance(tree, Leaf):
        return calculate_force(particle, tree.particle)
    mass_center = tree.mass_center()
    if vec_distance(mass_center, particle[0:3]) / diameter > FI:
        return calculate_force(particle, np.array([mass_center[0], mass_center[1], mass_center[2], 0, 0, 0, tree.mass()]))
    return sum(traverse_graph(particle, subtree, diameter) for subtree in tree.nodes)


def update_forces(particles_parameters, forces):
    forces.fill(0)
    tree = createTree(particles_parameters)
    for (index, particle) in enumerate(particles_parameters):
        force = traverse_graph(particle, tree, tree.diameter())
        forces[index] = force


def update_position_and_velocity(particles_parameters, forces):
    for (index, (particle, force)) in enumerate(zip(particles_parameters, forces)):
        acc = force / particle[6] * delta_time
        new_position = particle[0:3] + particle[3:6] * delta_time + acc * delta_time / 2
        new_velocity = particle[3:6] + acc
        particles_parameters[index][0:3] = new_position
        particles_parameters[index][3:6] = new_velocity

for it in range(iterations_number):
    track[it] = (particles_parameters[:, 0:3])
    update_forces(particles_parameters, forces)
    update_position_and_velocity(particles_parameters, forces)

print("Finished")

# ax = None
# for i in range(particles_number):
#     if ax:
#         df = pd.DataFrame(track[:, i, 0:2])
#         df.plot.scatter(x=0, y=1, ax=ax, linewidths=0.1)
#     else:
#         df = pd.DataFrame(track[:, i, 0:2])
#         ax = df.plot.scatter(x=0, y=1, c='r', linewidths=0.5)
    
# matplotlib.pyplot.show()
np.savetxt("foo.csv", particles_parameters, delimiter=",")