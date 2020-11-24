import numpy as np
import pandas as pd
import matplotlib 
import sys

# Const
G = 6.6732

# To replace from cmd
particles_number = 10000

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
iterations_number = 1

track = np.zeros((iterations_number, particles_number, dimensions), np.dtype(np.float))

# Time of one step
delta_time = 1e-3

def update_forces(particles_parameters, forces):
    forces.fill(0)
    for (index, particle) in enumerate(particles_parameters):
        for (other_index, other_particle) in enumerate(particles_parameters[index+1:]):
            # (x1 - x2), (y1 - y2), (z1 - z2)
            delta_position = particle[0:3] - other_particle[0:3]
            # sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2))
            r2 = np.sum(delta_position * delta_position)
            r = np.sqrt(r2)
            
            particle_mass = particle[6]
            other_particle_mass = other_particle[6]

            f = G * particle_mass * other_particle_mass / r2
            fp = f * delta_position / r
            # print(fp)
            forces[index] += -fp
            forces[other_index + (index + 1)] += fp

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