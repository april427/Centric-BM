# %%
import numpy as np
import random
import matplotlib.pyplot as plt

# %%

def update_position(position, velocity, dt, config):
    """Update UE position given velocity and timestep, and reverse direction if outside square area"""
    angle = velocity[1] * dt # calculate angular displacement
    direction = np.array([np.cos(angle), np.sin(angle)]) # calculate direction vector
    displacement = velocity[0] * direction * dt # calculate linear displacement
    if config == 0:
        new_position = position + displacement  # calculate new position
    elif config == 1:
        new_position = position - displacement  # calculate new position
    elif config == 2:
        displacement[0], displacement[1] = displacement[1], displacement[0]
        new_position = position + displacement  # calculate new position
    else:
        displacement[0], displacement[1] = displacement[1], displacement[0]
        new_position = position - displacement  # calculate new position

    return new_position


def sample_velocity():
    """Sample velocity from exponential and normal distributions"""
    
    linear_velocity = np.random.normal(loc = 30, scale=3) 
    if linear_velocity < 0:
        linear_velocity = 0
    # sample linear velocity from exponential distribution
    angular_velocity = np.random.normal(loc=0, scale=0.01) # sample angular velocity from normal distribution
    return np.array([linear_velocity, angular_velocity])

# #Example usage
# dt = 0.1 # timestep
# position = np.array([0, 80]) # initial position
# velocity = sample_velocity() # sample initial velocity
#
# # Update position for 10 timesteps
# for i in range(1000):
#     position = update_position(position, velocity, dt) # update position
#     velocity = sample_velocity() # sample new velocity
#     print(f"Position after timestep {i+1}: {position}")
#     plt.scatter(position[0], position[1], color='b')
# plt.show()