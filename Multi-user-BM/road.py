import numpy as np
import matplotlib.pyplot as plt
import itertools

def sample_velocity():
    """Sample velocity from exponential and normal distributions"""
    
    linear_velocity = np.random.normal(loc = 30, scale=2) 
    if linear_velocity < 0:
        linear_velocity = 0
    # sample linear velocity from exponential distribution
    return np.array([linear_velocity])

def road(cross_pos):
    '''Create a road matrix with cross positions'''

    map_width = 100
    map_length = 100
    single_lane_width = 5
    #cross_pos = np.array([[27, 80], [30, 70]])

    road_matrix = np.zeros((map_width, map_length))
    road_matrix[cross_pos[0,0] - single_lane_width:(cross_pos[0,0]+ single_lane_width), :] = 1

    road_matrix[cross_pos[0,1] - single_lane_width:(cross_pos[0,1]+single_lane_width), :] = 1

    road_matrix[:, cross_pos[1,0] - single_lane_width:(cross_pos[1,0]+single_lane_width)] = 1

    road_matrix[:, cross_pos[1,1] - single_lane_width:(cross_pos[1,1]+single_lane_width)] = 1

    return road_matrix

def set_direction(user_pos, cross_pos, single_lane_width, N_cross=4):
    '''Set the direction of the user based on the closest cross center'''
    cross_center = np.array(list(itertools.product(cross_pos[0], cross_pos[1])))

    dis = np.zeros((N_cross))
    for j in range(N_cross):
        dis[j] = np.linalg.norm(user_pos - cross_center[j])

    x = cross_center[np.argmin(dis)] # closest cross center

    if (user_pos[0] - (x[0] - single_lane_width) < 0*single_lane_width) and (abs(user_pos[1] - x[1]) < single_lane_width):
        config = 0      # move to x axis positive
    elif (user_pos[0] - (x[0] + single_lane_width) > 0*single_lane_width) and (abs(user_pos[1] - x[1]) < single_lane_width):
        config = 1      # move to x axis negative
    elif (user_pos[1] - (x[1] - single_lane_width) < 0*single_lane_width) and abs(user_pos[0] - x[0]) < single_lane_width:
        config = 2    # move to y axis positive
    elif (user_pos[1] - (x[1] + single_lane_width) > 0*single_lane_width) and abs(user_pos[0] - x[0]) < single_lane_width:
        config = 3   # move to y axis negative
    else:
        config = 4  # make a turn

    return config

def user_pos_direction_init(road_matrix, cross_pos, N_UEs, single_lane_width):
    '''Initialize user position and direction'''

    user_pos = np.zeros([N_UEs, 2])
    config = np.empty([N_UEs, 2], dtype = int)
    if_turned = np.zeros(N_UEs, dtype=bool)

    cross_center = np.array(list(itertools.product(cross_pos[0], cross_pos[1])))

    A = np.where(road_matrix == 1)
    
    for i in range(N_UEs):
        indices = np.random.choice(np.arange(i*len(A[1])/N_UEs, (i+1)*len(A[1])/N_UEs-1, dtype=int), replace=False)
        if_turned[i] = False
        user_pos[i,:] = [A[0][indices], A[1][indices]]
        config[i,0] = set_direction(user_pos[i,:], cross_center, single_lane_width)
        config[i,1] = config[i,0] if config[i,0] != 4 else 2
    return user_pos, config, if_turned


def make_a_turn(position, displacement, config):
    """Make a turn at the intersection"""
    direction = np.array([[-1,-1], [1,1], [1,-1], [-1,1]])
    if config[1] == 0:
        a = np.random.choice([1,2,4])
        if a != 4:
            position[0] = position[0] + direction[a,0] * displacement/2
            position[1] = position[1] + direction[a,1] * displacement/2
            config[1] = 2 if direction[a,1] == -1 else 3
        else:
            position[0] = position[0] + displacement

    elif config[1] == 1:
        a = np.random.choice([0,3,4])
        if a != 4:
            position[0] = position[0] + direction[a,0] * displacement/2
            position[1] = position[1] + direction[a,1] * displacement/2
            config[1] = 2 if direction[a,1] == -1 else 3
        else:
            position[0] = position[0] - displacement

    elif config[1] == 2:
        a = np.random.choice([1,3,4])
        if a != 4:
            position[0] = position[0] + direction[a,0] * displacement/2
            position[1] = position[1] + direction[a,1] * displacement/2
            config[1] = 0 if direction[a,0] == 1 else 1
        else:
            position[1] = position[1] + displacement

    elif config[1] == 3:
        a = np.random.choice([0,2,4])
        if a != 4:
            position[0] = position[0] + direction[a,0] * displacement/2
            position[1] = position[1] + direction[a,1] * displacement/2
            config[1] = 0 if direction[a,0] == 1 else 1
        else:
            position[1] = position[1] - displacement

    return position, config

def update_positions(position, cross_pos, velocity, dt, single_lane_width, config, if_turned=False):
    """Update UE position given velocity and timestep, and reverse direction if outside square area"""
    cross_center = np.array(list(itertools.product(cross_pos[0], cross_pos[1])))
    new_position = position.copy()
    displacement = velocity * dt # calculate linear displacement
    #config[0] = config[1]
    if ~if_turned:
        if config[0] == 4:
            new_position, config = make_a_turn(new_position, displacement, config)
            if_turned = True
        else:
            match config[0]:
                case 0:
                    new_position[0] = position[0] + displacement  # move to x axis positive
                
                case 1:
                    new_position[0] = position[0] - displacement  # move to x axis negative
                
                case 2:
                    new_position[1] = position[1] + displacement  # move to y axis positive
                
                case 3:  
                    new_position[1] = position[1] - displacement  # move to y axis negative
            config[0] = set_direction(new_position, cross_center, single_lane_width)
    
    else:
        match config[1]:
            case 0:
                new_position[0] = position[0] + displacement  # move to x axis positive
            
            case 1:
                new_position[0] = position[0] - displacement  # move to x axis negative
            
            case 2:
                new_position[1] = position[1] + displacement  # move to y axis positive
            
            case 3:  
                new_position[1] = position[1] - displacement  # move to y axis negative
        

    return new_position, config, if_turned

# %%
# cross_pos = np.array([[27, 80], [30, 70]])
# #cross_center = np.array(list(itertools.product(cross_pos[0], cross_pos[1])))

# road_matrix = road(cross_pos)
# plt.imshow(road_matrix.T)
# step = 5
# N_UEs = 2
# single_lane_width = 5

# dt = 1
# user_pos = np.zeros((step+1, N_UEs, 2))
# config = np.empty([N_UEs, 2], dtype = int)

# user_pos[0,:,:], config, if_turned = user_pos_direction_init(road_matrix, cross_pos, N_UEs)
# plt.scatter(user_pos[0,:,0], user_pos[0, :,1], color='r')



# for s in range(step):
#     for i in range(N_UEs):
#         user_pos[s+1, i,:], config[i,:], if_turned = update_positions(user_pos[s,i,:], cross_pos, sample_velocity(), dt, config[i,:], if_turned) 

# for i in range(N_UEs):
#     ue_pos_x = user_pos[:,i,0]
#     ue_pos_y = user_pos[:,i,1]
#     du = np.diff(ue_pos_x)
#     dv = np.diff(ue_pos_y)
#     pos_x = ue_pos_x[:-1] + du/2
#     pos_y = ue_pos_y[:-1] + dv/2
#     norm = np.sqrt(du**2+dv**2) 

#     plt.plot(ue_pos_x,ue_pos_y, marker="o")
#     plt.quiver(pos_x, pos_y, du/norm, dv/norm, angles="xy", zorder=5, pivot="mid")
#     plt.xlim(0,100)
#     plt.ylim(0,100)   
#     # hold the plot
# plt.legend(['UE1','','UE2','', 'UE3', ''])    
# plt.show()
# %%
