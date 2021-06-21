import numpy as np

val = 1
action_table = []

action_table.append(["mov","x",val])
action_table.append(["mov","x",-val])
action_table.append(["mov","y",val])
action_table.append(["mov","y",-val])
action_table.append(["mov","z",val])
action_table.append(["mov","z",-val])
action_table.append(["rot","x",val])
action_table.append(["rot","x",-val])
action_table.append(["rot","y",val])
action_table.append(["rot","y",-val])
action_table.append(["rot","z",val])
action_table.append(["rot","z",-val])

def rand_reset(variance, state = [[0,0,0,0,0,0]]):

    x_pos = np.random.randint(state[0][0]-variance, state[0][0] + variance)
    y_pos = np.random.randint(state[0][1]-variance, state[0][1] + variance)
    z_pos = np.random.randint(state[0][2]-variance, state[0][2] + variance)
    # x_rot = np.random.randint(-180, 180)
    # y_rot = np.random.randint(-180, 180)
    # z_rot = np.random.randint(-180, 180)
    x_rot = np.random.randint(state[0][3]-variance, state[0][3] + variance)
    y_rot = np.random.randint(state[0][4]-variance, state[0][4] + variance)
    z_rot = np.random.randint(state[0][5]-variance, state[0][5] + variance)

    state = [[x_pos, y_pos, z_pos, x_rot, y_rot, z_rot]]

    return np.array(state)

def rand_reset_full(variance, state = [[0,0,0,0,0,0]]):

    x_pos = np.random.randint(state[0][0]-variance, state[0][0] + variance)
    y_pos = np.random.randint(state[0][1]-variance, state[0][1] + variance)
    z_pos = np.random.randint(state[0][2]-variance, state[0][2] + variance)
    # x_rot = np.random.randint(-180, 180)
    # y_rot = np.random.randint(-180, 180)
    # z_rot = np.random.randint(-180, 180)
    x_rot = np.random.randint(state[0][3]-180, state[0][3] + 180)
    y_rot = np.random.randint(state[0][4]-180, state[0][4] + 180)
    z_rot = np.random.randint(state[0][5]-180, state[0][5] + 180)

    state = [[x_pos, y_pos, z_pos, x_rot, y_rot, z_rot]]

    return np.array(state)

def zero_reset():
    rsr = [[0, 0, 0, 0, 0, 0]]

    return np.array(rsr)

def mov(arr,axis,val):
    state = np.copy(arr)[0]

    if axis == "x":
        state[0] += val

    elif axis == "y":
        state[1] += val

    elif axis == "z":
        state[2] += val

    return state

def rot(arr,axis,val):# change state[3] = 20 +state[3] to state[3] = 360 +state[3] for all states
    state = np.copy(arr)[0]

    if axis == "x":
        index = 3

    elif axis == "y":
        index = 4

    elif axis == "z":
        index = 5

    state[index] += val

    if state[index] == -181:
        state[index] = 179

    if state[index] == 181:
        state[index] = -179

    return state

def act(a, cur_state, increment = val):
    # a = action to carry out: relates to an action in action table
    # cur_state = current state array of the part

    if action_table[a][0] == "mov":
        new_state = mov(cur_state, action_table[a][1], action_table[a][2] * increment)
    else:
        new_state = rot(cur_state, action_table[a][1], action_table[a][2] * increment)

    new = zero_reset()
    new[0] = new_state
    return new

def c_act(actions, cur_state):
    cur_state = np.add(cur_state, actions)

    for index in range(3,6):
        if cur_state[0,index] < -180:
            cur_state[0,index] = 180 + (cur_state[0,index]+180)

        if cur_state[0,index] > 180:
            cur_state[0,index] = -180 + (cur_state[0,index]-180)

    return cur_state
