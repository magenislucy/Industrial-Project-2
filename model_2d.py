import numpy as np
import matplotlib.pyplot as plt
import Functionsonly as fc

def get_startOpinions(N, initial, dim = 2):
    '''
    Returns a opinion matrix. 
    N rows for individuals, dim columns for opinions.

    '''
    if initial == "uniform_rand":
        opinions = np.random.rand(N,dim)

    # even distribution with random permutation
    elif initial == "uniform_even":
        opinions = np.linspace(0, 1, N)
        np.random.shuffle(opinions)
        for i in range(dim-1):
            col = np.linspace(0, 1, N)
            np.random.shuffle(col)
            opinions = np.column_stack((opinions, col))

    elif initial == "normal_rand":
        mean = 0.5
        std_dev = 0.25
        random_numbers = np.random.normal(mean, std_dev, (N, dim))
        opinions=np.clip(random_numbers, 0, 1)
    
    return opinions


def calc_weights(opinions_list, confidence, dim = 2):
    '''
    returns a matrix representing neighbours (or weights?). 
    1 for a neighbour, 0 for not a neighbour.
    
    first idx of the tensor represents the weight for each "member",
    second idx represents how the member think of this person,
    third idx represents how the member think of this person on this topic.

    i.e. [0,1,2] represents how 0th member believe the 1st member on the 2nd topic
    
    '''
    N, dim = np.shape(opinions_list)
    weight = np.zeros([N, N, dim])
    distance = np.zeros([N, dim])  # the distance for a certain member
    # distance = np.zeros(N)

    for i in range(N):
        opinion = opinions_list[i,:]

        for j in range(N):
            # distance[j,:] = np.absolute(opinions_list[j,:] - opinion)  # seperately
            distance[j,:] = np.linalg.norm(opinions_list[j,:] - opinion)  # L2 norm

        weight[i,:,:] = distance <= confidence[0]  # work seperately, weight in 0 or 1
    return weight

def calc_weights1(opinions_list, confidence, dim = 2):
    N, dim = np.shape(opinions_list)
    weight = np.zeros([N, N, dim])
    distance = np.zeros([N, dim])  # the distance for a certain member

    for i in range(N):
        opinion = opinions_list[i,:]

        for j in range(N):
            distance[j,:] = np.absolute(opinions_list[j,:] - opinion)  # first compute seperately
            # distance[j,:] = np.linalg.norm(opinions_list[j,:] - opinion)  # L2 norm
            
            for k in range(dim):
                weight[i,j,k] = distance[j,k] <= confidence[k]  # work seperately, weight in 0 or 1
            
            # add the interest when outside the confidence this one but inside other topic
            for k in range(dim):
                if weight[i,j,k] == 0:
                    others_opn = 0.04 * (sum(weight[i,j,:]) - weight[i,j,k]) / dim
                    weight[i,j,k] = others_opn
    return weight

# def calc_weights2(opinions_list, confidence, inconfidence, dim = 2):
#     N, dim = np.shape(opinions_list)
#     weight = np.zeros([N, N, dim])
#     distance = np.zeros([N, dim])  # the distance for a certain member
#     disagree = np.zeros([N, dim])

#     for i in range(N):
#         opinion = opinions_list[i,:]

#         for j in range(N):
#             distance[j,:] = np.absolute(opinions_list[j,:] - opinion)  # first compute seperately
#             # distance[j,:] = np.linalg.norm(opinions_list[j,:] - opinion)  # L2 norm
            
#             for k in range(dim):
#                 weight[i,j,k] = distance[j,k] <= confidence[k]  # work seperately, weight in 0 or 1
#                 disagree[j,k] = distance[j,k] >= inconfidence[k]

#             # add the interest when outside the confidence this one but inside other topic
#             for k in range(dim):
#                 # if weight[i,j,k] == 0:
#                 others_near = 0.04 * (sum(weight[i,j,:]) - weight[i,j,k]) / dim

#                 # weight[i,:,:] = distance >= inconfidence[0]
#                 others_far = -1 * (sum(disagree[j,:])) / dim
#                 weight[i,j,k] += (others_near + others_far)
#                 weight[i,j,k] = np.clip(weight[i,j,k], 0, 1)
#     return weight

def calc_weights2(opinions_list, confidence, inconfidence, dim=2):
    N, _ = np.shape(opinions_list)
    weight = np.zeros([N, N, dim])

    distance = np.abs(opinions_list[:, np.newaxis, :] - opinions_list[np.newaxis, :, :])
    
    weight = (distance <= confidence).astype(float)
    outside_inconfidence = (distance >= inconfidence).astype(float)
    
    for i in range(N):
        for j in range(N):
            for k in range(dim):
                
                others_near = 0.04 * (np.sum(weight[i, j, :]) - weight[i, j, k]) / dim
                others_far = -1 * np.sum(outside_inconfidence[i, j, :]) / dim
                weight[i, j, k] += (others_near + others_far)

    weight = np.clip(weight, 0, 1)
    
    return weight

def update_opinions(opinions, weights):
    '''
    finds the average of the neighbouring opinions 

    '''
    new_opinions = opinions.copy()
    for i, weight in enumerate(weights):
        new_opinions[i,:] = np.average(opinions, axis=0, weights=weight)  # weighted average
    return new_opinions


def check_convergence_ongoing(model_t1, model_t2, convergence_val):
    '''
    Checks if there is "change" in the model between model_t1 and model_t2

    '''
    if sum([np.linalg.norm(opn_t1-opn_t2) for opn_t1, opn_t2 in zip(model_t1, model_t2)]) < convergence_val:
        return True
    else:
        return False


def run_model_0(num_agents, initial, num_repetitions, confidence, dim = 2, until_convergence = False, convergence_val = 0.0001):
    '''
    not consider death/birth, no effect between 2 topics, use a certain confidence
    
    '''
    # initialization
    model = get_startOpinions(num_agents, initial, dim = dim).reshape(1,num_agents,dim)

    # do iterations
    for i in range(1, num_repetitions):
        weights = calc_weights(model[i-1], confidence, dim = dim)  # generate new weights
        new_model = update_opinions(model[i-1], weights)  # update with new weights
        model = np.concatenate((model, new_model.reshape(1,num_agents,dim)), axis=0)
        
        if until_convergence == True:
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                break
    
    return np.array(model)

def run_model_1(num_agents, initial, num_repetitions, confidence, dim = 2, until_convergence = False, convergence_val = 0.0001):
    '''
    not consider death/birth, no effect between 2 topics, use a certain confidence
    
    '''
    # initialization
    model = get_startOpinions(num_agents, initial, dim = dim).reshape(1,num_agents,dim)

    # do iterations
    for i in range(1, num_repetitions):
        weights = calc_weights1(model[i-1], confidence, dim = dim)  # generate new weights
        new_model = update_opinions(model[i-1], weights)  # update with new weights
        model = np.concatenate((model, new_model.reshape(1,num_agents,dim)), axis=0)
        
        if until_convergence == True:
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                break
    
    return np.array(model)

def run_model_2(num_agents, initial, num_repetitions, confidence, inconfidence, dim = 2, until_convergence = False, convergence_val = 0.0001):
    '''
    not consider death/birth, affect between 2 topics, use a certain confidence
    
    '''
    # initialization
    model = get_startOpinions(num_agents, initial, dim = dim).reshape(1,num_agents,dim)

    # do iterations
    for i in range(1, num_repetitions):
        weights = calc_weights2(model[i-1], confidence, inconfidence, dim = dim)  # generate new weights
        new_model = update_opinions(model[i-1], weights)  # update with new weights
        model = np.concatenate((model, new_model.reshape(1,num_agents,dim)), axis=0)
        
        if until_convergence == True:
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                break
    
    return np.array(model)


num_agents = 101
initial = "uniform_rand"  # "uniform_even", "uniform_rand", "normal_rand"
num_repetitions = 20
confidence = [0.2, 0.2]
inconfidence = [0.9, 0.9]
dim = 2
# model = run_model_1(num_agents, initial, num_repetitions, confidence, dim = dim, until_convergence = False, convergence_val = 0.0001)
model = run_model_2(num_agents, initial, num_repetitions, confidence, inconfidence, dim = dim, until_convergence = False, convergence_val = 0.0001)
print(model)

for i in range(dim):
    fc.plot_model_Graph_b(model[:,:,i])
    plt.title(f'topic {i} with confidence interval {confidence[i]}', fontsize=15)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=11))
    # plt.savefig(f"topic{i}_2dplus.svg", format='svg')
    plt.show()