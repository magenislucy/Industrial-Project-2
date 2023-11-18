import numpy as np
import matplotlib.pyplot as plt

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
    distance = np.zeros([N, dim])
    # distance = np.zeros(N)

    for i in range(N):
        opinion = opinions_list[i,:]

        for j in range(N):
            distance[j,:] = np.absolute(opinions_list[j,:] - opinion)

        weight[i,:,:] = distance <= confidence  # work seperately, weight in 0 or 1
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
    
    # initialization
    model = get_startOpinions(num_agents, initial, dim = dim).reshape(1,num_agents,dim)

    for i in range(1, num_repetitions):
        weights = calc_weights(model[i-1], confidence, dim = dim)
        new_model = update_opinions(model[i-1], weights)
        model = np.concatenate((model, new_model.reshape(1,num_agents,dim)), axis=0)
        
        if until_convergence == True:
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                break
    
    return np.array(model)


num_agents = 10
initial = "uniform_even"
num_repetitions = 20
confidence = 0.25
a = run_model_0(num_agents, initial, num_repetitions, confidence, dim = 2, until_convergence = False, convergence_val = 0.0001)
print(a)