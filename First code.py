import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import truncnorm


def get_startOpinions(N, dist):
    '''
    
    Returns a list of N random numbers. Potentially include other distributions?
    
    '''
    opinions = []
    
    if dist == "uniform_rand":
        opinions = np.random.rand(N)
    elif dist == "uniform_even":
        opinions = np.linspace(0, 1, N)
    elif dist == "normal_rand":
        opinions = truncnorm(a=0., b=1., scale = 1, loc=0).rvs(size=N)
    #elif dist == "Normal_Even":
        #base = np.linspace(N)
        #opinions = stats.norm.pdf(base, 0.5, 0.5)
    
    
    return list(opinions)

def calc_neighbours(opinions_list, confidence):
    '''
    
    returns a matrix representing neighbours. 1 represents a neighbour (opinion is within confidence distance), 0 represents not a neighbout (too far away).
    
    each row of the matrix represent the neighbours for each "member" in the opinions
    ie row 5 represents the neighbours for the 5th opinion in opinions_list
    
    
    '''
    neighbor_mat = []
    
    for opinion in opinions_list:
        abs_differences = np.absolute(opinions_list - opinion)
        neighbor_mat.append((abs_differences < confidence)*1)
        
    return neighbor_mat
    
def average_surrounding_opinions(opinions, neighbours):
    
    
    '''
    finds the average of the neighbouring opinions 
    '''
    averages = []
    
    for a in neighbours:
        averages.append(np.sum(a*opinions)/sum(a))
        
    return list(averages)

def run_basic_model(starting_opinions, num_repetitions, confidence, until_convergence = False, convergence_val = 0.0001):
    
    model = [starting_opinions]
    
    for i in range(1, num_repetitions):
        neighbours = calc_neighbours(model[i-1], confidence)
        model.append(average_surrounding_opinions(model[i-1], neighbours))
        if until_convergence == True:
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                break
    
    return np.array(model)

def run_model_1(starting_opinions, num_repetitions, confidence, change_rate, until_convergence = False, convergence_val = 0.0001):
    '''
    Adds the fact that as time goes on, the agents lean more towards an opinion of 1 at a steady rate
    '''
    
    model = [starting_opinions]
    
    for i in range(1, num_repetitions):
        neighbours = calc_neighbours(model[i-1], confidence)
        model.append(average_surrounding_opinions(model[i-1], neighbours))
        model[i] = [np.min([x + change_rate, 1]) for x in model[i]]
        if until_convergence == True:
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                break
    
    return np.array(model)

def check_convergence_ongoing(model_t1, model_t2, convergence_val):
    '''
    Checks if there is "change" in the model between model_t1 and model_t2
    '''
    
    if sum([abs(a-b) for a, b in zip(model_t1, model_t2)]) < convergence_val:
        return True
    else:
        return False


def plot_model_Graph_a(model):


    for a in range(1, len(model[:,0])+1):
        
        plt.scatter(model[a-1], [a] * (len(model[a-1])))
        
    plt.ylabel("time steps")
    plt.xlabel("opinion")
    
def plot_model_Graph_b(model):

    for a in range(0,len(model[0])-1):
         plt.plot(range(0, len(model[:,0])), model[:, a])
    

# below is just to test stuff
test = get_startOpinions(30, "uniform_even")


model = run_model_1(test, 20, 0.2, until_convergence = True, change_rate = 0.01)


plot_model_Graph_b(model)

