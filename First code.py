

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.stats as stats
from scipy.stats import truncnorm


def get_startOpinions_1D(N, dist):
    '''
    
    Returns a list of N random numbers. Potentially include other distributions?
    
    '''
    opinions = []
    
    if dist == "uniform_rand":
        opinions = np.random.rand(N)
    elif dist == "uniform_even":
        opinions = np.linspace(0, 1, N)
    elif dist == "normal_rand":
        mean = 0.5
        std_dev = 0.25
        random_numbers = np.random.normal(mean, std_dev, size=100)
        opinions=np.clip(random_numbers, 0, 1)
    #elif dist == "normal_even":
        
    
    
    return list(opinions)

def calc_neighbours(opinions_list, confidence):
    '''
    
    returns a matrix representing neighbours. 1 represents a neighbour (opinion is within confidence distance), 0 represents not a neighbout (too far away).
    
    each row of the matrix represent the neighbours for each "member" in the opinions
    ie row 5 represents the neighbours for the 5th opinion in opinions_list
    
    
    '''
    neighbor_mat = []
    for opinion in opinions_list:
        abs_differences = np.absolute(list(np.array(opinions_list) - opinion))
        neighbor_mat.append((abs_differences <= confidence)*1)
    return neighbor_mat


def calc_neighbours_V2(opinions_list, confidence):
    '''
    deals with Dead agents
    
    returns a matrix representing neighbours. 1 represents a neighbour (opinion is within confidence distance), 0 represents not a neighbout (too far away).
    
    each row of the matrix represent the neighbours for each "member" in the opinions
    ie row 5 represents the neighbours for the 5th opinion in opinions_list
    
    
    '''
    neighbor_mat = []
    
    for opinion in opinions_list:
        
        
        abs_differences = []
        for otheropinion in opinions_list:
            
            if opinion != None:
                if otheropinion == None:
                    abs_differences.append(np.inf)
                else:
                    
                    abs_differences.append(np.absolute(otheropinion - opinion))
        
        neighbor_mat.append(list(np.array(abs_differences) <= confidence)*1)
        
        
        
    return neighbor_mat

def average_surrounding_opinions(opinions, neighbours):
    
    
    '''
    finds the average of the neighbouring opinions 
    '''
    averages = []
    
    for n in neighbours:
        
        averages.append(np.average(n*np.array(opinions), weights = n))
        
    return list(averages)

def average_surrounding_opinions_V2(opinions, neighbours):
    
    
    '''
    Deals with Dead agent
    finds the average of the neighbouring opinions 
    '''
    averages = []
    tempsum = 0
    for ns in neighbours:
        for op in opinions:
            for n in ns:
                if op != None:
                    tempsum += n * op
        
        
        if sum(ns) == 0:
            averages.append(0)
        else:
            averages.append(tempsum/sum(ns))
        
    return list(averages)

def average_surrounding_opinions_weighted(opinions, neighbours, agentweights):
    
    
    '''
    finds the average of the neighbouring opinions 
    '''
    averages = []
    
    
    for n in neighbours:
        
        
        averages.append(np.average(n*opinions, weights = n*agentweights)) #doesnt work yet
        
    return list(averages)





def check_convergence_ongoing(model_t1, model_t2, convergence_val):
    '''
    Checks if there is "change" in the model between model_t1 and model_t2
    '''
    
    if sum([abs(a-b) for a, b in zip(model_t1, model_t2)]) < convergence_val:
        return True
    else:
        return False

def calc_howmanyconcensuses(model, tolerance = 0.00001):
    
    finaliterance = model[-1]
    
    consensus_points = []
    points_sorted = sorted(finaliterance)
    curr_point = points_sorted[0]
    curr_cluster = [curr_point]
    for point in points_sorted[1:]:
        if point <= curr_point + tolerance:
            curr_cluster.append(point)
        else:
            consensus_points.append(curr_cluster)
            curr_cluster = [point]
        curr_point = point
    consensus_points.append(curr_cluster)
    
    
    return len(consensus_points)




def run_model_0(starting_opinions, num_repetitions, confidence, until_convergence = False, convergence_val = 0.0001):
    
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

def run_model_2(starting_opinions, num_repetitions, confidence, change_rate, until_convergence = False, convergence_val = 0.0001, rateofdecrease = 0.01):
    '''
    Adds the fact that as time goes on, the agents lean more towards an opinion of 1 at a steady rate
    Adds a linear decrease in confidence size
    '''
    
    
    
    model = [starting_opinions]
    
    for i in range(1, num_repetitions):
        newconfidence = np.max([confidence - i*rateofdecrease, 0])
        
        neighbours = calc_neighbours(model[i-1], newconfidence)
        
        model.append(average_surrounding_opinions(model[i-1], neighbours))
        
        model[i] = [np.min([x + change_rate, 1]) for x in model[i]]
        
        if until_convergence == True:
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                break
    
    return np.array(model)

def run_model_3_V1(starting_opinions, num_repetitions, confidence, influentialagents, influencingconfidencevalues, until_convergence = False, convergence_val = 0.0001, ):
    
    '''
    Runs model with some values having a strong range or not JUST ADJUSTS CONFIDENCE INTERVALS
    '''
    model = [starting_opinions]
    
    confidences = [confidence]*len(starting_opinions)
    
    for i in range(len(influentialagents)):
        confidences[influentialagents[i]] *= influencingconfidencevalues[i]    
    
    for i in range(1, num_repetitions):
        
        neighbours = calc_neighbours(model[i-1], confidences)
        
        model.append(average_surrounding_opinions(model[i-1], neighbours))
        
        if until_convergence == True:
            
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                
                break
    
    
    return np.array(model)

def run_model_3_V2(starting_opinions, num_repetitions, confidence, influentialagents, influencingconfidencevalues, influencingweightvalues, until_convergence = False, convergence_val = 0.0001, ):
    
    '''
    Runs model with some values having a strong range or not adjust confidence interval AND influencing weight
    '''
    model = [starting_opinions]
    
    confidences = [confidence]*len(starting_opinions)
    weights = [1]*len(starting_opinions)
    
    for i in range(len(influentialagents)):
        confidences[influentialagents[i]] *= influencingconfidencevalues[i] 
        weights[influentialagents[i]] *= influencingweightvalues[i]
    
    print(confidences)
    
    for i in range(1, num_repetitions):
        
        neighbours = calc_neighbours(model[i-1], confidences)
        
        model.append(average_surrounding_opinions_weighted(model[i-1], neighbours, weights))
        
        if until_convergence == True:
            
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                
                break
    
    
    return np.array(model)

def run_model_4(starting_opinions, num_repetitions, confidence, until_convergence = False, convergence_val = 0.0001):
    
    '''
    
    '''
    
    
    model = [starting_opinions]
    
    num_agents = len(model[0])
    
    agentsages = [int(max(1, round(x*10/num_agents, 0))) for x in range(0, num_agents)]
    
    
    for i in range(1, num_repetitions):
        
        print(i)
        print(agentsages)
        print(model[i-1])
        
        neighbours = calc_neighbours_V2(model[i-1], confidence)
        
        nextvals = average_surrounding_opinions_V2(model[i-1], neighbours)
        
        for i, age in enumerate(agentsages):
            if age > 10:
                nextvals[i] = None
        
        model.append(nextvals)
        
        agentsages = [x+1 for x in agentsages]
        
        if until_convergence == True:
            
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                
                break
    
    
    return np.array(model)




def plot_model_Graph_a(model):


    for a in range(1, len(model[:,0])+1):
        
        plt.scatter(model[a-1], [a] * (len(model[a-1])))
        
    plt.ylabel("time steps")
    plt.xlabel("opinion")
 
def plot_model_Graph_b(model, x_label = "Iteration", y_label = "Opinion", axislabelsize = 14, confidence_dist = 0, title = ""):
    
    num_iterations = len(model[:,0])
    num_agents = len(model[0])
    
    colors = pl.cm.jet(np.linspace(0,1,num_agents))
    
    for a in range(0,num_agents-1):
         plt.plot(range(0, num_iterations), model[:, a], color = colors[a])
         
    plt.xlabel(x_label, fontsize = axislabelsize)
    plt.ylabel(y_label, fontsize = axislabelsize)
    
    plt.title(title)

    #plt.ylim(-0.05, 1)    

    plt.xticks(range(0, num_iterations))
    

    if confidence_dist != 0:
        plt.plot([-0.3, -0.3], [0, confidence_dist], color = "black")

def plot_model_Graph_c(model, x_label = "Iteration", y_label = "Opinion", axislabelsize = 14, confidence_dist = 0, title = ""):
    
    '''
    plots average 
    '''
    num_iterations = len(model[:,0])
    num_agents = len(model[0])
    
    colors = pl.cm.jet(np.linspace(0,1,num_agents))
    
    for a in range(0,num_agents-1):
         plt.plot(range(0, num_iterations), model[:, a], color = colors[a])
         
    plt.xlabel(x_label, fontsize = axislabelsize)
    plt.ylabel(y_label, fontsize = axislabelsize)

    plt.title(title)    

    plt.xticks(range(0, num_iterations))
    

    if confidence_dist != 0:
        plt.plot([-0.3, -0.3], [0, confidence_dist], color = "black")





test = get_startOpinions_1D(20, "uniform_even")



model = run_model_4(test, 20, 0.2)
#model = run_model_3_V2(test, 20, 0.2, until_convergence = False, influentialagents = [0] , influencingconfidencevalues = [5], influencingweightvalues= [100])



#print(calc_howmanyconcensuses(model))

plot_model_Graph_b(model, title = "model 0, 100 start, 0.2 eps")

