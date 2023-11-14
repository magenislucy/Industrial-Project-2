

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy.stats as stats
from scipy.stats import truncnorm
import math
import random
from statistics import mean


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
        
    #print(neighbor_mat)
        
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
    lessaverages = 0
    
    
    
    for ns in neighbours:
        for i, n in enumerate(ns):
            if opinions[i] != None:
                tempsum += n * opinions[i]
            elif n == True:
                lessaverages += 1
        
        if sum(ns) == 0:
            averages.append(0)
        else:
            averages.append(tempsum/(sum(ns)-lessaverages))
        lessaverages = 0
        tempsum = 0
        
        
    for i, prevop in enumerate(opinions):
        if prevop == None:
            averages[i] = None
        
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

def run_model_4_V1(starting_opinions, num_repetitions, confidence, agentsages, until_convergence = False, convergence_val = 0.0001):
    
    '''
    Only looks at deaths
    '''
    
    
    model = [starting_opinions]
    
    num_agents = len(model[0])
    
    
    
    
    for i in range(1, num_repetitions):
        
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

def run_model_4_V2(starting_opinions, num_repetitions, confidence, agentsages, shiftval = 0, until_convergence = False, convergence_val = 0.0001):
    
    '''
    Only looks at deaths
    '''

    deaths = []
    births = []
    
    
    
    numaliveagents = len(starting_opinions)
    totalnumagents = num_repetitions*numaliveagents +1
    totalunbornagents = totalnumagents - numaliveagents + 1
    
    
    model = [starting_opinions]
    for i in range(0, totalunbornagents):
        model[0].append(None)
        model[0][-1] = None
        agentsages.append(None)
        
    nexttobeborn = numaliveagents
    
    
    
    for i in range(1, num_repetitions):
        
        neighbours = calc_neighbours_V2(model[i-1], confidence)
        
        
        nextvals = average_surrounding_opinions_V2(model[i-1], neighbours)
        
        print(agentsages)
        
        for j, age in enumerate(agentsages):
            if age != None and age > 10:
                if nextvals[j] != None:
                    deaths.append([i-1, model[-1][j]])
                
                nextvals[j] = None
                
                
                newvalofbirth = random.uniform(0, 0.5)
                nextvals[nexttobeborn] = newvalofbirth
                births.append([i, newvalofbirth + shiftval])
                
                agentsages[nexttobeborn] = 1
                nexttobeborn += 1
                
        for i in range(0, len(nextvals)):
            if nextvals[i] != None:
                nextvals[i] += shiftval
        
        
        model.append(nextvals)
        
        
        
        
        newagentsages = []
        for x in agentsages:
            if x == None or x > 10:
                newagentsages.append(None)
            else:
                newagentsages.append(x+1)
        agentsages = newagentsages
        
        #model[-1] = [np.min([x + 0.1, 1]) for x in model[-1] if x != None]
        
        
        if until_convergence == True:
            
            if check_convergence_ongoing(model[i], model[i-1], convergence_val) == True:
                
                break
    
    
    return np.array(model), np.array(deaths), np.array(births)


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
    
    #plt.plot([0, 10], [0.2, 0.2], linestyle = "--")


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

def plot_model_Graph_d(model, deaths, births, x_label = "Iteration", y_label = "Opinion", axislabelsize = 14, confidence_dist = 0, title = "", showaverage = False):
    
    
    
    
    
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
    
    plt.scatter(deaths[:, 0], deaths[:, 1], marker = "x", color = "red")
    plt.scatter(births[:, 0], births[:, 1], marker = "o", color = "green")
    
    if confidence_dist != 0:
        plt.plot([-0.3, -0.3], [0, confidence_dist], color = "black")
        
        
    if showaverage == True:
        plt.plot(range(0, num_iterations), [mean(a for a in x if a != None) for x in model], linestyle = "--", color = "black")

num_agents = 20

test = get_startOpinions_1D(num_agents, "uniform_even")

#agentsages = [int(max(1, round(x*10/num_agents, 0))) for x in range(0, num_agents)]
agentsages = list(np.random.randint(0, 10, num_agents))



#print(average_surrounding_opinions_V2([1, None], [[1, 0],[0, 1]]))

model, deaths, births = run_model_4_V2(test, 100, 0.2, agentsages, shiftval = 0.02)
#model = run_model_3_V2(test, 10, 0.2, until_convergence = False, influentialagents = [0] , influencingconfidencevalues = [1], influencingweightvalues= [1])
#model = run_model_2(test, 20, 0.2, 0, rateofdecrease = 0.04, until_convergence = True)


#print(calc_howmanyconcensuses(model))

#print(deaths)

#plot_model_Graph_b(model)
plot_model_Graph_d(model, deaths, births, showaverage = True)

