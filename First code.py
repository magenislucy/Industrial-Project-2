import numpy as np
import matplotlib.pyplot as plt


def get_randOpinions(N, dist = "Uniform"):
    '''
    
    Returns a list of N random numbers. Potentially include other distributions?
    
    '''
    opinions = []
    
    if dist == "Uniform":
        opinions = np.random.rand(N)
    
    
    return opinions

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
        
    return averages

def run_model(starting_opinions, num_repetitions, confidence):
    
    
    model = [[0 for x in range(len(starting_opinions))] for y in range(num_repetitions)]
    
    model[0] = starting_opinions
    
    for i in range(1, num_repetitions):
        neighbours = calc_neighbours(model[i-1], confidence)
        model[i] = average_surrounding_opinions(model[i-1], neighbours)
    
    return model



# below is just to test stuff
test = get_randOpinions(100)
neighbours = (calc_neighbours(test, 0.1))
print(test)

for a, x in enumerate(neighbours):
    print(str(round(test[a], 2)) + " - " +str(x))


averages = average_surrounding_opinions(test, neighbours)

print(averages)

simulation = run_model(test, 10, 0.2)

num_repetitions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


for a in num_repetitions:
    plt.scatter(simulation[a-1], [a] * len(simulation[0]))
    
plt.ylabel("time steps")
plt.xlabel("opinion")
plt.invert_yaxsis()

