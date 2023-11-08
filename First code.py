import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import truncnorm
from Functionsonly import *



test = get_startOpinions(30, "uniform_even")


model = run_model_1(test, 20, 0.2, until_convergence = True, change_rate = 0.01)


plot_model_Graph_b(model)

