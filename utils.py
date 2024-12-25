import numpy as np

def rescale(x,mn=0,mx=1,a=0,b=1):
	return a + b*(x - mn)/(mx-mn)

class Color:
    RED = np.array([1,0,0])
    GREEN = np.array([0,1,0])
    BLUE = np.array([0,0,1])
    MAGENTA = np.array([1,0,1])
    CYAN = np.array([0,1,1])
    YELLOW = np.array([1,1,0])
    GREY = np.array([.6,.6,.6])
    LIGHTGREY = np.array([.9,.9,.9])
    WHITE = np.array([1,1,1])
