from decimal import Decimal
from pathlib import Path

# assumes to get units in A^2/ps
# outputs cm2/s
#call with at least python3.6
#python get_diffusion.py filename.dat [-x min:max, -c,-l]

def linear_regression(x,y):
    """ 
    Return slope and axis intercept of a linear regression of the y.
    """
    n=len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(map(lambda a: a*a,x))
    sum_products = sum([x[i]*y[i] for i in range(n)])
    m = (sum_products - (sum_x*sum_y)/n) / (sum_x2-((sum_x**2)/n))
    b = (sum_y - m*sum_x)/n
    return m,b

def slope_from_file(filename: Path, xmin: int = None, xmax: int = None, verbose=False):
    #filename = sys.argv[1]
    f=open(filename,"r")

    column = 1
    start = None
    stop  = None
    #xmin  = None
    #xmax  = None
    # for i in range(len(sys.argv)):
    #     if (sys.argv[i]=="-c"):
    #         column = int(sys.argv[i+1])-1
    #     if (sys.argv[i]=="-l"):
    #         argument = (sys.argv[i+1]).split(":")        
    #         start    = int(argument[0])-1
    #         if (len(argument)>1):
    #             stop = int(argument[1])-1
    #     if (sys.argv[i]=="-x"):
    #         argument = (sys.argv[i+1]).split(":")  
    #         xmin     = float(argument[0])
    #         if (len(argument)>1):
    #             xmax = float(argument[1])
    if verbose:
        print("Filename: ",filename)
        print("Column: ",column+1)
        if (start is not None):
            print("Start:  ",start+1)
        if (stop is not None):
            print("Stop:  ",stop+1)
        if (xmin is not None):
            print("Xmin: ",xmin)
        if (xmax is not None):
            print("Xmax: ",xmax)
        
    # Reading data
    x = []
    y = []
    nframe = 0
    low    = None
    high   = None
    for line in f:
        nframe += 1
    #   empty lines    
        if (len(line)<2):
            break
    #   only lines between start and stop are considered    
        if (start is not None):
            if (nframe<start):
                continue
        if (stop is not None):
            if (nframe>stop):
                break
        
        buffer = line.split()
        current_x = float(buffer[0])
        current_y = float(buffer[column])

    #   only lines with x-values between xmin and xmax are considered    
        if (xmin is not None):
            if (current_x<xmin):
                continue
            
        if (xmax is not None):
            if (current_x>xmax):
                break
        x.append(current_x)
        y.append(current_y)
        
    #   lowest and highest value
        if (low is None):
            low = current_y
        if (high is None):
            high = current_y
        if (current_y<low):
            low = current_y
        if (current_y>high):
            high = current_y

    (m,_) = linear_regression(x,y)
    return m

def diffusion_from_file(filename: Path, xmin: int = None, xmax: int = None, verbose=False, unit="cms"):
    slope = slope_from_file(filename, xmin, xmax, verbose)
    #calculate Diffusioncoefficient
    #else:
    diffusion = slope / 6
    diffusion_cms = diffusion * 10**-4
    if verbose:
        print(f"Diffusioncoefficient: {Decimal(diffusion_cms):.10e} [cm^2/s]")
    if unit=="cms":
        return diffusion_cms
    else: #base units assume A^2/ps
        return diffusion
