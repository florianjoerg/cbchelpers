# Helper functions for analysis of MD Trajectories
# Usage:
# import in other files for analysis like so:

# import sys
# sys.path.append("/home/florian/pythonfiles")
# from helpers import msd_com, msd_mj

import logging
import os
import sys
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np


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
    #base units assume A^2/ps
    return diffusion

def setup_logger(filename=None, replace=True):
    if filename is None:
        filename = "logfile.log"
    if not replace and os.path.exists(filename):
        os.rename(filename, "old_" + filename)
    logging.basicConfig(filename=filename, filemode="w")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    return logger

class Tee:
    def __init__(self, out1, out2):
        self.out1 = out1
        self.out2 = out2

    def write(self, *args, **kwargs):
        self.out1.write(*args, **kwargs)
        self.out2.write(*args, **kwargs)

    def flush(self):
        pass

def log_print(filename=None, replace=True):
    if filename is None:
        filename = "logfile.log"
    if not replace and os.path.exists(filename):
        os.rename(filename, "old_" + filename)
    f = open(filename, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, f)


def prettify(string: str) -> str:
    orig_string = string
    string = string.upper()
    if string == "IM1H":
        return "$\mathrm{Im_1H^+}$"
    if string == "OAC":
        return "$\mathrm{OAc^-}$"
    if string == "IM1":
        return "$\mathrm{Im_1}$"
    if string == "HOAC":
        return "HOAc"
    if string == "TFA":
        return "$\mathrm{TFA^-}$"
    if string == "AC":
        return "Ac"
    if string == "IM":
        return "Im"
    if string == "MEOH":
        return "MeOH"
    if string == "MEOH2":
        return "$\mathrm{MeOH_2^+}$"
    if string == "HPTS":
        return "$\mathrm{HPTS^{4-}}$"
    if string == "HPTSH":
        return "$\mathrm{HPTSH^{3-}}$"
    else:
        warnings.warn(f"Could not prettify {orig_string}", UserWarning)
        return orig_string


def rPBC(coor1, coor2, boxl):
    dx = coor1[0] - coor2[0]
    if dx > boxl / 2:
        dx = boxl - dx
    elif dx <= -boxl / 2:
        dx = boxl + dx
    dy = coor1[1] - coor2[1]
    if dy > boxl / 2:
        dy = boxl - dy
    elif dy <= -boxl / 2:
        dy = boxl + dy
    dz = coor1[2] - coor2[2]
    if dz > boxl / 2:
        dz = boxl - dz
    elif dz <= -boxl / 2:
        dz = boxl + dz
    # return np.sqrt(dx * dx + dy * dy + dz * dz)
    return np.array([dx, dy, dz])


def msd_com(com_array):
    """
    Calculates mean squared displacement using the Fast Correlation Algorithm in the package tidynamics
    Dependency: tidynamics
    input: numpy array with shape (timesteps, particles, xyz)
    returns: numpy array with shape (timestep, msd)
    """
    import tidynamics

    com_array = com_array.astype(np.float64)
    time, n_particles, ndim = com_array.shape
    msds_by_particle = np.zeros((time, n_particles))
    for particle in range(n_particles):
        msds_by_particle[:, particle] = tidynamics.msd(com_array[:, particle, :])
    msd = msds_by_particle.mean(axis=1)
    return msd


def msd_mj(*args: tuple[np.array, int]):
    """
    Calculates mj using the Fast Correlation Algorithm in the package tidynamics
    Dependency: tidynamics
    input: any number of tuples: (numpy array with shape (timesteps, n_particles, ndim), int charge or numpy array of charges)
    numpy array can either be 1D with n_particle charge values or 2D with n_timesteps, n_particle number of charges
    input arrays must all have same timesteps and dimension, number of particles may vary between two separate arrays
    Example call of function: msdmj = msd_mj((com_cat, 1), (com_an, -1)) or msd_mj((com_cat, np.array([[1,1,1,][1,1,1]])) where charges are for three particles with two timesteps
    int charges and array charges can be mixed for different com array inputs

    if only one input tuple is given, just the autocorelation of it will be calculated, eg. sigma++ for ie im1h or sigma-- for oac (analog to msdMJcrossterms) not working for cossterms like simga+-
    """
    import tidynamics

    time, particles, ndim = args[0][0].shape
    mj = np.zeros((time, ndim), dtype=np.float64)
    for msd, charge in args:
        if isinstance(charge, int):
            mj_tmp = msd.sum(axis=1) * charge
        elif isinstance(charge, (np.ndarray, np.generic)):
            if charge.ndim > 2:
                print(
                    f"Error in msd_mj function! Charge array has more dimensions ({charge.ndim}) than allowed."
                )
                print(
                    "Only 1D (charges for positions, same at each timestep) or 2D (charges for each timestep and position) numpy arrays are allowed"
                )
                print("Exiting...")
                sys.exit()
            if (
                charge.shape[0] == time and charge.shape[1] == msd.shape[1]
            ):  # time, particles
                # print("Using custom charges for each timestep and particle")
                mj_tmp = (msd * charge[:, :, np.newaxis]).sum(axis=1)
            elif charge.shape[0] == msd.shape[1]:  # particles
                # print("Using custom charges for each particle")
                mj_tmp = (msd * charge[np.newaxis, :, np.newaxis]).sum(axis=1)
        else:
            print("Error in msd_mj function!!!")
            print(
                "charge must be suplied either as one (1) int or as numpy array with 1 or 2 dimensions"
            )
            print("Exiting...")
            sys.exit()
        mj += mj_tmp
    msd_mj = tidynamics.msd(mj)
    # msd_mj = tidynamics.correlation(args[0][0].sum(axis=1)*args[0][1], args[1][0].sum(axis=1)*args[1][1])

    return msd_mj


# Note: to use msd_mj with charge array do sth like:
# charges = [[charge]*n_particles]*time
# charges = np.asarray(charges)
# msdmj = msd_mj((com_cat. charges), (com_an, -1))

if __name__ == "__main__":
    print("Hi")
    print("You can use the functions from here in your analysis routines")
    print("Just import them, i.e. like:")
    print('sys.path.append("/home/florian/pythonfiles")')
    print("from helpers import msd_com, msd_mj")
    print("Hope it helps!")
    print("Bye")
