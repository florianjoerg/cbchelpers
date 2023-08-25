import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linear_fit(x, a,b):
    return a*x+b

# Custom plotting class that extends plt.plot()
class CustomPlotter():

    def plot(self, *args, ax=None, fit_function=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if fit_function is not None:
            if len(args) != 2:
                raise ValueError("Exactly two arguments are expected for x and y data.")

            x, y = args
            popt, pcov = curve_fit(fit_function, x, y)  # Fit the data using curve_fit
            fit_label = f'Fit: {fit_function.__name__}'
            ax.plot(np.linspace(x[0],x[-1],num=200), fit_function(np.linspace(x[0],x[-1],num=200), *popt), ls="--", color=kwargs["color"])

        return ax.plot(*args, **kwargs), popt, pcov  # Call the original plt.plot()

# Create an instance of the custom plotting class
# usage: custom_plt.plot(x,y,ax=axfit_function=linear_fit)
custom_plt = CustomPlotter()