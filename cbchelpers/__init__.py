"""Various functions and methods to facilitate my work"""

# Add imports here
from ._version import __version__
from .cbchelpers import *
from .colors import UniColors, UniColorsContext
from .cond_helpers import Charges, UpdatePair, get_pairs
from .helpers import log_print, msd_com, msd_mj, prettify
from .plot_helpers import custom_plt, linear_fit
