from enum import Enum
import warnings
import inspect
import sys
class UniColors(Enum):
    BLUE = '#0063a6'   # blue
    BLUE66 = '#0063a655' # blue 66% noch da(55)
    BLUE33 = '#0063a6AA' # blue 33% noch da (AA -> 66% von 255(fully transparent) in hex)
    ORANGE = '#dd4814'   # orange
    ORANGE66 = '#dd481455' # orange 66%
    ORANGE33 = '#dd4814AA' # orange 33% 
    RED = '#a71c49'   # dark red/bordeaux
    RED33 = '#a71c49AA' # dark red 33%
    GREEN = '#94c154'   # green
    GREEN66 = '#94c15455' # green 66%
    GREEN33 = '#94c154AA' # green 33%
    GRAY = '#666666'   # gray
    YELLOW = '#f6a800'   # yellow
    MINT = '#11897a'   # mint
    BLACK = '#000000'    # black
    WHITE = '#ffffff' # white

    def alpha(self, value: float):
        if ".plot(" in inspect.stack()[1].code_context[0]:
            warnings.warn(f"During plotting use plt.plot(x,y,*args, **kwargs, alpha={value})", UserWarning)
        if value < 0 or value > 1:
            raise RuntimeError("alpha value has to be between 0 and 1")
        value *= 100
        return self.value + hex(int(value))[2:] # we do not want the 0x identifier
        
