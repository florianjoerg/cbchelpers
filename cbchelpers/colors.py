from enum import Enum
import warnings
import inspect
#import sys

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

    @classmethod
    def show(cls, short=False) -> None:
        for name, value in map(lambda x: (x.name,x.value), cls._member_map_.values()):
            if short and any(c.isdigit() for c in name):
                continue
            print(f"{name:8s} : {value}")

    def alpha(self, value: float):
        if ".plot(" in inspect.stack()[1].code_context[0]:
            warnings.warn(f"During plotting use plt.plot(x,y,*args, **kwargs, alpha={value})", UserWarning)
        if value < 0 or value > 1:
            raise RuntimeError("alpha value has to be between 0 and 1")
        value *= 100
        return self.value + hex(int(value))[2:] # we do not want the 0x identifier
    
    @classmethod
    @property
    def defaultcolors(cls) -> list:
        return cls.BLUE.value, cls.ORANGE.value, cls.RED.value, cls.GREEN.value, cls.YELLOW.value, cls.MINT.value, cls.BLACK.value

    @classmethod
    def set_as_default(cls) -> None:
        import matplotlib
        matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=cls.defaultcolors)
        

def test_defaultcolors():
    #uc = UniColors()
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    UniColors.show()
    print("now short")
    UniColors.show(short=True)
    UniColors.set_as_default()
    #matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=UniColors.defaultcolors)
    x = np.linspace(0, 20, 100)

    fig, axes = plt.subplots(nrows=2)
    
    for i in range(10):
        axes[0].plot(x, i * (x - 10)**2)
    
    for i in range(10):
        axes[1].plot(x, i * np.cos(x))
    
    plt.show()

def main():
    test_defaultcolors()

if __name__ == "__main__":
    main()
