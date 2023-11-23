import inspect
import warnings
from enum import Enum

# import sys
import matplotlib as mpl


class UniColors(Enum):
    @staticmethod
    def _register(name, colors):
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list(name, colors)
        my_cmap_r = my_cmap.reversed()
        mpl.colormaps.register(cmap=my_cmap)
        mpl.colormaps.register(cmap=my_cmap_r)

    @classmethod
    def _register_mpl_colormaps(cls):
        colors = [cls.BLUE.value, cls.ORANGE.value, cls.YELLOW.value, cls.RED.value]
        cls._register("mycmap", colors)
        colors = [cls.BLACK.value, cls.RED.value, cls.ORANGE.value, cls.YELLOW.value]
        cls._register("uni_inferno", colors)

    @classmethod
    def _build(cls):
        cls._register_mpl_colormaps()

    BLUE = "#0063a6"  # blue
    BLUE66 = "#0063a655"  # blue 66% noch da(55)
    BLUE33 = (
        "#0063a6AA"  # blue 33% noch da (AA -> 66% von 255(fully transparent) in hex)
    )
    ORANGE = "#dd4814"  # orange
    ORANGE66 = "#dd481455"  # orange 66%
    ORANGE33 = "#dd4814AA"  # orange 33%
    RED = "#a71c49"  # dark red/bordeaux
    RED33 = "#a71c49AA"  # dark red 33%
    GREEN = "#94c154"  # green
    GREEN66 = "#94c15455"  # green 66%
    GREEN33 = "#94c154AA"  # green 33%
    GRAY = "#666666"  # gray
    YELLOW = "#f6a800"  # yellow
    MINT = "#11897a"  # mint
    BLACK = "#000000"  # black
    WHITE = "#ffffff"  # white

    # _ignore_ = ["_original_colors"]
    # _original_colors = mpl.rcParams['axes.prop_cycle']

    @classmethod
    def show(cls, short=False) -> None:
        for name, value in map(lambda x: (x.name, x.value), cls._member_map_.values()):
            if short and any(c.isdigit() for c in name):
                continue
            print(f"{name:8s} : {value}")

    def alpha(self, value: float):
        if ".plot(" in inspect.stack()[1].code_context[0]:
            warnings.warn(
                f"During plotting use plt.plot(x,y,*args, **kwargs, alpha={value})",
                UserWarning,
            )
        if value < 0 or value > 1:
            raise RuntimeError("alpha value has to be between 0 and 1")
        value *= 100
        return self.value + hex(int(value))[2:]  # we do not want the 0x identifier

    @classmethod
    @property
    def defaultcolors(cls) -> list:
        return (
            cls.BLUE.value,
            cls.ORANGE.value,
            cls.RED.value,
            cls.GREEN.value,
            cls.YELLOW.value,
            cls.MINT.value,
            cls.BLACK.value,
        )

    @classmethod
    def set_as_default(cls) -> None:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cls.defaultcolors)

    # does not work why???
    # @classmethod
    # def unset_as_default(cls) -> None:
    #     mpl.rcParams['axes.prop_cycle'] = cls._original_colors


class MedUniColors(Enum):
    @staticmethod
    def _register(name, colors):
        my_cmap = mpl.colors.LinearSegmentedColormap.from_list(name, colors)
        my_cmap_r = my_cmap.reversed()
        mpl.colormaps.register(cmap=my_cmap)
        mpl.colormaps.register(cmap=my_cmap_r)

    @classmethod
    def _register_mpl_colormaps(cls):
        colors = [cls.BLUE.value, cls.LIGHTBLUE.value, cls.GREEN.value, cls.CORAL.value]
        cls._register("meduni", colors)
        # colors = [cls.BLACK.value, cls.RED.value, cls.ORANGE.value, cls.YELLOW.value]
        # cls._register("uni_inferno", colors)
        colors = [
            cls.LIGHTBLUE.value,
            cls.LIGHTBLUE1.value,
            cls.LIGHTBLUE2.value,
            cls.LIGHTBLUE3.value,
            cls.LIGHTBLUE4.value,
        ]
        cls._register("meduni_lightblue", colors)
        colors = [
            cls.GREEN.value,
            cls.GREEN1.value,
            cls.GREEN2.value,
            cls.GREEN3.value,
            cls.GREEN4.value,
        ]
        cls._register("meduni_green", colors)
        colors = [
            cls.CORAL.value,
            cls.CORAL1.value,
            cls.CORAL2.value,
            cls.CORAL3.value,
            cls.CORAL4.value,
        ]
        cls._register("meduni_coral", colors)

    @classmethod
    def _build(cls):
        cls._register_mpl_colormaps()

    BLUE = "#111d4e"  # blue
    LIGHTBLUE = "#5fb4e5"
    LIGHTBLUE1 = "#97CFEC"
    LIGHTBLUE2 = "#B5DCF1"
    LIGHTBLUE3 = "#CFE8F6"
    LIGHTBLUE4 = "#E7F3FB"
    GREEN = "#2f8e91"
    GREEN1 = "#84C9BC"
    GREEN2 = "#A8D7CD"
    GREEN3 = "#C8E5DF"
    GREEN4 = "#E4F2EF"
    CORAL = "#f0a794"
    CORAL1 = "#F8C0B0"
    CORAL2 = "#FAD1C4"
    CORAL3 = "#FCE1D8"
    CORAL4 = "#FDF0EB"
    BLACK = "#000000"  # black
    WHITE = "#ffffff"  # white

    # _ignore_ = ["_original_colors"]
    # _original_colors = mpl.rcParams['axes.prop_cycle']

    @classmethod
    def show(cls, short=False) -> None:
        for name, value in map(lambda x: (x.name, x.value), cls._member_map_.values()):
            if short and any(c.isdigit() for c in name):
                continue
            print(f"{name:8s} : {value}")

    def alpha(self, value: float):
        if ".plot(" in inspect.stack()[1].code_context[0]:
            warnings.warn(
                f"During plotting use plt.plot(x,y,*args, **kwargs, alpha={value})",
                UserWarning,
            )
        if value < 0 or value > 1:
            raise RuntimeError("alpha value has to be between 0 and 1")
        value *= 100
        return self.value + hex(int(value))[2:]  # we do not want the 0x identifier

    @classmethod
    @property
    def defaultcolors(cls) -> list:
        return (
            cls.BLUE.value,
            cls.LIGHTBLUE.value,
            cls.GREEN.value,
            cls.CORAL.value,
            cls.BLACK.value,
        )

    @classmethod
    @property
    def lightblues(cls) -> list:
        return (
            cls.LIGHTBLUE.value,
            cls.LIGHTBLUE1.value,
            cls.LIGHTBLUE2.value,
            cls.LIGHTBLUE3.value,
            cls.LIGHTBLUE4.value,
        )

    @classmethod
    @property
    def greens(cls) -> list:
        return (
            cls.GREEN.value,
            cls.GREEN1.value,
            cls.GREEN2.value,
            cls.GREEN3.value,
            cls.GREEN4.value,
        )

    @classmethod
    @property
    def corals(cls) -> list:
        return (
            cls.CORAL.value,
            cls.CORAL1.value,
            cls.CORAL2.value,
            cls.CORAL3.value,
            cls.CORAL4.value,
        )

    @classmethod
    def set_as_default(cls) -> None:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cls.defaultcolors)


UniColors._build()
MedUniColors._build()


class UniColorsContext:
    # define enter and exit to use it as context manager, i.e.:
    # with UniColors:
    #   plt.plot(...) #uses as color the uni colors
    def __enter__(self):
        self.original_colors = mpl.rcParams["axes.prop_cycle"]
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=UniColors.defaultcolors)

    def __exit__(self, exc_type, exc_value, exc_tb):
        mpl.rcParams["axes.prop_cycle"] = self.original_colors


class MedUniColorsContext:
    # define enter and exit to use it as context manager, i.e.:
    # with UniColors:
    #   plt.plot(...) #uses as color the uni colors
    def __enter__(self):
        self.original_colors = mpl.rcParams["axes.prop_cycle"]
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=MedUniColors.defaultcolors)

    def __exit__(self, exc_type, exc_value, exc_tb):
        mpl.rcParams["axes.prop_cycle"] = self.original_colors


def test_defaultcolors():
    # uc = UniColors()
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np

    UniColors.show()
    print("now short")
    UniColors.show(short=True)
    UniColors.set_as_default()
    print(matplotlib.colormaps["mycmap"])
    # matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=UniColors.defaultcolors)
    x = np.linspace(0, 20, 100)

    fig, axes = plt.subplots(nrows=2)

    for i in range(10):
        axes[0].plot(x, i * (x - 10) ** 2)

    for i in range(10):
        axes[1].plot(x, i * np.cos(x))

    plt.show()


def main():
    test_defaultcolors()


if __name__ == "__main__":
    main()
