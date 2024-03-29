import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Union

from .helpers import slope_from_file


def calc_cond(slope, temp=300, boxl=48.):
    # assumes slope in e^2Angstrom^2/ps
    # assumes temp in K
    # assumes boxl in Angstrom
    # returns conductivty in S/m
    k_b = 1.38065e-23 #J/K
    eV = 1.60218e-19
    vol_m3 = (boxl*1e-10)**3 
    prefactor = 1/(6*vol_m3*k_b*temp)
    slope_SI = slope * eV * eV * 1e-10 * 1e-10 / 1e-12
    #Only working for IM1H OAC system with 1000 molecules
    #conductivity = ( 1.602176634 * slope ) / (6 * 8.617332478 * 10**-8 * temp * boxl**3 )
    conductivity_Sm = prefactor * slope_SI
    return conductivity_Sm

def conductivity_from_file(filename: Path, xmin: int = None, xmax: int = None, boxl: float = 48., temperature: float = 300., verbose=False, unit="mS/cm"):
    slope = slope_from_file(filename, xmin=xmin, xmax=xmax, verbose=verbose)
    conductivity_Sm = calc_cond(slope, temperature, boxl)
    if unit=="mS/cm":
        return conductivity_Sm * 10
    #assume base unit S/m    
    return conductivity_Sm
    


class Charges:
    """Charges stores the the charge information for a simulation.
    Usually one will initialize the class and construct it from a file like so:
    charges = Charges()
    charges.from_file("charge_changes.out")
    the charge file should come from the custom ChargeReporter of OpenMM/Protex
    afterwards the charges for any step can be obtained via:
    charges.charges_at_step(step: int)
    """

    def __init__(self):
        self.data = {}
        self.dcd_save_freq = None
        self.sim_time_per_file = None
        self.sim_time = None
        self.update_steps = None
        self.steps_between_updates = None
        self.time_between_updates = None
        self.dt = None
        self.total_steps = None
        self.total_saved_steps = None

    def from_file(self, files: Union[list[str], str] or list[str]):
        """Expects a file generated by charge reporter in an Openmm Protex Simulation
        Header data right now not very flexible -> TODO: make better dict?
        Reads the charge data into a dict where the step is the key and the list of charges is the value
        Only stores the specific steps at which the charges change
        Use charges_at_step to get the right charges for any step
        """
        if isinstance(files, str):
            files = [files]
        prev_step = 0
        first = True
        for file in files:
            with open(file, "r") as f:
                data_comes = False
                for line in f.readlines():
                    line = line.split()
                    if data_comes:
                        step = int(line[0])
                        charge = [int(i.strip("[],")) for i in line[1:]]
                        # get step size from first two entries
                        # if prev_step and first: #bug:prev_step = 0 -> always false
                        if first:
                            first = False
                            step_size = step - prev_step
                        # if multiple files and the step counter ist started from begining calculate the current step manually
                        if prev_step >= step:
                            step = prev_step + step_size
                        self.data[step] = charge
                        self.total_steps = step
                        prev_step = step
                    if first and "dcd_save_freq:" in line:
                        self.dcd_save_freq = int(line[1].strip(","))
                        self.sim_time_per_file = int(line[3].strip(","))
                        self.update_steps = int(line[5].strip(","))
                        self.steps_between_updates = int(float(line[7].strip(",")))
                        self.dt = float(line[9].strip(","))
                        self.time_between_updates = self.steps_between_updates * self.dt

                    if line[0] == "Step" and line[1] == "Charges":
                        # From next line on the charge information for the steps comes
                        data_comes = True
            self.sim_time = self.sim_time_per_file * len(files)
            self.total_saved_steps = int(self.total_steps / self.dcd_save_freq)

    def charges_for_step(self, step: int) -> list:
        """Extracts the current charges for the specified step
        the charges are always valid until the step key is reached.
        so for keys 10000, 20000, the charges for 0-10000 are in the 10000 key-value,
        and from 10001 to 20000 in the 20000 key-value
        Keep in mind that MDAnalysis trajectories are indexed from 0,
        hence ts.frame starts at 0, but the step count in the charges
        dict is from 1, so ts.frame+1 is what you want to search for in the dict
        """
        step = step * self.dcd_save_freq
        try:
            key = next(
                val for val in self.data.keys() if val >= step
            )  # Crucial is i.e. 10000 before or after update if starting at 0 ??
        except StopIteration:
            raise RuntimeError("Step is larger than total Simulation Steps")
        return self.data[key]

    def info(self):
        print(f"Charges for {self.total_steps} steps")
        print(f"DCD Save Frequency: {self.dcd_save_freq}")
        print(f"Simulation time per file: {self.sim_time_per_file} ps")
        print(f"Total Simulation time: {self.sim_time} ps")
        print(f"Number of steps for each update: {self.update_steps}")
        print(f"Number of steps between each update: {self.steps_between_updates}")
        print(f"Timestep: {self.dt} ps")
        print(f"Time between two updates: {self.time_between_updates} ps")
        print(f"Total number of steps: {self.total_steps}")
        print(f"Total number of saved steps: {self.total_saved_steps}")
        print("Data stored in cls.data instance")


@dataclass(repr=True)
class UpdatePair:
    save_step: ClassVar[int] = None
    skip_value: ClassVar[int] = None
    name1_before: str
    idx1: int
    charge1_from: int
    name2_before: str
    idx2: int
    charge2_from: int
    step: int

    def __post_init__(self):
        self.idx1 = int(self.idx1)
        self.charge1_from = int(self.charge1_from)
        self.idx2 = int(self.idx2)
        self.charge2_from = int(self.charge2_from)
        self.name1_after = self._get_name_after(self.name1_before)
        self.name2_after = self._get_name_after(self.name2_before)
        self.charge1_to = self._get_charge_to(self.name1_after)
        self.charge2_to = self._get_charge_to(self.name2_after)
        self.step = int(self.step)
        # print(self.skip_value)
        # print(type(self.skip_value))
        if self.skip_value != 1 and self.skip_value is not None:
            warnings.warn("It is probably not working. Use a skip of 1!!!", UserWarning)

    @property
    def pos_in_arr(self):
        assert self.step % (self.save_step * self.skip_value) == 0
        return int(self.step / (self.save_step * self.skip)) - 1

    def get_frozenset_before(self):
        return frozenset([self.name1_before, self.name2_before])

    def _get_name_after(self, before_name):
        if before_name == "IM1H":
            return "IM1"
        elif before_name == "OAC":
            return "HOAC"
        elif before_name == "IM1":
            return "IM1H"
        elif before_name == "HOAC":
            return "OAC"

    def _get_charge_to(self, name):
        if name == "IM1H":
            return 1
        elif name == "OAC":
            return -1
        elif name == "IM1":
            return 0
        elif name == "HOAC":
            return 0

    def multiply_factor(self) -> int:
        if (self.charge1_to, self.charge2_to) == (1, -1):
            return -1
        elif (self.charge1_to, self.charge2_to) == (-1, 1):
            return 1
        elif (self.charge1_from, self.charge2_from) == (1, -1):
            return 1
        elif (self.charge1_from, self.charge2_from) == (-1, 1):
            return -1
        elif (self.charge1_from, self.charge2_from) == (-1, 0):
            return -1
        elif (self.charge1_from, self.charge2_from) == (0, -1):
            return 1
        elif (self.charge1_from, self.charge2_from) == (1, 0):
            return 1
        elif (self.charge1_from, self.charge2_from) == (0, 1):
            return -1
        else:
            raise RuntimeError(f"We have a case that is not covered... {self}")

    def correct_pos(self):
        """
        Assumes that the numpy position array is indexed at the position
        after the update occured, hence if the new charge is zero,
        we need the position of the previous frame
        """
        if (self.charge1_to, self.charge2_to) == (1, -1):
            return (0, 0)
        elif (self.charge1_to, self.charge2_to) == (-1, 1):
            return (0, 0)
        elif (self.charge1_from, self.charge2_from) == (1, -1):
            return (-1, -1)
        elif (self.charge1_from, self.charge2_from) == (-1, 1):
            return (-1, -1)
        elif (self.charge1_from, self.charge2_from) == (-1, 0):
            return (-1, 0)
        elif (self.charge1_from, self.charge2_from) == (0, -1):
            return (0, -1)
        elif (self.charge1_from, self.charge2_from) == (1, 0):
            return (-1, 0)
        elif (self.charge1_from, self.charge2_from) == (0, 1):
            return (0, -1)
        else:
            raise RuntimeError(f"We have a case that is not covered... {self}")


def get_pairs(files: Union[list[str], str]):
    if isinstance(files, str):
        files = [files]
    pairs: list[list[UpdatePair]] = []
    ctr = 0
    first = True
    for file in files:
        with open(file, "r") as f:
            if first:
                for line in f:
                    if '#"Step"' in line:
                        found_interval = False
                        while not found_interval:
                            try:
                                step1 = int(next(f).split(",")[0])
                                step2 = int(next(f).split(",")[0])
                                UpdatePair.save_step = int(step2 - step1)
                                found_interval = True
                                print(f"{UpdatePair.save_step=}")
                                # break
                            except ValueError:
                                pass

                f.seek(0)
                first = False
                # print(UpdatePair.save_step)
            offset = 0
            for line in f:
                offset += len(line)
                # print(line)
                # Start of update section
                if "Update trial" in line:
                    pairs.append([])
                    # Skipt the two "########" lines
                    line = next(f)
                    offset += len(line)
                    line = next(f)
                    offset += len(line)
                    curr_pointer_pos = offset
                    step = None
                    for update_line in f:
                        if re.search("^[0-9]+,[0-9]+.[0-9]+,", update_line):
                            step = int(update_line.split(",")[0])
                            break
                    f.seek(curr_pointer_pos)
                    for update_line in f:
                        offset += len(update_line)
                        if update_line.startswith("UpdatePair"):
                            line1 = update_line.strip().split(":")
                            pairs[ctr].append(UpdatePair(*line1[1:], step))
                            # print(pairs[ctr].pos_in_arr)
                        if "len(candidate_pairs)" in update_line:
                            # Section is over
                            ctr += 1
                        if "########" in update_line:
                            # finish update section
                            l = next(f)
                            offset += len(l)
                            break
    return pairs
