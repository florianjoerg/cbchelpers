import datetime
import math
import os
import warnings
from collections import OrderedDict

from parmed.charmm import CharmmParameterSet, CharmmPsfFile
from parmed.charmm._charmmfile import CharmmFile
from parmed.charmm.parameters import _fit_IC_table
from parmed.exceptions import CharmmError, ParameterWarning
from parmed.modeller import PatchTemplate, ResidueTemplate
from parmed.periodic_table import AtomicNum, element_by_mass
from parmed.topologyobjects import (
    Angle,
    Atom,
    AtomType,
    Bond,
    Dihedral,
    DrudeAnisotropy,
    DrudeAtom,
    ExtraPoint,
    Improper,
)


def enable_psf_from_scratch():
    """
    Import this function to use ParmEd for generating a psf file.
    It adds the new classmethod from_scratch to CharmmPsfFile.
    The topology reading of CharmmParameterSet is also extended.
    Just call the function at the beginning of your script and
    you are ready to use the standard ParmEd functions, with the additions included.

    Example:
    from cbchelpers.parmed_extension import enable_psf_from_scratch
    enable_psf_from_scratch()
    molecules = {"RESI1": "number": n_res1, "drude_mass": 0.4,
                 "RESI2": "number": n_res2, "drude_mass": 0.4
                 ...
                 }
    #Note: if the keyword "drude_mass" is missing, no drude particles will be added to the psf.
    params = CharmmParameterSet(parameter_files)
    psf = CharmmPsfFile.from_scratch(params, molecules)
    #i.e. write psf to file:
    psf.write("my_generated_psf.psf")
    """
    CharmmParameterSet.read_topology_file = read_topology_file
    CharmmPsfFile.from_scratch = from_scratch


def read_topology_file(self, tfile):
    """
    Reads _only_ the atom type definitions from a topology file. This is
    unnecessary for versions 36 and later of the CHARMM force field.

    Reads also the topology information into ResidueTemplates and ResiduePatches.

    Parameters
    ----------
    tfile : str
        Name of the CHARMM topology file to read
    """
    print("USING CUSTOM TOPOLOGY READER")
    conv = CharmmParameterSet._convert
    if isinstance(tfile, str):
        own_handle = True
        f = iter(CharmmFile(tfile))
    else:
        own_handle = False
        f = tfile
    hpatch = tpatch = None  # default Head and Tail patches
    residues = OrderedDict()
    patches = OrderedDict()
    hpatches = OrderedDict()
    tpatches = OrderedDict()
    line = next(f)
    line_index = 0
    skip_adding_residue = False
    try:
        while line:
            line = line.strip()
            if line[:4].upper() == "MASS":
                words = line.split()
                try:
                    idx = conv(
                        words[1], int, "atom type", line_index=line_index, line=line
                    )
                    name = words[2].upper()
                    mass = conv(
                        words[3], float, "atom mass", line_index=line_index, line=line
                    )
                except IndexError:
                    raise CharmmError("Could not parse MASS section of %s" % tfile)
                # The parameter file might or might not have an element name
                try:
                    elem = words[4].upper()
                    if len(elem) == 2:
                        elem = elem[0] + elem[1].lower()
                    atomic_number = AtomicNum[elem]
                except (IndexError, KeyError):
                    # Figure it out from the mass
                    atomic_number = AtomicNum[element_by_mass(mass)]
                atype = AtomType(
                    name=name, number=idx, mass=mass, atomic_number=atomic_number
                )
                self.atom_types_str[atype.name] = atype
                self.atom_types_int[atype.number] = atype
                self.atom_types_tuple[(atype.name, atype.number)] = atype
            elif line[:4].upper() == "DECL":
                pass  # Not really sure what this means
            elif line[:4].upper() == "DEFA":
                words = line.split()
                if len(words) < 5:
                    warnings.warn(
                        "DEFA line has %d tokens; expected 5" % len(words),
                        ParameterWarning,
                    )
                else:
                    it = iter(words[1:5])
                    for tok, val in zip(it, it):
                        if val.upper() == "NONE":
                            val = None
                        if tok.upper().startswith("FIRS"):
                            hpatch = val
                        elif tok.upper() == "LAST":
                            tpatch = val
                        else:
                            warnings.warn(f"DEFA patch {val} unknown", ParameterWarning)
            elif line[:4].upper() in ("RESI", "PRES"):
                restype = line[:4].upper()
                # Get the residue definition
                words = line.split()
                resname = words[1].upper()
                if resname in self.residues:
                    warnings.warn(f"Replacing residue {resname}", ParameterWarning)
                # Assign default patches
                hpatches[resname] = hpatch
                tpatches[resname] = tpatch
                try:
                    charge = float(words[2])
                except (IndexError, ValueError):
                    warnings.warn(f"No charge for {resname}", ParameterWarning)
                if restype == "RESI":
                    res = ResidueTemplate(resname)
                elif restype == "PRES":
                    res = PatchTemplate(resname)
                else:
                    assert False, "restype != RESI or PRES"
                skip_adding_residue = False
                line = next(f)
                group = []
                ictable = []
                # lonepairs = {}
                while line:
                    line = line.lstrip()
                    if line[:5].upper() == "GROUP":
                        if group:
                            res.groups.append(group)
                        group = []
                    elif line[:4].upper() == "ATOM":
                        words = line.split()
                        name = words[1].upper()
                        type = words[2].upper()
                        charge = float(words[3])
                        mass = self.atom_types_str[type].mass
                        if "ALPHA" in words:
                            # This is a polarizable atom.
                            alpha = float(words[words.index("ALPHA") + 1])
                            thole = 1.3
                            drude_type = "DRUD"
                            if "THOLE" in words:
                                thole = float(words[words.index("THOLE") + 1])
                            if "TYPE" in words:
                                drude_type = words[words.index("TYPE") + 1]

                            atom = DrudeAtom(
                                name=name,
                                type=type,
                                atomic_number=AtomicNum[element_by_mass(mass)],
                                charge=charge,
                                alpha=alpha,
                                thole=thole,
                                drude_type=drude_type,
                                mass=mass,
                            )
                        elif words[1].startswith("LP"):
                            # This is a lonepair.
                            atom = ExtraPoint(
                                name=name,
                                type=type,
                                charge=charge,
                                mass=mass,
                                atomic_number=0,
                            )
                            # lonepairs[name] = atom #used?
                        else:
                            atom = Atom(
                                name=name,
                                type=type,
                                charge=charge,
                                mass=mass,
                                atomic_number=AtomicNum[element_by_mass(mass)],
                            )
                        group.append(atom)
                        res.add_atom(atom)
                    elif line[:6].upper() == "DELETE":
                        words = line.split()
                        name = words[2].upper()
                        entity_type = words[1].upper()
                        if entity_type == "ATOM":
                            res.delete_atoms.append(name)
                        elif entity_type == "IMPR":
                            res.delete_impropers.append(words[2:5])
                        else:
                            warnings.warn(
                                f'WARNING: Ignoring "{line.strip()}" because entity type '
                                f"{entity_type} not used.",
                                ParameterWarning,
                            )
                    elif line.strip().upper() and line.split()[0].upper() in (
                        "BOND",
                        "DOUBLE",
                    ):
                        it = iter([w.upper() for w in line.split()[1:]])
                        for a1, a2 in zip(it, it):
                            if restype == "PRES":
                                # Patches can have bonds that refer to atoms not in the patch, so store these in a list of tuples
                                order = 1
                                if line.split()[0].upper() == "DOUBLE":
                                    order = 2
                                res.add_bonds.append((a1, a2, order))
                                continue

                            if a1.startswith("-"):
                                res.head = res[a2]
                                continue
                            if a2.startswith("-"):
                                res.head = res[a1]
                                continue
                            if a1.startswith("+"):
                                res.tail = res[a2]
                                continue
                            if a2.startswith("+"):
                                res.tail = res[a1]
                                continue
                            res.add_bond(a1, a2)
                    elif line[:4].upper() == "CMAP":
                        pass
                    elif line[:5].upper() == "DONOR":
                        pass
                    elif line[:6].upper() == "ACCEPT":
                        pass
                    elif line[:8].upper() == "LONEPAIR":
                        # See: https://www.charmm.org/charmm/documentation/by-version/c40b1/params/doc/lonepair/
                        # TODO: This currently doesn't handle some formats, like Note 3 in the above URL
                        words = line.split()
                        lptype_keyword = words[1][0:4].upper()
                        if not skip_adding_residue and lptype_keyword not in [
                            "BISE",
                            "RELA",
                        ]:
                            warnings.warn(
                                f"LONEPAIR type {words[1]} not supported; only BISEctor and "
                                "RELAtive supported",
                                ParameterWarning,
                            )
                            skip_adding_residue = True
                            break
                        a1, a2, a3, a4 = words[2:6]
                        keywords = {
                            words[index][0:4].upper(): float(words[index + 1])
                            for index in range(6, len(words), 2)
                        }
                        r = abs(keywords["DIST"])  # angstrom #abs to be sure
                        theta = keywords["ANGL"]  # degrees
                        phi = keywords["DIHE"]  # degrees
                        lptypes = {"BISE": "bisector", "RELA": "relative"}
                        lonepair = (
                            lptypes[lptype_keyword],
                            a1,
                            a2,
                            a3,
                            a4,
                            r,
                            theta,
                            phi,
                        )  # TODO: Define a LonePair object?
                        res.lonepairs.append(lonepair)

                    elif line[:2].upper() == "IC":
                        words = line.split()[1:]
                        ictable.append(
                            (
                                [w.upper() for w in words[:4]],
                                [float(w) for w in words[4:]],
                            )
                        )
                    elif line[:3].upper() == "END":
                        break
                    elif line[:5].upper() == "PATCH":
                        it = iter(line.split()[1:])
                        for tok, val in zip(it, it):
                            if val.upper() == "NONE":
                                val = None
                            if tok.upper().startswith("FIRS"):
                                hpatches[resname] = val
                            elif tok.upper().startswith("LAST"):
                                tpatches[resname] = val
                    elif line[:4].upper() in ("IMPR", "IMPH"):
                        it = iter(w.upper() for w in line.split()[1:])
                        for a1, a2, a3, a4 in zip(it, it, it, it):
                            res._impr.append((a1, a2, a3, a4))
                            if a2[0] == "-" or a3[0] == "-" or a4 == "-":
                                res.head = res[a1]
                    elif line[:10].upper() == "ANISOTROPY":
                        words = line.split()
                        atoms = [res[name] for name in words[1:5]]
                        keywords = {
                            words[index].upper(): float(words[index + 1])
                            for index in range(5, len(words), 2)
                        }
                        a11 = float(keywords["A11"])
                        a22 = float(keywords["A22"])
                        # shouldnt it be put on the drude atom?
                        atoms[0].anisotropy = DrudeAnisotropy(*atoms, a11=a11, a22=a22)
                    elif line[:4].upper() in ("RESI", "PRES", "MASS"):
                        # Back up a line and bail
                        break
                    line = next(f)
                if group:
                    res.groups.append(group)
                _fit_IC_table(res, ictable)
                if skip_adding_residue:
                    # Do not add this residue to the lookup library
                    continue
                elif restype == "RESI":
                    residues[resname] = res
                elif restype == "PRES":
                    patches[resname] = res
                else:
                    assert False, "restype != RESI or PRES"
                # We parsed a line we need to look at. So don't update the
                # iterator
                continue
            # Get the next line and cycle through
            line = next(f)
            line_index += 1
    except StopIteration:
        pass

    # Go through the patches and add the appropriate one
    self.patches.update(patches)
    for resname, res in residues.items():
        patch_name = hpatches[resname]
        if patch_name is not None:
            try:
                res.first_patch = self.patches[patch_name]
            except KeyError:
                warnings.warn(f"Patch {patch_name} not found", ParameterWarning)

        patch_name = tpatches[resname]
        if patch_name is not None:
            try:
                res.last_patch = self.patches[patch_name]
            except KeyError:
                warnings.warn(f"Patch {patch_name} not found", ParameterWarning)
    # Now update the residues and patches with the ones we parsed here
    self.residues.update(residues)

    if own_handle:
        f.close()


# CharmmParameterSet.read_topology_file = read_topology_file


@classmethod
def from_scratch(
    cls, params: CharmmParameterSet, molecules: dict[str, dict[str, int, str, float]]
) -> CharmmPsfFile:
    """
    make a psf file from rtf and prm
    params is a CharmmParameterSet object, but using the custom reat_topology_file function!
    molecules = {"RESI1": "number": n_res1, "drude_mass": 0.4,
                 "RESI2": "number": n_res2, "drude_mass": 0.4
                 ...
                 }
    #Note: if the keyword "drude_mass" is missing, no drude particles will be added to the psf.
    """

    def _get_idx_to_name(name, template):
        for atom in template.atoms:
            if name == atom.name:
                return atom.idx

    def _drude_charge(alpha, kdrude):
        ccelec = 332.0716
        qdip = 2 * kdrude * alpha / ccelec
        qdip = -math.sqrt(abs(qdip))
        qdip = round(qdip, 4)
        return qdip

    def _anisotropy(a11, a22, kdrude):
        a33 = 3 - a11 - a22
        k11 = 1 / a11
        k22 = 1 / a22
        k33 = 1 / a33
        k11 = kdrude * k11
        k22 = kdrude * k22
        k33 = kdrude * k33
        k33 = k33 - kdrude
        k11 = k11 - kdrude - k33
        k22 = k22 - kdrude - k33
        return [k11, k22, k33]

    psf = cls()
    new_idx = 0
    for molecule, settings in molecules.items():
        # print(molecule)
        n_residues = settings["number"]
        is_drude = "drude_mass" in settings
        if is_drude:
            drude_mass = settings["drude_mass"]
            # TODO: individual DRUD types are not covered
            try:
                k_drude = params.bond_types.get(("X", "DRUD")).k
            except AttributeError:
                warnings.warn("Using default Drude constant of 1000 kcal/(A**2 mol)")
                k_drude = 1000
        template_residue = params.residues[molecule]
        lonepair_names = {
            a1: pos
            for pos, (lptype, a1, a2, a3, a4, r, theta, phi) in enumerate(
                template_residue.lonepairs
            )
        }
        for idx in range(n_residues):
            idx_mapping = {}  # old:new
            resnum = idx + 1
            for old_idx, atom in enumerate(template_residue.atoms):
                # print(atom)
                name = atom.name
                resname = atom.residue.name
                chain = resname
                segid = resname
                attype = atom.type
                charge = atom.charge
                mass = atom.mass
                atomic_number = atom.atomic_number
                if isinstance(atom, DrudeAtom):
                    alpha = atom.alpha
                    thole = atom.thole
                    drude_type = atom.drude_type
                    drude_charge = _drude_charge(alpha, k_drude)
                    new_charge = charge - drude_charge
                    new_mass = mass - drude_mass

                    new_atom = Atom(
                        name=name,
                        type=attype,
                        charge=new_charge,
                        mass=new_mass,
                        atomic_number=atomic_number,
                    )
                    new_atom.props = ["0", f"{alpha:.5f}", f"{thole:.5f}"]
                    psf.add_atom(new_atom, resname, resnum, chain=chain, segid=segid)
                    if is_drude:  # do we need this
                        drude_atom = DrudeAtom(
                            name=f"D{name}",
                            type=drude_type,
                            atomic_number=0,
                            parent=new_atom,
                            charge=drude_charge,
                            alpha=alpha,
                            thole=thole,
                            drude_type=drude_type,
                            mass=drude_mass,
                        )
                        if isinstance(atom.anisotropy, DrudeAnisotropy):
                            drude_atom.anisotropy = atom.anisotropy
                        drude_atom.props = ["0", "0.00000", "0.00000"]
                        psf.add_atom(
                            drude_atom, resname, resnum, chain=chain, segid=segid
                        )
                        # new_idx+=1 #we added an extra drude atom
                # handle lonepairs
                # TODO: fix the negative charge for im1h,im1
                elif isinstance(atom, ExtraPoint):
                    new_atom = ExtraPoint._copy(atom)
                    lp_name = atom.name
                    lptype, a1, a2, a3, a4, r, theta, phi = template_residue.lonepairs[
                        lonepair_names.get(lp_name)
                    ]
                    assert a1 == name  # remove
                    # some check how is this with the lps???
                    offset = len(psf.atoms) - old_idx
                    # idx2 = offset + _get_idx_to_name(a2, template_residue)
                    idx2 = idx_mapping[_get_idx_to_name(a2, template_residue)]
                    idx3 = idx_mapping[_get_idx_to_name(a3, template_residue)]
                    idx4 = idx_mapping[_get_idx_to_name(a4, template_residue)]
                    if lptype == "bisector":
                        # bisector uses the negative distance
                        # but in the rtf it is always specified as positive value
                        neg_r = -r
                        frame = CharmmPsfFile._get_frame3(
                            psf, idx2, idx4, idx3, neg_r, theta, phi
                        )

                    elif lptype == "relative":
                        # relative uses the positive distance, as specified in the rtf
                        frame = CharmmPsfFile._get_frame3(
                            psf, idx2, idx4, idx3, r, theta, phi
                        )
                    else:
                        print(lptype, "problem", molecule)
                    new_atom.frame_type = frame
                    if is_drude:
                        new_atom.props = ["-1", "0.00000", "0.00000"]
                    psf.add_atom(new_atom, resname, resnum, chain=chain, segid=segid)
                elif isinstance(atom, Atom):
                    new_atom = Atom._copy(atom)
                    if hasattr(atom, "anisotropy"):
                        # important since it is not directly on the drudes, maybe change this
                        new_atom.anisotropy = atom.anisotropy
                    if is_drude:
                        new_atom.props = ["0", "0.00000", "0.00000"]
                    psf.add_atom(new_atom, resname, resnum, chain=chain, segid=segid)
                idx_mapping[old_idx] = new_idx
                new_idx += 1
                if isinstance(atom, DrudeAtom):
                    new_idx += 1

            for bond in template_residue.bonds:
                old_idx1 = bond.atom1.idx
                old_idx2 = bond.atom2.idx
                new_idx1 = idx_mapping[old_idx1]
                new_idx2 = idx_mapping[old_idx2]
                atom1 = psf.atoms[new_idx1]
                atom2 = psf.atoms[new_idx2]
                psf.bonds.append(Bond(psf.atoms[new_idx1], psf.atoms[new_idx2]))
                # print(bond.atom1.name, bond.atom1.idx, bond.atom2.name, bond.atom2.idx)

            # add angles
            def _findangles(bondlist):
                # think it works, but seems way to umstädnlich
                # TODO: refactor
                angles = []
                atom_idx_map = {}
                for i in range(len(bondlist)):
                    bond1 = bondlist[i]
                    b1_atom1, b1_atom2 = bond1.atom1, bond1.atom2
                    b1_idx1, b1_idx2 = (
                        idx_mapping[bond1.atom1.idx],
                        idx_mapping[bond1.atom2.idx],
                    )
                    atom_idx_map[b1_atom1] = b1_idx1
                    atom_idx_map[b1_atom2] = b1_idx2
                    if isinstance(b1_atom1, ExtraPoint) or isinstance(
                        b1_atom2, ExtraPoint
                    ):
                        continue
                    for j in range(i + 1, len(bondlist)):
                        bond2 = bondlist[j]
                        b2_atom1, b2_atom2 = bond2.atom1, bond2.atom2
                        b2_idx1, b2_idx2 = (
                            idx_mapping[bond2.atom1.idx],
                            idx_mapping[bond2.atom2.idx],
                        )
                        atom_idx_map[b2_atom1] = b2_idx1
                        atom_idx_map[b2_atom2] = b2_idx2
                        if isinstance(b2_atom1, ExtraPoint) or isinstance(
                            b2_atom2, ExtraPoint
                        ):
                            continue
                        # print(
                        #    f"atoms:{b1_atom1=}, {b1_atom2=},{b2_atom1=}, {b2_atom2=}"
                        # )
                        middle_atom = list(
                            {b1_atom1, b1_atom2}.intersection({b2_atom1, b2_atom2})
                        )
                        # print(f"{middle_atom=}")
                        if len(middle_atom) == 1:
                            outer_atoms = list(
                                {b1_atom1, b1_atom2}.symmetric_difference(
                                    {b2_atom1, b2_atom2}
                                )
                            )
                            # print(f"{outer_atoms=}")
                            a1 = psf.atoms[atom_idx_map[outer_atoms[0]]]
                            a2 = psf.atoms[atom_idx_map[middle_atom[0]]]
                            a3 = psf.atoms[atom_idx_map[outer_atoms[1]]]
                            angle = Angle(a1, a2, a3)
                            # angle = Angle(
                            #    outer_atoms[0], middle_atom[0], outer_atoms[1]
                            # )
                            angles.append(angle)
                return angles

            angles = _findangles(template_residue.bonds)
            # should i work somehow with atom.angle_partners???
            for angle in angles:
                psf.angles.append(angle)
                # psf.angles[-1].funct = 5 # urey-bradley #?? is in psf.py

            # add dihedrals
            def _finddihedrals(bondlist):
                # think it works, but seems way to umstädnlich
                # TODO: refactor
                dihedrals = []
                atom_idx_map = {}
                for i in range(len(bondlist)):
                    bond1 = bondlist[i]
                    b1_atom1, b1_atom2 = bond1.atom1, bond1.atom2
                    b1_idx1, b1_idx2 = (
                        idx_mapping[bond1.atom1.idx],
                        idx_mapping[bond1.atom2.idx],
                    )
                    atom_idx_map[b1_atom1] = b1_idx1
                    atom_idx_map[b1_atom2] = b1_idx2
                    if isinstance(b1_atom1, ExtraPoint) or isinstance(
                        b1_atom2, ExtraPoint
                    ):
                        continue
                    for j in range(i + 1, len(bondlist)):
                        bond2 = bondlist[j]
                        b2_atom1, b2_atom2 = bond2.atom1, bond2.atom2
                        b2_idx1, b2_idx2 = (
                            idx_mapping[bond2.atom1.idx],
                            idx_mapping[bond2.atom2.idx],
                        )
                        atom_idx_map[b2_atom1] = b2_idx1
                        atom_idx_map[b2_atom2] = b2_idx2
                        if isinstance(b2_atom1, ExtraPoint) or isinstance(
                            b2_atom2, ExtraPoint
                        ):
                            continue
                        # print(
                        #     f"atoms:{b1_atom1.name},{b1_atom2.name},{b2_atom1.name},{b2_atom2.name}"
                        # )
                        for k in range(j + 1, len(bondlist)):
                            bond3 = bondlist[k]
                            b3_atom1, b3_atom2 = bond3.atom1, bond3.atom2
                            b3_idx1, b3_idx2 = (
                                idx_mapping[bond3.atom1.idx],
                                idx_mapping[bond3.atom2.idx],
                            )
                            atom_idx_map[b3_atom1] = b3_idx1
                            atom_idx_map[b3_atom2] = b3_idx2
                            if isinstance(b3_atom1, ExtraPoint) or isinstance(
                                b3_atom2, ExtraPoint
                            ):
                                continue
                            if b2_atom1 in [b1_atom1, b1_atom2] or b2_atom2 in [
                                b1_atom1,
                                b1_atom2,
                            ]:
                                # we have a bond between b1 and b2
                                if b3_atom1 in [b1_atom1, b1_atom2] or b3_atom2 in [
                                    b1_atom1,
                                    b1_atom2,
                                ]:
                                    # we have a bond between b1 and b3
                                    # this means b1 is the middle bond
                                    outer_atom1 = list(
                                        {b1_atom1, b1_atom2}.symmetric_difference(
                                            {b2_atom1, b2_atom2}
                                        )
                                    )
                                    outer_atom1 = [
                                        atom
                                        for atom in outer_atom1
                                        if atom in [b2_atom1, b2_atom2]
                                    ]
                                    middle_atom1 = list(
                                        {b1_atom1, b1_atom2}.intersection(
                                            {b2_atom1, b2_atom2}
                                        )
                                    )
                                    outer_atom2 = list(
                                        {b1_atom1, b1_atom2}.symmetric_difference(
                                            {b3_atom1, b3_atom2}
                                        )
                                    )
                                    outer_atom2 = [
                                        atom
                                        for atom in outer_atom2
                                        if atom in [b3_atom1, b3_atom2]
                                    ]
                                    middle_atom2 = list(
                                        {b1_atom1, b1_atom2}.intersection(
                                            {b3_atom1, b3_atom2}
                                        )
                                    )
                                elif b3_atom1 in [b2_atom1, b2_atom2] or b3_atom2 in [
                                    b2_atom1,
                                    b2_atom2,
                                ]:
                                    # we have a bond between b2 and b3
                                    # this means b2 is the middle bond
                                    outer_atom1 = list(
                                        {b2_atom1, b2_atom2}.symmetric_difference(
                                            {b1_atom1, b1_atom2}
                                        )
                                    )
                                    outer_atom1 = [
                                        atom
                                        for atom in outer_atom1
                                        if atom in [b1_atom1, b1_atom2]
                                    ]
                                    middle_atom1 = list(
                                        {b2_atom1, b2_atom2}.intersection(
                                            {b1_atom1, b1_atom2}
                                        )
                                    )
                                    outer_atom2 = list(
                                        {b2_atom1, b2_atom2}.symmetric_difference(
                                            {b3_atom1, b3_atom2}
                                        )
                                    )
                                    outer_atom2 = [
                                        atom
                                        for atom in outer_atom2
                                        if atom in [b3_atom1, b3_atom2]
                                    ]
                                    middle_atom2 = list(
                                        {b2_atom1, b2_atom2}.intersection(
                                            {b3_atom1, b3_atom2}
                                        )
                                    )
                            else:
                                # bond3 is in the middle
                                outer_atom1 = list(
                                    {b3_atom1, b3_atom2}.symmetric_difference(
                                        {b2_atom1, b2_atom2}
                                    )
                                )
                                outer_atom1 = [
                                    atom
                                    for atom in outer_atom1
                                    if atom in [b2_atom1, b2_atom2]
                                ]
                                middle_atom1 = list(
                                    {b3_atom1, b3_atom2}.intersection(
                                        {b2_atom1, b2_atom2}
                                    )
                                )
                                outer_atom2 = list(
                                    {b3_atom1, b3_atom2}.symmetric_difference(
                                        {b1_atom1, b1_atom2}
                                    )
                                )
                                outer_atom2 = [
                                    atom
                                    for atom in outer_atom2
                                    if atom in [b1_atom1, b1_atom2]
                                ]
                                middle_atom2 = list(
                                    {b3_atom1, b3_atom2}.intersection(
                                        {b1_atom1, b1_atom2}
                                    )
                                )
                            if (
                                len(outer_atom1)
                                == len(middle_atom1)
                                == len(middle_atom2)
                                == len(outer_atom2)
                                == 1
                            ) and len(
                                {
                                    outer_atom1[0],
                                    middle_atom1[0],
                                    middle_atom2[0],
                                    outer_atom2[0],
                                }
                            ) == 4:
                                a1 = psf.atoms[atom_idx_map[outer_atom1[0]]]
                                a2 = psf.atoms[atom_idx_map[middle_atom1[0]]]
                                a3 = psf.atoms[atom_idx_map[middle_atom2[0]]]
                                a4 = psf.atoms[atom_idx_map[outer_atom2[0]]]
                                dihedral = Dihedral(a1, a2, a3, a4)
                                # dihedral = Dihedral(
                                #     outer_atom1[0],
                                #     middle_atom1[0],
                                #     middle_atom2[0],
                                #     outer_atom2[0],
                                # )
                                dihedrals.append(dihedral)
                return dihedrals

            dihedrals = _finddihedrals(template_residue.bonds)
            for dihedral in dihedrals:
                psf.dihedrals.append(dihedral)
            # add impropers
            for improper in template_residue._impr:
                name1, name2, name3, name4 = improper
                idx1 = idx_mapping[_get_idx_to_name(name1, template_residue)]
                idx2 = idx_mapping[_get_idx_to_name(name2, template_residue)]
                idx3 = idx_mapping[_get_idx_to_name(name3, template_residue)]
                idx4 = idx_mapping[_get_idx_to_name(name4, template_residue)]
                new_at1, new_at2, new_at3, new_at4 = (
                    psf.atoms[idx1],
                    psf.atoms[idx2],
                    psf.atoms[idx3],
                    psf.atoms[idx4],
                )
                new_improper = Improper(new_at1, new_at2, new_at3, new_at4)
                psf.impropers.append(new_improper)

            # TODO: CMAPS

    # add bonds between parent and drude atom
    for pos, atom in enumerate(psf.atoms):
        # print(f"{atom.name},{atom.bond_partners=}")
        if isinstance(atom, DrudeAtom):
            # add parent drude bonds
            psf.bonds.append(Bond(psf.atoms[pos - 1], atom))
            # fix anisotropy atoms
            if isinstance(atom.anisotropy, DrudeAnisotropy):
                offset = min([atom.idx for atom in atom.residue.atoms])
                names = [atom.name for atom in atom.residue.atoms]
                atom1 = atom.anisotropy.atom1
                atom2 = atom.anisotropy.atom2
                atom3 = atom.anisotropy.atom3
                atom4 = atom.anisotropy.atom4
                a11, a22 = atom.anisotropy.a11, atom.anisotropy.a22
                idx1 = offset + names.index(atom1.name)
                idx2 = offset + names.index(atom2.name)
                idx3 = offset + names.index(atom3.name)
                idx4 = offset + names.index(atom4.name)
                parent_atom = psf.atoms[idx1]
                at2 = psf.atoms[idx2]
                at3 = psf.atoms[idx3]
                at4 = psf.atoms[idx4]
                k11, k22, k33 = _anisotropy(a11, a22, k_drude)
                atom.anisotropy = DrudeAnisotropy(
                    parent_atom,
                    at2,
                    at3,
                    at4,
                    a11,
                    a22,
                    k11=k11,
                    k22=k22,
                    k33=k33,
                )
    psf.title = [
        "* Created by ParmEd, CharmmPsfFile.from_scratch function.",
        "* HANDLE WITH CAUTION!",
        f"* DATE: {datetime.datetime.now():%Y-%m-%d   %H:%M}   CREATED BY USER: {os.getlogin()}",
    ]
    psf.flags = ["EXT", "XPLOR"]  # what is CHEQ, CMAP?
    if is_drude:
        psf.flags.append("DRUDE")
    return psf

    # what i have to do:
    # psf.atoms = struct.atoms
    # psf.residues = struct.residues
    # psf.bonds = struct.bonds
    # psf.angles = struct.angles
    # psf.urey_bradleys = struct.urey_bradleys
    # psf.dihedrals = struct.dihedrals
    # psf.impropers = struct.impropers
    # psf.acceptors = struct.acceptors
    # psf.donors = struct.donors
    # psf.groups = struct.groups
    # psf.cmaps = struct.cmaps

    # psf.bond_types = struct.bond_types
    # psf.angle_types = struct.angle_types
    # psf.urey_bradley_types = struct.urey_bradley_types
    # psf.dihedral_types = struct.dihedral_types
    # psf.improper_types = struct.improper_types
    # psf.cmap_types = struct.cmap_types

    # for atom in psf.atoms:
    #     atom.type = typeconv(atom.type)
    #     if atom.atom_type is not UnassignedAtomType:
    #         atom.atom_type.name = typeconv(atom.atom_type.name)

    # # If no groups are defined, make each residue its own group
    # if not psf.groups:
    #     for residue in psf.residues:
    #         chg = sum(a.charge for a in residue)
    #         if chg < 1e-4:
    #             psf.groups.append(Group(residue[0], 1, 0))
    #         else:
    #             psf.groups.append(Group(residue[0], 2, 0))
    #     psf.groups.nst2 = 0

    # return psf


# CharmmPsfFile.from_scratch = from_scratch


def main():
    enable_psf_from_scratch()
    para_files = [
        "toppar/toppar_drude_master_protein_2013f_lj025.str",
        "toppar/im1h_d.str",
        "toppar/oac_d_dummy.str",
        "toppar/im1_d_dummy.str",
        "toppar/hoac_d.str",
    ]

    molecules = {
        "IM1H": {"number": 1, "drude_mass": 0.4},
        "OAC": {"number": 1, "drude_mass": 0.4},
        "IM1": {"number": 1, "drude_mass": 0.4},
        "HOAC": {"number": 1, "drude_mass": 0.4},
    }

    params = CharmmParameterSet(*para_files)  # this parses also the residue sections
    # print(params.residues["IM1H"])
    psf_comp = CharmmPsfFile("im1h_oac_im1_hoac_1.psf")

    psf = CharmmPsfFile.from_scratch(params, molecules)
    print(len(psf.residues))
    psf.write_psf("test.psf")
    psf_comp.write_psf("pm_short.psf")


def test_psf_read_write():
    # lesen und schreiben geht mit drude, wenn es immer CharmmPsfFile bleibt
    psf = CharmmPsfFile("im1h_oac_im1_hoac_1.psf")
    psf.write_psf("pm_short.psf")


if __name__ == "__main__":
    main()
    # test_psf_read_write()
