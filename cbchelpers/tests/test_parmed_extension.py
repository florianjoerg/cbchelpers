import pytest
from cbchelpers.parmed_extension import enable_psf_from_scratch
from parmed.charmm import CharmmParameterSet, CharmmPsfFile
import os


def test_from_scratch():
    enable_psf_from_scratch()
    para_files = [
        "cbchelpers/data/toppar/toppar_drude_master_protein_2013f_lj025.str",
        "cbchelpers/data/toppar/im1h_d.str",
        "cbchelpers/data/toppar/oac_d_dummy.str",
        "cbchelpers/data/toppar/im1_d_dummy.str",
        "cbchelpers/data/toppar/hoac_d.str",
    ]

    molecules = {
        "IM1H": {"number": 1, "drude_mass": 0.4},
        "OAC": {"number": 1, "drude_mass": 0.4},
        "IM1": {"number": 1, "drude_mass": 0.4},
        "HOAC": {"number": 1, "drude_mass": 0.4},
    }

    params = CharmmParameterSet(*para_files)  # this parses also the residue sections
    # print(params.residues["IM1H"])
    psf = CharmmPsfFile.from_scratch(params, molecules)
    assert len(psf.residues) == sum(
        [molecules[molecule]["number"] for molecule in molecules]
    )
    psf.write_psf("test.psf")
    os.remove("test.psf")
    # psf_comp = CharmmPsfFile("cbchelpers/data/im1h_oac_im1_hoac_1.psf")
    # psf_comp.write_psf("pm_short.psf")
    # os.remove("pm_short.psf")
