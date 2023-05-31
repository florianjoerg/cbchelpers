"""Fuctions to generate a dielectric spectrum and fit md0mdt as wells as msdmj data"""
from __future__ import annotations

import json
import sys
from pprint import pprint

import numpy as np
from GenDiCon_Lithium.data import CorrelationFunctions
from GenDiCon_Lithium.spectrum import Frequency, Prefactor, Spectrum

from .correlation_fits import InputClass, fit


def prepare_mdmd_fromJson(json_file):
    input = InputClass()
    try:
        JsonFile = json_file #sys.argv[1]
        with open(JsonFile) as infile:
            data = json.load(infile)
        input.fromJson(data)
        input.info()
    except IndexError:
        print('\n! Error!')
        print('!\t Json input file is missing: python3 mdmd_fit.py ___.json. ')
        print('!\t Writing example input fit_example.json')
        input.ExampleInput()
        sys.exit()
    except FileNotFoundError:
        print('\n! Error!')
        print('!\t Json input file %s not found!'%JsonFile)
        sys.exit()

    return input

def fit_mdmd(input: InputClass, json_file: str | None=None, verbose=False):
    fit_results = fit(input, json_file = json_file, verbose=verbose)
    return fit_results
    
def prepare_msdmj_fromJson(json_file):
    input = InputClass()
    try:
        JsonFile = json_file #sys.argv[1]
        with open(JsonFile) as infile:
            data = json.load(infile)
        input.fromJson(data)
        input.info()
    except IndexError:
        print('\n! Error!')
        print('!\t Json input file is missing: python3 mdmd_fit.py ___.json. ')
        print('!\t Writing example input fit_example.json')
        input.ExampleInput()
        sys.exit()
    except FileNotFoundError:
        print('\n! Error!')
        print('!\t Json input file %s not found!'%JsonFile)
        sys.exit()

    return input

def fit_msdmj(input: InputClass, json_file: str | None=None, verbose=False):
    fit_results = fit(input, json_file = json_file, verbose=verbose)
    return fit_results

def combine_fit_results(inputclasses: list[InputClass], save_path=".", temperature=300, boxlength=None, name_identifier="1", mdmd_all=True):
    #prefactor specs
    boxtype = "cubic"
    #frequency specs
    freq_type = "nue"
    unit = "THz"
    mini = 1e-05
    maxi = 50.0
    logscale = True
    #spectrum specs
    smoothing=True

    combined_input_dict = {"correlations": {}}
    epsilon = []
    theta0 = []
    if mdmd_all:
        mdmd_all_dict = {"key": "all",
                         "out": f"{save_path}/epsilon_all_{name_identifier}.dat",
                         "mdmd": [],
                         "smoothing": smoothing}
    for inputclass in inputclasses:
        gendicon_dict = inputclass.to_gendicon_input()
        #combined_input_dict["correlations"].update(gendicon_dict["correlations"])
        
        if "mdmd" in gendicon_dict["correlations"].keys():
            #check if key already exists, otherwise add it
            try:
                combined_input_dict["correlations"]["mdmd"].append(gendicon_dict["correlations"]["mdmd"][0])
            except KeyError:
                combined_input_dict["correlations"]["mdmd"] = []
                combined_input_dict["correlations"]["mdmd"].append(gendicon_dict["correlations"]["mdmd"][0])


            #add it to the spectrum output
            name = inputclass.name.lower()
            new_data = {"key": name, 
                        "out": f"{save_path}/epsilon_{name}_{name_identifier}.dat",
                        "mdmd": [name],
                        "smoothig": smoothing}
            epsilon.append(new_data)
            if mdmd_all:
                mdmd_all_dict["mdmd"].append(name)
        elif "deltamj2" in gendicon_dict["correlations"].keys():
            #check if key already exists, otherwise add it
            try:
                combined_input_dict["correlations"]["deltamj2"].append(gendicon_dict["correlations"]["deltamj2"][0])
            except KeyError:
                combined_input_dict["correlations"]["deltamj2"] = []
                combined_input_dict["correlations"]["deltamj2"].append(gendicon_dict["correlations"]["deltamj2"][0])

            #add it to the spectrum output    
            new_data = {"key": "all", 
                        "out": f"{save_path}/theta_{name_identifier}.dat",
                        "deltamj2": ["all"],
                        "smoothig": smoothing}
            theta0.append(new_data)
        #extract boxlength if msdmj is given
        #boxlength = 
        

    epsilon.append(mdmd_all_dict)

    gendicon_part = {"spectrum": {}}
    gendicon_part["spectrum"]["prefactor"] = [{"temperature": temperature, "boxlength": boxlength, "boxtype": boxtype}]
    gendicon_part["spectrum"]["frequency"] = [{"type": freq_type, "unit": unit, "Min": mini, "Max": maxi, "logscale": logscale}]

    gendicon_part["spectrum"]["epsilon"] = epsilon
    gendicon_part["spectrum"]["theta0"] = theta0
    gendicon_part["spectrum"]["gendicon"] = [{"key": "all", "out": f"{save_path}/gd_all_{name_identifier}.dat", "smoothing": True}]
    
    gendicon_total = combined_input_dict | gendicon_part #merge the dicts

    return gendicon_total

def calculate_gendicon(input_dict):
    data = input_dict
       
    #...........................................................................
    # Stage 1  
    #...........................................................................
    print('\n')
    print('-'*120)
    print('1. Correlation functions ')
    print('-'*120)

    mdmd = {}
    mdj = {}
    jj = {}
    deltamj2 = {}
    
    # Reading Json file
    #print('> 1.1 Reading Json file %s ...'%JsonFile)
    if 'mdmd' in data['correlations']:
        mdmd = CorrelationFunctions.readJson(data,'mdmd')
        
    if 'mdj' in data['correlations']:
        mdj = CorrelationFunctions.readJson(data,'mdj')
        
    if 'jj' in data['correlations']:
        jj = CorrelationFunctions.readJson(data,'jj')

    if 'deltamj2' in data['correlations']:
        deltamj2 = CorrelationFunctions.readJson(data,'deltamj2')
    # jj and Delta MJ^2 ?
    if len(jj)>0 and len(deltamj2):
        print('\n! Warning! gendicon.main')
        print('!\t <J(0)*J(t)> and < Delta MJ^2(t)> are both present!')
        print('!\t Please make sure that you are not double counting contributions.')
    
    print('\n> 1.2 Setting up correlation functions ...')
    # Reading correlation data and computing residuals
    for key in mdmd.values():
        CorrelationFunctions.setup(key)

    for key in mdj.values():
        CorrelationFunctions.setup(key)
    
    for key in jj.values():
        CorrelationFunctions.setup(key)

    for key in deltamj2.values():
        CorrelationFunctions.setup(key)
        
    #...........................................................................
    # Stage 2
    #...........................................................................
    print('\n')
    print('-'*120)
    print('2. Dielectric spectra ')
    print('-'*120)

    #print('> 2.1 Reading Json file %s ...'%JsonFile)
    prefactor = Prefactor.PrefactorClass()
    prefactor.fromJson(data['spectrum'])
    prefactor.info()

    frequency = Frequency.FrequencyClass()
    frequency.fromJson(data['spectrum'])
    frequency.info()
    # nue in THz
    frequency.frequencyMin *= frequency.frequencyUnit
    frequency.frequencyMax *= frequency.frequencyUnit
    frequency.calculateFrequencies()
    
    mdmdKeys     = list(mdmd.keys())
    mdjKeys      = list(mdj.keys())
    jjKeys       = list(jj.keys())
    deltamj2Keys = list(deltamj2.keys())

    epsilon = {}
    theta0  = {}
    gendicon = {}
    
    if 'epsilon' in data['spectrum']:
        epsilon = Spectrum.readJson(data['spectrum'],
                                    'epsilon',
                                    mdmdKeys,
                                    mdjKeys,
                                    jjKeys,
                                    deltamj2Keys)

    if 'theta0' in data['spectrum']:
        theta0 = Spectrum.readJson(data['spectrum'],
                                   'theta0',
                                   mdmdKeys,
                                   mdjKeys,
                                   jjKeys,
                                   deltamj2Keys)

    if 'gendicon' in data['spectrum']:
        gendicon = Spectrum.readJson(data['spectrum'],
                                     'gendicon',
                                     mdmdKeys,
                                     mdjKeys,
                                     jjKeys,
                                     deltamj2Keys)
    
    print('\n> 2.2 Computing Laplace transformations of the individual correlation functions ...')

    for i in mdmd.values():
        print('\n>\t',end='')
        print('-'*80)
        print('> \t <MD(0)*MD(t)> %s ...'%i.key)
        print('>\t',end='')
        print('-'*80)
        i.LaplaceTransform(frequency.frequencies)
        
    for i in mdj.values():
        print('\n>\t',end='')
        print('-'*80)
        print('> \t <MD(0)*J(t)> %s ...'%i.key)
        print('>\t',end='')
        print('-'*80)
        i.LaplaceTransform(frequency.frequencies)

    for i in jj.values():
        print('\n>\t',end='')
        print('-'*80)
        print('> \t <J(0)*J(t)> %s ...'%i.key)
        print('>\t',end='')
        print('-'*80)
        i.LaplaceTransform(frequency.frequencies)

    for i in deltamj2.values():
        print('\n>\t',end='')
        print('-'*80)
        print('> \t 0.5* d^2/dt^2 <Delta MJ^2(t)> %s ...'%i.key)
        print('>\t',end='')
        print('-'*80)
        i.LaplaceTransform(frequency.frequencies)

    print('\n> 2.3 Computing contributions to the dielectric spectrum (completely new) ...') 
    for i in epsilon.values():
        Spectrum.evaluateEpsilon(prefactor,frequency,i,mdmd,mdj)
        
    for i in theta0.values():
        Spectrum.evaluateTheta0(prefactor,frequency,i,mdj,jj,deltamj2)

    for i in gendicon.values():
        #if all is present, do not double count individual contriutions and all
        if "all" in epsilon.keys(): 
            Spectrum.evaluateGendicon(prefactor,frequency,i,{"all":epsilon["all"]},theta0)
        else:
            Spectrum.evaluateGendicon(prefactor,frequency,i,epsilon,theta0)

    #...........................................................................
    # Stage 3
    #...........................................................................
    """
    Terminate GenDiCon
    """
    print('='*120)
    print('                      T H E    E N D')
    print('='*120)
    #sys.exit()

