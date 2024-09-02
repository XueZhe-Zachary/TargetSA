import itertools
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
import copy
import networkx as nx
import math
import random
import time
# import matplotlib.pyplot as plt
import csv
from contextlib import contextmanager
import sys
import os
from sascorer import calculateScore

import subprocess
from vina import Vina
from openbabel import pybel
import meeko
from meeko import MoleculePreparation
from meeko import obutils
import tempfile
import AutoDockTools
import contextlib

def Similarity(mol, target, radius=2, nBits=2048,
               useChirality=True):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    fingerprint = 'rdkit'
    if fingerprint == 'ECFP':
        x = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useChirality=useChirality)
        target = rdMolDescriptors.GetMorganFingerprintAsBitVect(target, radius=radius, nBits=nBits, useChirality=useChirality)
        return DataStructs.TanimotoSimilarity(x, target)
    elif fingerprint == 'FCFP':
        x = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius,nBits=nBits)
        target = AllChem.GetMorganFingerprintAsBitVect(target, radius=radius,nBits=nBits)
        return DataStructs.DiceSimilarity(x, target)
    elif fingerprint == 'rdkit':
        x = Chem.RDKFingerprint(mol)
        target = Chem.RDKFingerprint(target)
        return DataStructs.FingerprintSimilarity(x, target)
    else:
        raise ValueError('fingerprint type not supported')

def Penalized_logp(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def prop_all(mol):
    """
    calculate properties of the input mol, including QED, SA, Lipinsk, logP, weight
    :param mol: rdkit mol object
    :return: float
    """
    # QED stands for quantitative estimation of drug-likeness
    M_qed = Descriptors.qed(mol)
    # SA, Synthetic accessibility [1,10], the smaller the better
    SA = calculateScore(mol)
    # M_SA = (10 - SA) / 9.0
    # Lipinski’s rule-of-five,a rule of thumb to evaluate drug-likeness
    # [0, 5] the higher the better
    M_Lipinski = obey_lipinski(mol)
    # LogP
    logp = Descriptors.MolLogP(mol)
    # weight
    M_weight = Descriptors.MolWt(mol)

    M_prop = [M_qed, SA, M_Lipinski, logp, M_weight]
    return M_prop

def obey_lipinski(mol):
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (Descriptors.MolLogP(mol)>=-2) & (Descriptors.MolLogP(mol)<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        ps = Chem.rdDistGeom.ETKDGv3()
        ps.randomSeed=0xf00d
        AllChem.EmbedMolecule(rdkit_mol, ps)        
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    """ @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None: 
            preparator.write_pdbqt_file(lig_pdbqt)
            return 
        else: 
            return preparator.write_pdbqt_string() """

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

def mol2sdf(mol):
    sdf_path = "input_sdf.sdf"
    input_mol = mol
    mol2 = AllChem.AddHs(input_mol)
    AllChem.EmbedMolecule(mol2, randomSeed=1000)
    AllChem.MMFFOptimizeMolecule(mol2)
    Chem.MolToMolFile(mol2, sdf_path)
    return sdf_path

def prepare_ligand(ligand_path):
    """
    prepare a input ligand molecule
    input is the original ligand path (sdf format)
    output is the path of prepared ligand (qdbqt format)
    :return: str
    """
    lig_pdbqt = 'lig.pdbqt'
    mol_file =  ligand_path
    a = PrepLig(mol_file, 'sdf')
    a.addH()
    a.gen_conf()

    temp_ligand_path = "conf_h.sdf"

    """ prepare_ligand = os.path.join(meeko.__path__[0], 'scripts/mk_prepare_ligand.py')
    subprocess.Popen(['python3', prepare_ligand, '-i', 'conf_h.sdf', '-o', prep_ligand_path],
                            stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate() """
    
    for mol in Chem.SDMolSupplier(temp_ligand_path, removeHs=False):
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        prep_ligand_path = "prepared_ligand.pdbqt"
        preparator.write_pdbqt_file(prep_ligand_path)

    return prep_ligand_path

def prepare_rep(rep_path):
    """
    prepare a input receptor protein
    input is the original receptor path (pdb format)
    output is the path of prepared receptor (qdbqt format)
    :return: str
    """
    prot_file = rep_path
    prot_dry = 'protein_dry.pdb'
    prot_pqr = 'protein.pqr'
    prot_pdbqt = 'prepared_rep.pdbqt'
    b = PrepProt(prot_file)
    b.del_water(prot_dry)
    b.addH(prot_pqr)
    b.get_pdbqt(prot_pdbqt)

    return prot_pdbqt

def calculate_center(input_molecule_file):
    origin_ligand = next(Chem.SDMolSupplier(input_molecule_file, removeHs=False))
    centroid =Chem.rdMolTransforms.ComputeCentroid(origin_ligand.GetConformer())
    return centroid

def DockingScore(mol, prep_rep, centroid, origin_rep, origin_ligand):
    """
    Reward for a docking score between a newly generated ligand
    and a fixed receptor (example: human GABAA receptor, protein id 6x3v in PDB)
    :param mol: rdkit mol object
    :return: float
    """
    smi = Chem.MolToSmiles(mol)  # SMILES of input molecule
    print("input SMILES for docking", smi)
    mol = Chem.MolFromSmiles(smi)
    input_rep = prep_rep
    input_centroid = centroid
    try:
        input_liand = prepare_ligand(mol2sdf(mol)) 
    except Exception as e :
        print("failed to prepare ligand", e)
        score = 88888888

    try:
        v = Vina(sf_name='vina', seed = 1000)
        v.set_receptor(input_rep)
        v.set_ligand_from_file(input_liand)
        v.compute_vina_maps(center=[input_centroid.x, input_centroid.y, input_centroid.z], box_size=[10, 10, 10])
        v.dock(exhaustiveness=8, n_poses=5)
        score = v.energies(n_poses=1)[0][0]
        v.write_poses('final_ligand.pdbqt', n_poses=1, overwrite=True)
    except Exception as e :
        print("failed to dock", e)
        score = 77777777

    if os.path.exists('conf_h.sdf'):
        os.unlink(os.getcwd() + '/conf_h.sdf')
    if os.path.exists('tmp_h.sdf'):
        os.unlink(os.getcwd() + '/tmp_h.sdf')
    if os.path.exists('input_sdf.sdf'):
        os.unlink(os.getcwd() + '/input_sdf.sdf')
    if os.path.exists('prepared_ligand.pdbqt'):
        os.unlink(os.getcwd() + '/prepared_ligand.pdbqt')
    print('#' * 80)
    print("score for docking", score)

    return score

def DockingScore_qvina_fixed(mol, prep_rep, centroid, origin_rep, origin_ligand):
    """
    calculte docking score using qvina
    """
    smi = Chem.MolToSmiles(mol)  # SMILES of input molecule
    print("input SMILES for docking", smi)
    mol = Chem.MolFromSmiles(smi)
    input_rep = prep_rep
    input_centroid = centroid
    try:
        input_liand = 'prepared_ligand.pdbqt'
        ligand_sdf = mol2sdf(mol)
        os.popen(f'obabel {ligand_sdf} -O {input_liand}').read
    except Exception as e :
        print("failed to prepare ligand", e)
        score = 88888888

    size = 10
    exhaustiveness = 8
    seed = 1000
    final_lig_path = 'final_ligand.pdbqt'
    try:
        out = os.popen(
            f'qvina2.1 --receptor {input_rep} '
            f'--ligand {input_liand} '
            f'--center_x {input_centroid.x:.4f} --center_y {input_centroid.y:.4f} --center_z {input_centroid.z:.4f} '
            f'--size_x {size} --size_y {size} --size_z {size} '
            # f'--out {final_lig_path}'
            # f'--seed {seed}'
            f'--exhaustiveness {exhaustiveness} --seed {seed} --out {final_lig_path}'
        ).read()

        if '-----+------------+----------+----------' not in out:
            score = 77777777
        else:
            out_split = out.splitlines()
            best_idx = out_split.index('-----+------------+----------+----------') + 1
            best_line = out_split[best_idx].split()
            assert best_line[0] == '1'
            score = float(best_line[1])
    except Exception as e :
        print("failed to dock", e)
        score = 77777777
    if os.path.exists('input_sdf.sdf'):
        os.unlink(os.getcwd() + '/input_sdf.sdf')
    if os.path.exists('prepared_ligand.pdbqt'):
        os.unlink(os.getcwd() + '/prepared_ligand.pdbqt')
    print('#' * 80)
    print("score for docking", score)

    return score