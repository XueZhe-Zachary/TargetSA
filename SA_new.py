# %%
import math,os,random
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles,MolToSmiles
import argparse
from rdkit.Chem import Draw,AllChem,DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from mol_generation import *
from utils import Penalized_logp,Similarity,prop_all,DockingScore_qvina_fixed
from utils import prepare_ligand, prepare_rep, calculate_center, mol2sdf
import subprocess
import os

import pickle
from optparse import OptionParser

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

# %%
seed= 1000
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# %%
from model import *
from mol_generation import GenerativeModel2, new_mol_to_graph_data

class AtomPredictionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(AtomPredictionNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            # layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(p = 0.2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.layers(x)
        return x

class MolecularPredictionNetwork(nn.Module):
    def __init__(self, atom_input_size, atom_hidden_size):
        super(MolecularPredictionNetwork, self).__init__()
        self.atom_network = AtomPredictionNetwork(atom_input_size, atom_hidden_size)
        self.gnn = GenerativeModel2(num_layer=5, emb_dim=300, gnn_type = "gin")
        # self.fc = nn.Linear(atom_hidden_size, 1)

    def forward(self, mol_list):
        molecule_scores_list = []
        X_list = []
        for i in range(len(mol_list)):
            cur_mol = mol_list[i]
            graph_feature = new_mol_to_graph_data(cur_mol)
            X = self.gnn(graph_feature)
            X_list.append(X)

        for x in X_list:
            atom_outputs = torch.zeros(len(x))
            for i in range((len(x))):
                atom_outputs[i] = self.atom_network(x[i])
            molecule_scores_list.append(atom_outputs)
        return molecule_scores_list

# %%
atom_input_size = 119
atom_hidden_size = 300
model_file = "model0105.pt"
apn = MolecularPredictionNetwork(atom_input_size, atom_hidden_size)
apn = torch.load(model_file)
apn.eval()

# %%
class simulate_anneal():
    def __init__(self,model_path,t_min,t_max,inter_circul,action_prob1,action_prob2,elements,topk,prop,threshold, orig_rep, orig_lig,prepared_rep, prepared_center, maxmize=True):
        self.model =load_generative_model(model_path)
        self.maskatom = MaskAtom(119,1)
        self.topk = topk
        self.threshold = threshold
        self.maxmize = maxmize
        self.prop = prop
        self.t_min, self.t_max, self.inter_circul, self.action_prob1, self.action_prob2 = t_min,t_max,inter_circul,action_prob1, action_prob2
        self.elements = elements
        self.prepared_rep, self.prepared_center = prepared_rep, prepared_center
        self.orig_rep, self.orig_lig = orig_rep, orig_lig

    def SA(self,mol):
        if not mol:
            print('The smiles is wrong!')
            return None
        self.initial=mol
        tem = self.t_max
        iters = 0
        idx = 0
        rep = self.prepared_rep
        center = self.prepared_center
        origin_rep, origin_ligand = self.orig_rep, self.orig_lig
#         self.write_log_file('Input: {}'.format(smile))
        all_mol,all_tem=[mol],[self.t_max]
        all_score=[self.objective_function(mol)]
        if self.prop=='logp':
            all_prop=[Penalized_logp(mol)]
        elif  self.prop=='docking':
            all_prop=[DockingScore_qvina_fixed(mol, rep, center, origin_rep, origin_ligand)]
        else:
            all_prop=[Descriptors.qed(mol)]
        while tem>=self.t_min:
            for i in range(self.inter_circul):
                atom_nums = mol.GetNumAtoms()  
                assert atom_nums>0
                atom_candidates = apn([mol])[0].argsort(descending=True)[:5].tolist()
                mol_candidates = mol_modify_candidates(mol,atom_candidates,self.maskatom,self.model,self.action_prob1,self.action_prob2,self.elements,topk=5)
                if not mol_candidates: continue
                candidates_scores = []
                for new_mol in mol_candidates:
                    candidates_scores.append(self.objective_function(new_mol))
                new_mol = mol_candidates[np.argmax(candidates_scores)]
                score_new = np.max(candidates_scores)                
                score_old = self.objective_function(mol)
                score_change = score_new-score_old                
                
                if score_change>0:
                    mol = new_mol
                    all_score.append(score_new)
                else:
                    p = math.exp(score_change/tem)
                    if p>np.random.uniform(low=0,high=1):
                        mol = new_mol
                        all_score.append(score_new)
                    else:
                        print("one more step")
                        new_new_mol, new_score_new = self.twostep(new_mol)
                        twosteps_score_change = new_score_new - score_old
                        if 0 < twosteps_score_change < 10\
                            and MolFromSmiles(MolToSmiles(new_new_mol))!=MolFromSmiles(MolToSmiles(mol)) \
                            and MolFromSmiles(MolToSmiles(new_new_mol))!=MolFromSmiles(MolToSmiles(new_mol)):
                            # 四个list先加入new_mol相关的信息
                            all_mol.append(new_mol)
                            all_score.append(score_new)
                            all_tem.append(tem)
                            if self.prop=='logp':
                                all_prop.append(Penalized_logp(new_mol))
                            elif self.prop == 'docking':
                                all_prop.append(DockingScore_qvina_fixed(new_mol, rep, center, origin_rep, origin_ligand))
                            else:
                                all_prop.append(Descriptors.qed(new_mol))
                            all_tem.append(tem)      
                                             
                            # iters+=1
                            # tem = self.t_max/(iters+1)
                            idx+=1
                            print("succeed in one more step")
                            print("smi for two-steps", MolToSmiles(mol), MolToSmiles(new_mol), MolToSmiles(new_new_mol))
                            print("score for two-steps", score_old, score_new, new_score_new)
                            print("*"*20)
                            mol = new_new_mol
                            all_score.append(new_score_new)
                        else:
                            print("failed in one more step")
                            print("smi for two-steps", MolToSmiles(mol), MolToSmiles(new_mol), MolToSmiles(new_new_mol))
                            print("score for two-steps", score_old, score_new, new_score_new)
                            all_score.append(score_old)

                all_mol.append(mol)
                
                if self.prop=='logp':
                    all_prop.append(Penalized_logp(mol))
                elif self.prop == 'docking':
                    all_prop.append(DockingScore_qvina_fixed(mol, rep, center, origin_rep, origin_ligand))
                else:
                    all_prop.append(Descriptors.qed(mol))
                all_tem.append(tem)

            iters+=1
            tem = self.t_max/(iters+1)
            idx+=1

        return all_mol[np.argmax(all_score)],all_mol,all_tem,all_prop

    def objective_function(self,mol):
        if self.prop=='logp':         
            proper = Penalized_logp(mol)
        elif self.prop=='qed':            
            proper = Descriptors.qed(mol)
        else:
            proper = DockingScore_qvina_fixed(mol, self.prepared_rep, self.prepared_center, self.orig_rep, self.orig_lig)

        similarity = Similarity(self.initial,mol)
        qed_score = Descriptors.qed(mol)
        SA_score = prop_all(mol)

        if similarity<self.threshold:
            return -10000000
    
        if self.prop=='logp':
            value = proper + 5*similarity
        elif self.prop == 'docking':
            # value = -5*proper
            value = -3*proper + 2*qed_score - 0.3*SA_score[1]
        else:
            value = 1000*proper + similarity
        return value
    
    def twostep(self, mol):
        candidates_all = []
        for i in range(self.inter_circul):
            atom_nums = mol.GetNumAtoms()  
            assert atom_nums>0
            # pos = idx % atom_nums
            pos = range(atom_nums)
            mol_candidates = mol_modify_candidates(mol,pos,self.maskatom,self.model,self.action_prob1,self.action_prob2,self.elements,topk=5)
            if not mol_candidates: # mol_candidates 为空
                continue    
            for candidate in mol_candidates:
                candidates_all.append(candidate)          
        
        if not candidates_all: # candidates_all 为空
            return MolFromSmiles('C'), -9999
        candidates_scores = []
        for new_new_mol in candidates_all:
            candidates_scores.append(self.objective_function(new_new_mol))
        twostep_new_mol = candidates_all[np.argmax(candidates_scores)]
        twostep_score_new = np.max(candidates_scores)  
        return twostep_new_mol, twostep_score_new

# %%
# 获得112个frag的smiles
ligand_list = []
file_path = "frag112.txt"
with open(file_path, 'r') as file:
    for line in file:
        smiles = line.strip()
        ligand_list.append(smiles)

# %%
parser = OptionParser()
parser.add_option("-p", "--pocket_path", dest="pocket_path", help="Specify the pocket path")

(options, args) = parser.parse_args()
if not options.pocket_path:
    parser.error("You must specify the pocket path using -p or --pocket_path option.")
workdir = options.pocket_path    
print("workdir:", workdir)

os.chdir(workdir)
for file_name in os.listdir(workdir):
    file_path = os.path.join(workdir, file_name)

    if file_name.endswith(".sdf"):
        ligand_sdf = file_path
    elif file_name.endswith(".pdb"):
        protein_pdb = file_path
    elif file_name == "prepared_rep.pdbqt":
        protein_pdbqt = file_path

origin_ligand = ligand_sdf
origin_receptor = protein_pdb

if os.path.exists('prepared_rep.pdbqt'):
    prepared_rep = protein_pdbqt
else:
    prepared_rep = prepare_rep(protein_pdb)
    os.unlink(os.getcwd() + '/protein.log')
    os.unlink(os.getcwd() + '/protein.pqr')
    os.unlink(os.getcwd() + '/protein_dry.pdb')
centroid = calculate_center(ligand_sdf)

print("current protein and ligand:", prepared_rep, ligand_sdf)
sa = simulate_anneal('model/model_ep100.pth',t_min=0.1,t_max=1,inter_circul=5,action_prob1=[0.1,0.3,0.5, 0.1],action_prob2=[0.6,0.25,0.05, 0.1],elements=['C', 'N', 'O', 'P', 'S'], topk=5,prop='docking',threshold=0.1, orig_rep= origin_receptor, orig_lig= origin_ligand, prepared_rep = prepared_rep, prepared_center= centroid, maxmize=True)

first_mol_all = []
final_mol_all = []
first_mol_prop = []
final_mol_prop = []
final_mol_with_coordinate = []
failed_list = []
failed_mol = MolFromSmiles('C')

for smi in ligand_list:
    try:
        input_mol = Chem.MolFromSmiles(smi)
        mol,all_mol,all_tem,all_prop = sa.SA(input_mol)
        first_mol_all.append(all_mol[0])
        final_mol_all.append(all_mol[-1])
        first_mol_prop.append(all_prop[0])
        final_mol_prop.append(all_prop[-1]) 
        out_pdbqt_file = 'final_ligand.pdbqt'
        out_pdb_file = 'final_ligand.pdb'
        out_sdf_file = 'final_ligand.sdf'
        os.popen(f'obabel {out_pdbqt_file} -O {out_pdb_file} -b -d').read()
        rdmol = Chem.MolFromPDBFile(str(out_pdb_file))
        print("succeed in ligand:", smi)
        final_mol_with_coordinate.append(rdmol)
    except:
        print("failed in ligand:", smi)
        failed_list.append(smi)
        final_mol_with_coordinate.append(failed_mol)
        continue
    if os.path.exists('final_ligand.pdbqt'):
        os.unlink(os.getcwd() + '/final_ligand.pdbqt')
    if os.path.exists('final_ligand.sdf'):
        os.unlink(os.getcwd() + '/final_ligand.sdf')
    if os.path.exists('final_ligand.pdb'):
        os.unlink(os.getcwd() + '/final_ligand.pdb')
# %%
# 将first_mol_all, final_mol_all, first_mol_prop, final_mol_prop 保存到pkl文件中
pocket_data = {
    'pocket_name': workdir, 
    'first_mol_all': first_mol_all,
    'final_mol_all': final_mol_all,
    'first_mol_prop': first_mol_prop,
    'final_mol_prop': final_mol_prop,
    'final_mol_with_coordinate': final_mol_with_coordinate,
    'failed_list': failed_list
}

with open('results.pkl', 'ab') as file:
    pickle.dump(pocket_data, file)
# %%
