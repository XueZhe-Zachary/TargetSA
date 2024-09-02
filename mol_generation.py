import math
import copy
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles, CanonSmiles, AllChem,Draw
import torch
from torch_geometric.data import Data
from model import GenerativeModel,GenerativeModel2
import datetime
import os


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

# TODO: process data

def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                    'possible_bond_dirs'].index(
                    bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def new_mol_to_graph_data(mol, feq_list):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                    'possible_bond_dirs'].index(
                    bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MaskAtom:
    def __init__(self, num_atom_type, mask_size):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.mask_size = mask_size

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            masked_atom_indices = random.sample(
                range(num_atoms), self.mask_size)

        # create mask node label by copying atom feature of mask

        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])
        return data

    def __repr__(self):
        return '{}(num_atom_type={}, mask_size={})'.format(
            self.__class__.__name__, self.num_atom_type, self.mask_size)

# TODO: load model

def load_generative_model(model_file):
    model = GenerativeModel(num_layer=5, emb_dim=300)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

def load_generative_model2(model_file):
    model = GenerativeModel2(num_layer=5, emb_dim=300)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    return model

def recommend_atom(model,data,topk=5):
    out = model(data)
#     print("out", out)
    value,atom_id = torch.topk(out,topk)
#     print("value", value)
#     print("atom_id", atom_id)
    atom_id = list(atom_id.squeeze().numpy())
  
    atoms = [Chem.Atom(int(i)) for i in atom_id]
#     print("recommend_atoms")
#     for atom in atoms:
#         print(atom.GetSymbol())
    return atoms

def mol_modify_candidates(mol,op_pos,maskatom,model,action_prob1, action_prob2, elements, topk=5):
    atom_idx = random.choice(op_pos)  # 从可操作位置中随机选择一个位置
    print("atom_idx", atom_idx)  # 这次选择的原子
    cur_atom = mol.GetAtomWithIdx(atom_idx)
    print("cur_atom", cur_atom.GetSymbol())  # 这次选择的原子的元素符号
    data = mol_to_graph_data(mol)
    data = maskatom(data,masked_atom_indices=[atom_idx])
    recom_atoms_0 = recommend_atom(model,data,topk=topk)
    recom_atoms = []
    edit_type_list = []
    for atom in recom_atoms_0:
        if atom.GetSymbol() not in elements:
            continue
        else:
            recom_atoms.append(atom)
    
    candidates = []
    idx_list = []
    ssr = Chem.GetSymmSSSR(Chem.RWMol(mol).GetMol())
    num_ring = len(ssr)
    print("This time num of ring", num_ring) 

    # 获取原子数量
    num_atoms = mol.GetNumAtoms()
    atoms_in_ring = set()
    for ring in ssr:
        for atom_index in ring:
            atoms_in_ring.add(atom_index)
    # 找到环外的原子
    atoms_outside_ring = set(range(num_atoms)) - atoms_in_ring

    # 可成环的flag，true代表允许成环
    cycle_flag = False
    if len(atoms_outside_ring) >=4:
        cycle_flag = True

    action_prob= action_prob1.copy() 
        
    double_flag = True
    Chem.Kekulize(mol)
    cur_atom = mol.GetAtomWithIdx(atom_idx)
    for neighbor in cur_atom.GetNeighbors():
        j = neighbor.GetIdx()  ##原子索引
        #print(neighbor.GetSymbol()) ##具体对应的原子
        bond = mol.GetBondBetweenAtoms(atom_idx, j)  ##原子之间键
        #print(bond.GetBondType())  ##键类型
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            double_flag = False
            #break
    Chem.SanitizeMol(mol)

    action = choose_action(action_prob)
    if action==0: #replace
        print("Action: replace")
        self_element = cur_atom.GetSymbol()
        recom_atoms_repla = []
        for atom in recom_atoms:
            if atom.GetSymbol() in self_element:
                continue
            else:
                recom_atoms_repla.append(atom)

        for new_atom in recom_atoms_repla:
            mw = Chem.RWMol(mol)
            mw.ReplaceAtom(atom_idx,new_atom)
            try:
                Chem.SanitizeMol(mw)
                candidates.append(mw.GetMol())
                idx_list.append(atom_idx)
                edit_type_list.append(0)
                print("successfully replaced:", MolToSmiles(mw))
            except:
                continue

    elif action==1: # add
        print("Action: original add")
        print("double_flag", double_flag)

        for new_atom in recom_atoms:
            if double_flag:
                bonds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE] 
            else:
                bonds = [Chem.BondType.SINGLE] # 限制只加单键
            #bonds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]

            atom_num = mol.GetNumAtoms()
            new_atom_idx = atom_num
            for bond in bonds:
                mw = Chem.RWMol(mol)
                mw.AddAtom(new_atom)
                mw.AddBond(atom_idx,new_atom_idx,bond)
                try:
                    Chem.SanitizeMol(mw)
                    candidates.append( mw.GetMol() )
                    idx_list.append(atom_idx)
                    edit_type_list.append(1)
                    # print("successfully added:", MolToSmiles(mw))
                    if double_flag:
                        print("add mol can double bond:", MolToSmiles(mw.GetMol()))
                    else:
                        print("add mol only single bond:", MolToSmiles(mw.GetMol()))
                except:
                    print("failed in original add")
                    continue

    elif action==2: # cyclization
        # atom_idx link to random op_pos 
        # constraint: the length of cycle is 6
        link_pos = list(set(op_pos) - set([atom_idx]))  # 所有可操作的位置，除去自己
        success_state = 0  # 枚举完所有可操作位置，是否成功成六元环

        if cycle_flag:
            for pos in link_pos:
                mw = Chem.RWMol(mol)
                ring_origin = Chem.GetSymmSSSR(mw.GetMol())
                num_ring_origin = len(ring_origin)
                for ring in ring_origin:
                    print("ring before cyclization",list(ring))
                print("mol before cyclization", MolToSmiles(mw.GetMol()))
                try:
                    # 添加键可能失败，因为可能本来就存在边了
                    mw.AddBond(atom_idx, pos, Chem.BondType.SINGLE)
                    new_bond_atom = pos
                except:
                    continue

                # 检查是几元环
                ssr = Chem.GetSymmSSSR(mw.GetMol())
                num_ring = len(ssr)
                print("num of ring", num_ring)  # add之后 一共几个环
                if num_ring - num_ring_origin == 0:
                    # 如果没有新增环，说明没有成环成功
                    continue
                
                for ring in ssr:
                    ring_atom = list(ring)
                    # 避免形成内环
                    duplicate_flag = True
                    for origin_list in ring_origin:
                        count = 0
                        for atom in ring_atom:
                            if atom in origin_list:
                                count += 1
                        if count >= 3:
                            duplicate_flag = False
                            break
                        
                    if not duplicate_flag: # 如果duplicate_flag为false
                        print("duplicate_flag is False", list(ring))
                        continue

                    if atom_idx in ring_atom and new_bond_atom in ring_atom \
                        and len(ring_atom) == 6:  # 是6元环:
                        # 是新形成的环
                        try:
                            Chem.SanitizeMol(mw)
                            candidates.append( mw.GetMol() )
                            idx_list.append(atom_idx)
                            success_state = 1
   
                            edit_type_list.append(2)
                            print("ring consisted of atoms id:", list(ring))    
                            print("new_bond_atom", new_bond_atom)                  
                            print("Action: cyclization", MolToSmiles(mw))
                            break
                        except:
                            continue
                    else:  # 不是6元环
                        print("failed to cyclization: not a hexatomic ring", list(ring))
                        continue
                
                if success_state == 1:  # 不用再枚举位置成环，否则继续，看下一个可连接的位置
                    break
            
            if success_state == 0: # 枚举完位置还是0，没能成功成环，这一次操作就是添加操作
                print("Action: add in cyclization")
                bonds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
                for new_atom in recom_atoms:
                    atom_num = mol.GetNumAtoms()
                    new_atom_idx = atom_num
                    for bond in bonds:
                        mw = Chem.RWMol(mol)
                        mw.AddAtom(new_atom)
                        mw.AddBond(atom_idx,new_atom_idx,bond)
                        try:
                            Chem.SanitizeMol(mw)
                            candidates.append( mw.GetMol() )
                            idx_list.append(atom_idx)
                            print("add in cyclization mol :", MolToSmiles(mw.GetMol()))
                            edit_type_list.append(1)
                        except:
                            continue    
        else:
            print("Action: add when forbidden to cyclization")
            bonds = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
            for new_atom in recom_atoms:
                atom_num = mol.GetNumAtoms()
                new_atom_idx = atom_num
                for bond in bonds:
                    mw = Chem.RWMol(mol)
                    mw.AddAtom(new_atom)
                    mw.AddBond(atom_idx,new_atom_idx,bond)
                    try:
                        Chem.SanitizeMol(mw)
                        candidates.append( mw.GetMol() )
                        idx_list.append(atom_idx)
                        edit_type_list.append(1)
                        print("add when forbidden to cyclization:", MolToSmiles(mw.GetMol()))
                    except:
                        continue    
                
    else: # remove
        print("Action: remove")
        mw = Chem.RWMol(mol)
        mw.RemoveAtom(atom_idx)       
        smis = MolToSmiles(mw).split('.')
        for smi in smis:
            if smi:
                m = MolFromSmiles(smi)
                if m:
                    try:
                        Chem.SanitizeMol(m)
                        candidates.append(m)
                        idx_list.append(atom_idx)
                        edit_type_list.append(3)
                    except:
                        continue
    return candidates, atom_idx, edit_type_list

def choose_action(c):
    r = np.random.random()
    c = np.array(c)
    for i in range(1, len(c)):
        c[i] = c[i]+c[i-1]
    for i in range(len(c)):
        if c[i] >= r:
            return i

