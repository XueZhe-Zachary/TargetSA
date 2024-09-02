from itertools import product
#from rdkit import Chem
from rdkit import  DataStructs
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from utilities import sascorer
from utilities import drd2_scorer
import networkx as nx
import torch, os, time
from torch import nn
from torch.autograd import Variable
import numpy as np
from numpy import *
import pickle,gzip


def clip_grads(params, clip_value):
    if not clip_value > 0:
        return
    for param in params:
        if param.grad is not None:
            param.grad.data.clamp_(-clip_value, clip_value)


def scale_grads(params, threshold):
    if not threshold > 0:
        return
    for param in params:
        l2 = torch.norm(param, 2).data
        if (l2 > threshold).any():
            param.grad.data *= threshold / l2


def get_lr(optimiser):
    for pg in optimiser.param_groups:
        lr = pg['lr']
    return lr


def match_weights(n_layers):
    rnn_fmt = "{}_{}_l{}".format
    cells_fmt = "{}.{}_{}".format

    n = range(n_layers)
    ltype = ['ih', 'hh']
    wtype = ['bias', 'weight']
    matchings = []
    for n, l, w in product(n, ltype, wtype):
        matchings.append((rnn_fmt(w, l, n), cells_fmt(n, w, l)))

    return matchings


def sample(z_mean, z_log_var, size, epsilon_std=0.01):
    epsilon = Variable(torch.cuda.FloatTensor(*size).normal_(0, epsilon_std))
    return z_mean + torch.exp(z_log_var / 2.0) * epsilon


def make_safe(x):
    return x.clamp(1e-7, 1 - 1e-7)


def binary_entropy(x):
    return - (x * x.log() + (1 - x) * (1 - x).log())


def info_gain(x):
    marginal = binary_entropy(x.mean(0))
    conditional = binary_entropy(x).mean(0)
    return marginal - conditional


def init_params(m):
    for module_name, module in m.named_modules():
        for param_name, param in module.named_parameters():
            if 'weight' in param_name:
                if 'conv' in param_name or 'lin' in param_name or 'ih' in param_name:
                    nn.init.xavier_uniform(param)
                elif 'hh' in param_name:
                    nn.init.orthogonal(param)
            elif param_name == 'bias':
                nn.init.constant(param, 0.0)


def qfun_loss(y, p):
    log_p = torch.log(make_safe(p))
    positive = torch.sum(log_p, 1)
    neg_prod = torch.exp(positive)
    negative = torch.log1p(-make_safe(neg_prod))

    return - torch.sum(y * positive + (1 - y) * negative)


class VAELoss(torch.autograd.Function):
    def __init__(self):
        self.binary_xentropy = nn.BCELoss()

    def forward(self, x, x_decoded_mean, z_mean, z_log_var):
        eta = 0.8
        xent_loss = 40*self.binary_xentropy.forward(x_decoded_mean.view(-1), x.view(-1))
        kl_loss = - 0.5 * torch.mean(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var))
        return xent_loss# + kl_loss


def one_hot(x, alphabet_size=27):
    x_one_hot = torch.cuda.LongTensor(x.numel(), alphabet_size).zero_()
    x_one_hot.scatter_(1, x.view(-1).unsqueeze(-1), 1)
    return x_one_hot.view(x.size(0),x.size(1), alphabet_size)


def corresponding(values, idxs, dim=-1):
    idxs = Variable(idxs)
    if len(values.size()) == 4:
        if len(idxs.size()) == 2:
            idxs = idxs.unsqueeze(0)
        idxs = idxs.repeat(values.size()[0], 1, 1)
    return values.gather(dim, idxs.unsqueeze(dim)).squeeze(dim)


def preds2seqs(preds):
    seqs = [torch.cat([torch.multinomial(char_preds, 1)
                       for char_preds in seq_preds])
            for seq_preds in preds]
    return torch.stack(seqs).data


def seqs_equal(seqs1, seqs2):
    return [torch.eq(s1, s2).all() for s1, s2 in zip(seqs1, seqs2)]


def sample_prior(n_samples, dec, model=None):
    samples = Variable(torch.cuda.FloatTensor(n_samples, 56).normal_(0, 1))
    p_hat = dec.forward(samples)
    if model:
        decoded = to_numpy(model.forward_cells(p_hat))
    else:
        decoded = decode(p_hat)
    return decoded

def remap(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def penalized_logp(s):
    if s is None: return -62.52, 0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -62.52, 0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    try:
        log_p = Descriptors.MolLogP(mol)
    except:
        return -62.52,1
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
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
    return normalized_log_p + normalized_SA + normalized_cycle, 1

def readSAModel(filename='data/SA_score.pkl.gz'):
    print("mol_metrics: reading SA model ...")
    start = time.time()
    if filename == 'SA_score.pkl.gz':
        filename = os.path.join(os.path.dirname(organ.__file__), filename)
    model_data = pickle.load(gzip.open(filename))
    outDict = {}
    for i in model_data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    SA_model = outDict
    end = time.time()
    print("loaded in {}".format(end - start))
    return SA_model


# SA_model = readSAModel()

def SA_score(smile):

    mol = Chem.MolFromSmiles(smile)
    if mol is None: return  0
    # fragment score
    fp = Chem.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.
    nf = 0
    # for bitId, v in fps.items():
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += SA_model.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = mol.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(
        mol, includeUnassigned=True))
    ri = mol.GetRingInfo()
    nSpiro = Chem.CalcNumSpiroAtoms(mol)
    nBridgeheads = Chem.CalcNumBridgeheadAtoms(mol)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - \
        spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0
    val = remap(sascore, 5, 1.5)
    val = np.clip(val, 0.0, 1.0)
    return val

def lipinski(s):
    if s is None: return 0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return  0
    num_hdonors = Lipinski.NumHDonors(mol)
    num_hacceptors = Lipinski.NumHAcceptors(mol)
    mol_weight = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    if num_hdonors > 5 or num_hacceptors>10 or mol_weight>=500 or logp>5:
        return 0
    else:
        return 1

def equation_value(s, x, y=0):
    WORST = 1000
    try:
        score = np.log(1+np.mean(np.minimum((np.array(eval(s)) -y)**2, WORST)))
        valid = 1 
    except:
        score = np.log(1+WORST)
        valid = 0
    return score.mean(), valid

def program_value(s, x, y=0):

    WORST = 1000
    try:
        s = 'def fun(v0):' +s
        exec(s, globals())
        ans = fun(x)
        score = np.log(1+np.mean(np.minimum((ans -y)**2, WORST)))
        valid=1
        if np.isnan(score):
            score = np.log(1+WORST)
        return score.mean(),valid
    except:
        score = np.log(1+WORST)
        return score ,0


def qed(s):
    if s is None: return 0, 0
    mol = Chem.MolFromSmiles(s)
    if mol is None: 
        return 0, 0
    else:
        try:
            return QED.qed(mol),1
        except:
            return 0,0

def drd2(s):
    if s is None: return 0.0, 0.0
    if Chem.MolFromSmiles(s) is None:
        return 0.0, 0.0
    return drd2_scorer.get_score(s),1

def similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 





class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))



import logging,sys
logger = logging.getLogger(__name__)

class mylog(object):

    def __init__(self,dirname, level=logging.DEBUG):
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        filename = dirname+'/logging'+time.strftime('-%b%d.%Hh%M', time.localtime())+'.log'
        logname = time.strftime('-%b%d.%Hh%M', time.localtime())
        handler = logging.FileHandler(filename)        
        handler.setFormatter(formatter)
        logger = logging.getLogger(logname)
        logger.setLevel(level)
        logger.addHandler(handler)
        self.loggg = logger

    def logging(self,str):
        self.loggg.info("\n"+str)

