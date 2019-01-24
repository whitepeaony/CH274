from rdkit import Chem
from glob import glob
from rdkit.Chem import rdMolTransforms
import numpy as np
import pickle

def sp2_indices(mol):
    sp2_index = []
    atoms = mol.GetAtoms()
    bonded = []
    for bond in mol.GetAtoms()[0].GetBonds():
        bonded.append(bond.GetEndAtomIdx())
    for atom in atoms:
        neighbors = atom.GetNeighbors()
        if len(neighbors) == 3:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                angle = rdMolTransforms.GetAngleDeg(mol.GetConformer(),bonded[a],0,bonded[b])
                if (angle > 109 and angle < 145):
                    sp2_index.append(atom.GetIdx())
    return list(set(sp2_index))

def connectivity_matrix(mol):
    atoms = mol.GetAtoms()
    carbons = [atom for atom in atoms if atom.GetAtomicNum() == 6 ]
    carbons_idx = [int(carbon.GetIdx()) for carbon in carbons]
    conmat = np.eye(len(carbons))
    for idx in carbons_idx:
        neighbors = atoms[idx].GetNeighbors()
        for nei in neighbors:
            nei_idx = int(nei.GetIdx())
            if nei_idx in carbons_idx:
                conmat[nei_idx,idx] = -1
            else:
                pass
    return conmat-np.eye(len(carbons))

def num_carbons(mol):
    atoms = mol.GetAtoms()
    return len([atom for atom in atoms if atom.GetAtomicNum() == 6 ])


def is_fully_aromatic(mol):
    if len(sp2_indices(mol)) == num_carbons(mol):
        return True
    else:
        return False
    
def choose_n(myid, choices, n_molecules=10):
    import random
    random.seed(myid)
    subset = random.sample(list(choices), n_molecules)
    return subset


def generate_dataset(myid=None, n_molecules=10, n_carbon_atoms=24, verbose=False):
    """
    This function picks 10 random molecules from a database based on your student ID.
    """
    
    allsp2_dict = pickle.load( open( "PAH_database/pahs.pkl", "rb" ) )
    maxidx = len(allsp2_dict)
    if verbose:
        print(' | --- Total number of fully aromatic molecules in database: ' + str(maxidx))
        print(' | --- By number of carbon atoms: ')
    # number of fully conjugated molecules by number of carbon atoms
        for num in range(2,40,2):
            print('      len = ' + str(num) + ' : molecules = ' + str(len([mol for mol in allsp2_dict if mol['num_carbon'] == num])))

    if verbose:
        print(' | --- Your molecule dataset contains molecules with {0} sp2 hybridised carbon atoms'.format(n_carbon_atoms))
    equal_carbon_number = [elm for elm in allsp2_dict if elm['num_carbon'] == n_carbon_atoms]

    if verbose:
        print(' | --- Based on your student ID we chose the following molecules for you: ')

    subset = choose_n(myid, equal_carbon_number, n_molecules)
    mols = []
    legend = []
    for pmol in subset:
        if verbose:
            print('      ' + pmol['name'])
        legend.append(pmol['name'])
        mols.append(Chem.MolFromMolFile('PAH_database/'+pmol['name']+'.mol'))

    return subset, mols, legend


def select_molecule(name):
    """
    This function selects a molecule from a database of PAH molecules.
    """
    
    allsp2_dict = pickle.load( open( "PAH_database/pahs.pkl", "rb" ) )
    maxidx = len(allsp2_dict)
    
    molecule = None
    mol_object = None
    for elm in allsp2_dict:
        if elm['name']==name:
            molecule = elm
            mol_object=Chem.MolFromMolFile('PAH_database/'+name+'.mol')

    if molecule is None:
        print('No molecule with this name found')
    return molecule, mol_object

def matprint(mat, fmt="g"):
    """
    This function takes a numpy matrix and prints it in a nice format.
    """
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def draw_MO(mol, eigenvecs, n=0):
    """
    draw molecular orbital onto 2D molecule structure using the rdkit_mol object and the eigenvectors
    """
    from rdkit.Chem.Draw import rdMolDraw2D
    from IPython.display import SVG, display

    highlight=range(len(eigenvecs))
    eigenvecs = np.array(eigenvecs)
    if len(eigenvecs.shape)==1:
        radii_list = np.abs(eigenvecs)
        color_list = np.sign(eigenvecs)
    else:
        radii_list=np.abs(eigenvecs[n])
        color_list = np.sign(eigenvecs[n])
    colors = {}
    radii = {}
    for i,c in enumerate(color_list):
        radii[i] = radii_list[i]
        if c<0:
            colors[i] = (0,0,1)
        elif c>=0:
            colors[i] = (1,0,0)
        else:
            raise ValueError('something is wrong with the signs.')

    drawer = rdMolDraw2D.MolDraw2DSVG(400,200)
    drawer.DrawMolecule(mol,
                    highlightAtoms=highlight,
                    highlightAtomColors=colors,
                    highlightAtomRadii=dict(radii),
                    highlightBonds=[],
                       )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    display(SVG(svg))