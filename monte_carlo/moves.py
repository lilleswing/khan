"""  classes to perform moves """
import abc
import itertools
from collections import namedtuple
import numpy as np
import inspect
import sys

RADIAN = 2.0*np.pi/360.0

# description is any data used to identify the coordinate
Coordinate = namedtuple("Coordinate", ["value", "description"])

def get_all_move_types():
    """
    Get a list of all defined move types
    this is defined as any class
    defined in this file which inherits from Move
    """
    names = []
    for name, handle in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if Move in handle.__bases__:
            names.append(name)

    return names 
    
class Move(metaclass=abc.ABCMeta):

    """ Base class for MC moves """
    def __init__(params):
        self._params = params

    @abc.abstractmethod
    def coordinates(self, st):
        """
        a list of coordinates to move
        """
        pass
        
    @property
    @abc.abstractmethod
    def max_change(self):
        """
        Max change in coordinate
        """
        pass

    @abc.abstractmethod
    def adjust_coordinate(self, st, c0, c1):
        """
        Adjust value of chosen coordinate in structure

        Parameters
        : st: Structure object
        : c0: initial coordinate instance
        : c1: final coordinate instance 
        """
        pass

    def _choose_coordinate(self, st):
        """
        Choose a coordinate to move
        """

        coords = self.coordinates(st)
        nc = len(coords)
        icoord = np.random.random_integers(0, nc, 1)[0]

        return coords[icoord]

    def _propose_adjustment(self, coord):
        """
        propose an adjusted coordinate
        """
        v0 = coord.value
        v1 = v0 + self.max_change*(2.0*np.random.random_sample() - 1.0)
        return Coordinate(v1, coord.description)

    def move(self, st):
        """
        Perform a trial move on the structure
        """
        c0 = self._choose_coordinate(st)
        c1 = self._propose_adjustment(self, c0)
        self.adjust_coordinate(st, c0, c1)
        
class AtomTranslation(Move):
    """
    Atomic translation
    """
    def coordinates(self):
        c = []
        for at in st.atom:
            c.append(Coordinate(at.x, (at.index, "x")))
            c.append(Coordinate(at.y, (at.index, "y")))
            c.append(Coordinate(at.z, (at.index, "z")))
        return c
            
    def max_change(self):
        return params.getValue("disp_max")

    def adjust_coordinate(self, st, c0, c1):
        idx, direction = c1.description
        setattr(st.atom[idx], direction, c1.value)

class BondStretch(Move):
    """
    bond stretching
    """
    def coordinates(self):
        c = []
        for b in st.bond:
            idx = (st.atom1.index, st.atom2.index)
            c.append(Coordinate(st.measure(*idx), idx))
        return c

    def max_change(self):
        return params.getValue("stretch_max")

    def adjust_coordinate(self, st, c0, c1):
        st.adjust(c1.value, *c1.description)

class AngleBend(Move):
    """
    bond stretching
    """
    def coordinates(self):
        c = []
        for ati in st.atom:
            for atj, atk in itertools.combinations(ati.bonded_atoms):
                 idx = (atj.index, ati.index, atk.index)
                 c.append(Coordinate(st.measure(*idx), idx))
        return c

    def max_change(self):
        return params.getValue("stretch_max")

    def adjust_coordinate(self, st, c0, c1):
        st.adjust(c1.value, *c1.description)

class TorsionRotation(Move):
    """ torsion rotation """
    def coordinates(self):
        c = []
        for bond in st.bond:
            i, j = (bond.atom1.index, bond.atom2.index)

            # neighbors less bonded bonded pair
            i_neighbors = [at.index for at in bond.atom1.bonded_atoms]
            j_neighbors = [at.index for at in bond.atom2.bonded_atoms]
            i_neighbors.remove(i)
            j_neighbors.remove(j)

            # choose any unique torsion
            if i_neighbors and j_neighbors:
                k = min(i_neighbors)
                l = min(j_neighbors)
                idx = (k, i, j, l)
                c.append(Coordinate(st.measure(*idx), idx))
            
        return c

    def max_change(self):
        return params.getValue("torsion_max")

    def adjust_coordinate(self, st, c0, c1):
        st.adjust(c1.value, *c1.description)

class MoleculeTranslation(Move):
    """ Rigid translation of molecule """
    def coordinates(self):
        c = []
        for i, m in st.molecule:
            c.append(Coordinate(0.0, (i, "x")))
            c.append(Coordinate(0.0, (i, "y")))
            c.append(Coordinate(0.0, (i, "z")))
            
        return c

    def max_change(self):
        return params.getValue("trans_max")

    def adjust_coordinate(self, st, c0, c1):
        imol, direction = c1.description
        dx = c1.value
        m = list(st.molecule)[imol]
        for i in st.getAtomIndices():
            x0 = getattr(st.atom[i], direction)
            setattr(st.atom[i], direction, x0 + dx)

class MoleculeRotation(Move):
    """ Rigid translation of molecule """
    def coordinates(self):
        c = []
        for i, m in st.molecule:
            c.append(Coordinate(0.0, (i, [1.0, 0.0, 0.0])))
            c.append(Coordinate(0.0, (i, [0.0, 1.0, 0.0])))
            c.append(Coordinate(0.0, (i, [0.0, 0.0, 1.0])))
            
        return c

    def max_change(self):
        return params.getValue("rot_max")

    def adjust_coordinate(self, st, c0, c1):
        imol, direction = c1.description
        dx = c1.value
        m = list(st.molecule)[imol]
        R = structutils.get_rotation_matrix(direction, dx*RADIAN)
        structutils.transform_atom_coordinates(
            m.getAtomIndices(),
            R
        )

if __name__ == '__main__':
    print("hello")
    print(get_all_move_types())
