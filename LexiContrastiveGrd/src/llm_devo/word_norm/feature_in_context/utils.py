# utils.py
import numpy as np
from itertools import zip_longest
import re
from nltk.corpus import wordnet as wn


def sigmoid(z):
    # if z is too negative, the exp overflows, so return zero.
    return 1.0 / (1 + np.exp(-z))

class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


def grouper(n, iterable, padvalue=None):
    """
    grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    """
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def cosine_similarity_matrix(A,B):
    """
    expects two numpy arrays. returns the cosine similarity matrix
    """
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)



def lemma_from_string(lemma_string):
    # grabs everything in between (' ') in a string
    # (needed to update from r"'(.*?)'" to deal with cases with quotes in word like o'clock)
    string = re.findall(r"\('(.*?)'\)", lemma_string)[0]
    #print(string)
    lemma = wn.lemma(string)
    return lemma

def lemma_name_from_string(lemma_string):
    # (needed to update from r"'(.*?)'" to deal with cases with quotes in word like o'clock)
    string = re.findall(r"\('(.*?)'\)", lemma_string)[0]
    #print(string)
    return string

def word_form_from_lemma_string(lemma_string):
    # (needed to update from r"'(.*?)'" to deal with cases with quotes in word like o'clock)
    string = re.findall(r"\('(.*?)'\)", lemma_string)[0]
    #print(string)
    return string