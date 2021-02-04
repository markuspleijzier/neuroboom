# INSERT LICENSE
import random
from typing import Union

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from sklearn.cluster import DBSCAN

import navis
import pymaid
import seaborn as sns

from neuroboom.utils import calc_cable, check_valid_neuron_input, check_valid_pymaid_input


# Contains functions to model steady state passive electrotonic properties of neurons
def prepare_neuron(
    x: Union[
        navis.TreeNeuron,
        navis.NeuronList,
        pymaid.core.CatmaidNeuron,
        pymaid.core.CatmaidNeuronList,
    ],
    change_units: bool = True,
    factor: int = 1e3,
):

    """
    Takes a navis or pymaid neuron and prepares it for electrotonic modelling

    Paramters
    ---------
    x:                a pymaid/Catmaid neuron object


    return_skdata:    bool
                    whether to return a list of node_ids topologically sorted
                    or to return a dict where the keys are treenodes and the values
                    are ranking in the topological sort

    factor:           the factor to scale the nodes from um to nm

    Returns
    --------

    x:                the neuron rescaled and ordered for electro-modelling

    Examples
    --------



    """

    if check_valid_neuron_input(x):

        node_sort = dict(
            [(i, k) for i, k in zip(range(len(x.nodes)), navis.graph_utils.node_label_sorting(x))]
        )
        node_sort_rev = {i: j for j, i in node_sort.items()}
        navis.downsample_neuron(x, downsampling_factor=float("inf"), inplace=True)
        x.nodes["node_rank"] = x.nodes.node_id.map(node_sort_rev).tolist()
        x.nodes.sort_values(by=["node_rank"], ascending=True, inplace=True)
        x.nodes.reset_index(drop=True, inplace=True)

        x = calc_cable(x, return_skdata=True)

        if not change_units:

            return x

        else:

            x.nodes["x"] = x.nodes["x"] / factor
            x.nodes["y"] = x.nodes["y"] / factor
            x.nodes["z"] = x.nodes["z"] / factor
            x.nodes["radius"] = x.nodes["radius"] / factor
            x.nodes["parent_dist"] = x.nodes["parent_dist"] / factor

            return x

    elif check_valid_pymaid_input(x):

        node_sort = dict(
            [(i, k) for i, k in zip(range(len(x.nodes)), pymaid.node_label_sorting(x))]
        )
        node_sort_rev = {i: j for j, i in node_sort.items()}
        x = pymaid.downsample_neuron(x, resampling_factor=float("inf"))
        x = pymaid.guess_radius(x)

        x.nodes["node_rank"] = x.nodes.treenode_id.map(node_sort_rev).tolist()
        x.nodes.sort_values(by=["node_rank"], ascending=True, inplace=True)
        x.nodes.reset_index(drop=True, inplace=True)

        x = pymaid.calc_cable(x, return_skdata=True)

        if not change_units:

            return x

        else:

            x.nodes["x"] = x.nodes["x"] / factor
            x.nodes["y"] = x.nodes["y"] / factor
            x.nodes["z"] = x.nodes["z"] / factor
            x.nodes["radius"] = x.nodes["radius"] / factor
            x.nodes["parent_dist"] = x.nodes["parent_dist"] / factor

            return x

    else:

        raise ValueError("Need to pass either a Navis or a Catmaid neuron type!")


def calculate_M_mat(
    x: Union[
        navis.TreeNeuron,
        navis.NeuronList,
        pymaid.core.CatmaidNeuron,
        pymaid.core.CatmaidNeuronList,
    ],
    Ra: float = 266.1,
    Rm: float = 20.8,
    Cm: float = 0.8,
    solve: bool = False,
):

    """
    Calculates the conductance matrix for a given neuron

    Paramters
    ---------
    x:                  a pymaid/Catmaid neuron object
                        This object needs to be passed through prepare_neuron()
                        before running this function


    Ra:                 float
                        Axial resistance with units in ohms / cm

    Rm:                 float
                        Membrane resistance with units in kilo-ohms / cm^2

    Cm:                 float
                        Membrane capacitance with units in uF / cm^2
                        uF = microFarads

    solve:              bool
                        Whether to solve the matrix (calculate its inverse)

    Returns
    --------

    M:
    M_solved:
    memcap:

    Examples
    --------



    """

    if check_valid_neuron_input(x):

        tn_to_index = dict(zip(x.nodes.node_id, x.nodes.index))
        nofcomps = x.nodes.shape[0]
        M = np.zeros((nofcomps, nofcomps))

        compdiam = x.nodes.radius * 2
        complength = np.zeros(nofcomps)

        # skip root node

        for i in range(1, nofcomps):

            aind = int(x.nodes.node_id[i])
            bind = int(x.nodes.parent_id[i])

            axyz = x.nodes[x.nodes.node_id == aind][["x", "y", "z"]].values
            bxyz = x.nodes[x.nodes.node_id == bind][["x", "y", "z"]].values

            complength[i] = np.sqrt(np.sum((axyz - bxyz) ** 2))

            meandiam = (
                x.nodes[x.nodes.node_id == aind].radius.values
                + x.nodes[x.nodes.node_id == bind].radius.values
            ) * 0.5

            area = (meandiam ** 2) / (4.0 * np.pi)

            M[tn_to_index[bind], tn_to_index[aind]] = (
                -area / complength[i] / Ra * 10 ** (-4)
            )
            M[tn_to_index[aind], tn_to_index[bind]] = M[
                tn_to_index[bind], tn_to_index[aind]
            ]

        complength[0] = complength[1]

        gleak = (compdiam * np.pi * complength) / (Rm * 10 ** 8)
        memcap = (compdiam * np.pi * complength) * Cm * (10 ** -6) / (10 ** 8)

        for i in range(nofcomps):
            M[i, i] = gleak[i] - np.sum(M[i])

        M = sparse.csr_matrix(M)

        if solve:

            M_solved = np.linalg.inv(sparse.csr_matrix.todense(M))

            return (M_solved, memcap)

        return (M, memcap)

    elif check_valid_pymaid_input(x):

        tn_to_index = dict(zip(x.nodes.treenode_id, x.nodes.index))
        nofcomps = x.nodes.shape[0]
        M = np.zeros((nofcomps, nofcomps))

        compdiam = x.nodes.radius * 2
        complength = np.zeros(nofcomps)

        # skip root node

        for i in range(1, nofcomps):

            aind = int(x.nodes.treenode_id[i])
            bind = int(x.nodes.parent_id[i])

            axyz = x.nodes[x.nodes.treenode_id == aind][["x", "y", "z"]].values
            bxyz = x.nodes[x.nodes.treenode_id == bind][["x", "y", "z"]].values

            complength[i] = np.sqrt(np.sum((axyz - bxyz) ** 2))

            meandiam = (
                x.nodes[x.nodes.treenode_id == aind].radius.values
                + x.nodes[x.nodes.treenode_id == bind].radius.values
            ) * 0.5

            area = (meandiam ** 2) / (4.0 * np.pi)

            M[tn_to_index[bind], tn_to_index[aind]] = (
                -area / complength[i] / Ra * 10 ** (-4)
            )
            M[tn_to_index[aind], tn_to_index[bind]] = M[
                tn_to_index[bind], tn_to_index[aind]
            ]

        complength[0] = complength[1]

        gleak = (compdiam * np.pi * complength) / (Rm * 10 ** 8)
        memcap = (compdiam * np.pi * complength) * Cm * (10 ** -6) / (10 ** 8)

        for i in range(nofcomps):
            M[i, i] = gleak[i] - np.sum(M[i])

        M = sparse.csr_matrix(M)

        if solve:

            M_solved = np.linalg.inv(sparse.csr_matrix.todense(M))
            return (M_solved, memcap)

        return (M, memcap)

    else:

        raise ValueError("Unknown object type!")


def current_injection(conducM: np.mat, curramp: float = 1e-11):

    """
    Calculates the conductance matrix for a given neuron

    Paramters
    ---------
    conducM:            numpy.matrix
                        The conductance matrix produced by

    curramp:            float
                        The amplitude of the injection current

    Returns
    --------

    Vm_mat:             numpy.matrix
                        A node-by-node matrix of relative membrane voltage (Vm)
                        given current injection

    Examples
    --------



    """

    nofcomps = conducM.shape[0]
    Vm_mat = np.zeros((nofcomps, nofcomps))

    for i in range(0, nofcomps):

        currinj = np.zeros(nofcomps)
        currinj[i] = curramp

        Vm = spsolve(conducM, currinj)

        Vm_mat[i, :] = Vm

    return Vm_mat


def dbs_func(x, eps: float = 1.0, min_samples: int = 10):

    """
    Parameters
    --------
    x:              A matrix to perform DBSCAN on

    eps:            float
                    The radius of the search circle

    min_samples:    int
                    The minimum number of samples to include in a cluster

    Returns
    --------
    labels:         cluster label of each datapoint
    n_clusters_:    number of unique clusters
    n_noise_:       number of noise points

    Examples
    --------

    """

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    return (labels, n_clusters_, n_noise_)


def find_clusters(x, phate_operator, eps: float = 1.0, min_samples: int = 10):

    """
    Parameters
    --------
    x:              A matrix to perform DBSCAN on

    phate_operator: phate.operator object
                    The operator used to perform the dimensionality reduction
                    The number of dimensions to reduce to
                    is specified in this operator object

    eps:            float
                    Radius of the search circle

    min_samples:    int
                    Minimum number of samples to include in a cluster

    Returns
    --------
    mat_red:        Matrix wih dimensions reduced
    labels:         Cluster label of each datapoint
    n_clusters_:    Number of unique clusters
    n_noise_:       Number of noise points

    Examples
    --------

    """

    # performing PHATE dimensionality reduction
    mat_red = phate_operator.fit_transform(x)

    # performing DBSCAN
    labels, n_clusters_, n_noise_ = dbs_func(mat_red, eps, min_samples)

    return (mat_red, labels, n_clusters_, n_noise_)


def cluster_palette(cmap: str = "hsv", n_objects: int = 10, shuffle: bool = False):

    """
    Parameters
    --------
    cmap:           str
                    The colour pallete to be used

    n_objects:      int
                    number of ojects to create an individual colour for.
                    E.g. number of clusters

    shuffle:        bool
                    Whether to shuffle the colourmap
                    or to keep the colours of nearby objects the similar

    Returns
    --------
    label_to_col:   a dictionary where the keys are the label and the
                    values are the colour (rgb, 0-1)

    Examples
    --------

    """

    pal = sns.color_palette(cmap, n_objects)

    if shuffle:

        random.shuffle(pal)

    label_to_col = dict(zip([i for i in range(0, n_objects)], pal))
    label_to_col[-1] = (0.0, 0.0, 0.0)

    return label_to_col
