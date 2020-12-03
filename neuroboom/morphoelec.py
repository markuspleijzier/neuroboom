# INSERT LICENSE

import scipy.sparse as sparse
import numpy as np

# Contains functions to model steady state passive electrotonic properties of neurons


def calculate_M_mat(x, Ra, Rm, Cm):
    """
    Calculates the voltage matrix

    Parameters
    ----------
    x : A navis neuron object

    returned_object : graph, graph_and_positions, positions

    prog : The layout type, can be dot, neato or fdp

    Returns
    -------
    graph : skeleton nodes as graph nodes with links between them as edges
    positions : a dictionary of the positions on the 2D plane of each node

    Examples
    --------
    """
    tn_to_index = dict(zip(x.nodes.treenode_id, x.nodes.index))
    nofcomps = x.nodes.shape[0]
    M = np.zeros((nofcomps, nofcomps))
    compdiam = x.nodes.radius * 2
    complength = np.zeros(nofcomps)

    # Skip root node
    for i in range(1, nofcomps):

        aind = int(x.nodes.treenode_id[i])
        bind = int(x.nodes.parent_id[i])

        axyz = x.nodes[x.nodes.treenode_id == aind][['x', 'y', 'z']].values
        bxyz = x.nodes[x.nodes.treenode_id == bind][['x', 'y', 'z']].values

        complength[i] = np.sqrt(np.sum((axyz - bxyz)**2))
        a_ind_nodes = x.nodes[x.nodes.treenode_id == aind].radius.values
        b_ind_nodes = x.nodes[x.nodes.treenode_id == bind].radius.values
        meandiam = (a_ind_nodes + b_ind_nodes) * .5
        # meandiam = (x.nodes[x.nodes.treenode_id == aind].radius.values + x.nodes[x.nodes.treenode_id == bind].radius.values) * .5
        area = (meandiam**2) / (4.0 * np.pi)

        M[tn_to_index[bind], tn_to_index[aind]] = -area / complength[i] / Ra*10**(-4)
        M[tn_to_index[aind], tn_to_index[bind]] = M[tn_to_index[bind], tn_to_index[aind]]

    complength[0] = complength[1]

    gleak = (compdiam * np.pi * complength) / (Rm * 10 ** 8)
    memcap = (compdiam * np.pi * complength) * Cm*(10 ** -6) / (10**8)

    for i in range(nofcomps):
        M[i, i] = gleak[i] - np.sum(M[i])

    M = sparse.csr_matrix(M)

    return(M, memcap)
