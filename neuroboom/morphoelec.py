# INSERT LICENSE
import random
from typing import Optional, Union, Any

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import scipy.spatial.distance as ssd
from sklearn.cluster import DBSCAN

import navis
import pymaid
import seaborn as sns

import itertools
from collections import Counter


from neuroboom import dendrogram as nbd
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


def match_connectors_to_nodes(synapse_connections, neuron, synapse_type = 'post'):

    if synapse_type == 'post':

        u = synapse_connections[['x_post','y_post','z_post']].values
        v = neuron.postsynapses[['x','y','z']].values

        data = ssd.cdist(u, v, metric = 'euclidean')

        dist_mat = pd.DataFrame(index = [i for i in range(0, synapse_connections.shape[0])],
                                columns = [i for i in range(0, neuron.postsynapses.shape[0])],
                                data = data)



        ind = [np.argmin(dist_mat.iloc[i, :]) for i in range(0, synapse_connections.shape[0])]

        syn_con = synapse_connections.copy()
        syn_con['connector'] = [neuron.postsynapses.iloc[i, :].connector_id for i in ind]

        # creating a connector to node dictionary

        c2n = dict(zip(neuron.postsynapses.connector_id.tolist(), neuron.postsynapses.node_id.tolist()))
        syn_con['node'] = syn_con.connector.map(c2n)

    elif synapse_type == 'pre':

        u = synapse_connections[['x_pre','y_pre','z_pre']].values
        v = neuron.presynapses[['x','y','z']].values

        data = ssd.cdist(u, v, metric = 'euclidean')

        dist_mat = pd.DataFrame(index = [i for i in range(0, synapse_connections.shape[0])],
                                columns = [i for i in range(0, neuron.presynapses.shape[0])],
                                data = data)



        ind = [np.argmin(dist_mat.iloc[i, :]) for i in range(0, synapse_connections.shape[0])]

        syn_con = synapse_connections.copy()
        syn_con['connector'] = [neuron.presynapses.iloc[i, :].connector_id for i in ind]

        # creating connector to node dictionary

        c2n = dict(zip(neuron.presynapses.connector_id.tolist(), neuron.presynapses.node_id.tolist()))
        syn_con['node'] = syn_con.connector.map(c2n)


    return(syn_con)

def permute_start_end(test, test_in_roi, cluster):

    start_end = [i for i in itertools.permutations(test.nodes[test_in_roi][test.nodes[test_in_roi].node_cluster == cluster].node_id.tolist(), 2)]

    return(start_end)

def cluster_to_all_nodes(neuron, start_end_node_pairs):

    g = nbd.create_graph_structure(neuron, returned_object = 'graph')
    g_rev = g.reverse()

    nodes_of_cluster = []

    for i, j in start_end_node_pairs:

        paths = nx.all_simple_paths(g_rev, source = i, target = j)

        for path in paths:

            nodes_of_cluster.append(path)

    nodes_of_cluster = list(np.unique(list(chain.from_iterable(nodes_of_cluster))))
    return(nodes_of_cluster)


def quick_optimisation(x, min_samples_start):

    """
    x : a neuronlist
    min_samples_start: the minimum number of samples to start off with

    """
    min_samples_optim_values = []
    err = []

    for i in x:

        print(i.id, i.name)

        n = i.copy()

        try:

            n_prep = prepare_neuron(n, change_units = True, factor = 1e3)

        except IndexError:

            err.append(i.id)

            continue

        n_m, n_memcap = calculate_M_mat(n_prep, solve = False)
        n_m_solved = np.linalg.inv(sparse.csr_matrix.todense(n_m))

        phate_operator = phate.PHATE(n_components = 3, n_jobs = -2, verbose = False)

        min_sample_range = [i for i in range(0, min_samples_start + 1)][::-1]

        for j in min_sample_range:

            mat_red, labels, n_clust, n_noise = nbm.find_clusters(n_m_solved, phate_operator, eps=1e-02, min_samples = j)

            if n_noise == 0:

                min_samples_optim_values.append((i.id, j))

                break

            else:

                continue

    return(min_samples_optim_values, err)

def find_compartments_in_roi(neuron, roi, min_samples):

    n = neuron.copy()
    n_prep = prepare_neuron(n, change_units=True, factor = 1e3)
    n_m, n_memcap = calculate_M_mat(n_prep, solve = False)
    n_m_solved = np.linalg.inv(sparse.csr_matrix.todense(n_m))

    phate_operator = phate.PHATE(n_components=3, n_jobs=-2, verbose = False)

    mat_red, labels, n_clusters, n_noise = nbm.find_clusters(n_m_solved, phate_operator, eps = 1e-02, min_samples = min_samples)

    node_to_label = dict(zip(range(0, len(labels)), labels))

    n_prep.nodes['node_cluster'] = n_prep.nodes.index.map(node_to_label)

    n_in_roi = navis.in_volume(n_prep.nodes[['x','y','z']] * 1e3, roi, mode = 'IN', inplace = False)

    return(n_prep, n_in_roi)

def matching_inputs_to_compartments(neuron_id, roi):

    # Fetch the healed skeleton
    full_skeleton = nvneu.fetch_skeletons(neuron_id, heal = True)[0]
    # which of the whole neuron's nodes are in the roi
    skeleton_in_roi = navis.in_volume(full_skeleton.nodes[['x','y','z']], roi, mode='IN')

    # Fetch the neurons synapses
    postsyn = nvneu.fetch_synapse_connections(target_criteria = neuron_id)
    # match the connectors to nodes
    syn_to_node = match_connectors_to_nodes(postsyn, full_skeleton, synapse_type = 'post')
    # which of these are in the roi
    roi_syn_con = syn_to_node[syn_to_node.node.isin(full_skeleton.nodes[skeleton_in_roi].node_id.tolist())].copy()

    # Count how many synapses the upstream neurons have on your neuron of interest
    n_syn = dict(Counter(roi_syn_con.bodyId_pre.tolist()).most_common())

    # fetch these neurons
    a, b = nvneu.fetch_neurons(roi_syn_con.bodyId_pre.unique())
    a['n_syn'] = [n_syn[i] for i in a.bodyId.tolist()]

    # adding the instance names to the synapses in the roi
    bid_to_instance = dict(zip(a.bodyId.tolist(), a.instance.tolist()))
    roi_syn_con['instance'] = [bid_to_instance[i] for i in roi_syn_con.bodyId_pre]

    # compartmentalise the neuron and find the nodes in each compartment for the prepared neuron
    compartments_in_roi, nodes_in_roi = find_compartments_in_roi(full_skeleton, ca, 6)
    clusters = compartments_in_roi.nodes.node_cluster.unique()

    cluster_dict = {}

    # find the nodes that make up each compartment in the full neuron
    for i in clusters:

        clust_nodes = cluster_to_all_nodes(full_skeleton, start_end_node_pairs=permute_start_end(compartments_in_roi, nodes_in_roi, cluster = i))
        cluster_dict[i] = clust_nodes

    #cluster_dict = {k : s for s, k in cluster_dict.items()}
    #roi_syn_con['compartment'] = [cluster_dict[i] for i in roi_syn_con.node.tolist()]

    return(cluster_dict)
    #return(roi_syn_con)

    #return(full_skeleton, skeleton_in_roi, skeleton_prep, roi_syn_con, a)


def find_compartments_of_missing_nodes(roi_syn_con, nodes_with_cluster, full_neuron, ds_neuron):

    """
    DS neuron -- downsampled_neuron
    """

    # find the connections associated to nodes that do not have a cluster
    nodes_to_query = roi_syn_con[~roi_syn_con.node.isin(nodes_with_cluster)].node.tolist()

    # create a geodesic matrix between the nodes that do not have a compartment and the original, non-downsampled neuron
    gmat = navis.geodesic_matrix(full_neuron, nodes_to_query)

    # subset the above geodesic matrix columns to only contain the nodes in the downsampled neuron
    reduced_gmat = gmat.T[np.isin(gmat.T.index, ds_neuron.nodes.node_id.tolist())].T.copy()

    # find the node which is the smallest distance away from the unclassified node
    closest_classified_node = list(reduced_gmat.columns[np.argmin(reduced_gmat.values, axis = 1)])

    closest_classified_node_to_cluster = ds_neuron.nodes[ds_neuron.nodes.node_id.isin(np.unique(closest_classified_node))]
    closest_classified_node_to_cluster = dict(zip(closest_classified_node_to_cluster.node_id, closest_classified_node_to_cluster.node_cluster))

    query_nodes_to_closest_node = dict(zip(np.unique(nodes_to_query), np.array(reduced_gmat.columns[np.argmin(reduced_gmat.values, axis = 1)])))

    comps = [closest_classified_node_to_cluster[query_nodes_to_closest_node[i]] for i in query_nodes_to_closest_node.keys()]

    query_nodes_to_compartment = dict(zip(query_nodes_to_closest_node.keys(), comps))

    return(query_nodes_to_compartment)


def node_to_compartment_full_neuron(original_neuron, ds_neuron):

    """
    DS neuron - downsampled neuron (one that has been run through find_compartments_in_roi function)
    """

    # nodes to query
    ntq = np.isin(original_neuron.nodes.node_id.tolist(), ds_neuron.nodes.node_id.tolist())
    ntq = original_neuron.nodes[~ntq].node_id.tolist()

    # geodesic matrix
    gmat = navis.geodesic_matrix(original_neuron, ntq)

    #subset the columns to the nodes that are in the downsampled neuron
    red_gmat = gmat.T[np.isin(gmat.T.index, ds_neuron.nodes.node_id)].T.copy()

    ntc = dict(zip(red_gmat.index.tolist(),
                  list(chain.from_iterable([ds_neuron.nodes[ds_neuron.nodes.node_id == i].node_cluster.tolist() for i in red_gmat.columns[np.argmin(red_gmat.values, axis = 1)]]))))

    ntc = {**ntc, **dict(zip(ds_neuron.nodes.node_id.tolist(), ds_neuron.nodes.node_cluster.tolist()))}

    original_neuron.nodes['node_cluster'] = [ntc[i] for i in original_neuron.nodes.node_id]

    return(original_neuron)

def compartmentalise_neuron(neuron_id, roi):

    # Fetching the neuron
    ds_neuron = nvneu.fetch_skeletons(neuron_id, heal = True)[0]
    original_neuron = nvneu.fetch_skeletons(neuron_id, heal = True)[0]

    # Electrotonic model
    DS_NEURON = prepare_neuron(ds_neuron, change_units = True, factor = 1e3)
    test_m, test_memcap = calculate_M_mat(DS_NEURON, solve = False)
    test_m_solved = np.linalg.inv(sparse.csr_matrix.todense(test_m))

    # running PHATE
    phate_operator = phate.PHATE(n_components = 3, n_jobs = -2, verbose = False)

    mat_red, labels, n_clusters, n_noise = nbm.find_clusters(test_m_solved,
                                                              phate_operator,
                                                              eps = 1e-02,
                                                              min_samples = 6)

    # index and labels
    index_to_label = dict(zip(range(0, len(labels)), labels))
    ds_neuron.nodes['node_cluster'] = ds_neuron.nodes.index.map(index_to_label)

    node_to_compartment = dict(zip(ds_neuron.nodes.node_id.tolist(),
                                   ds_neuron.nodes.node_cluster.tolist()))

    unique_compartments = ds_neuron.nodes.node_cluster.unique()

    # Finding the cluster of the nodes that were removed when downsampling the neuron
    whole_neuron_node_to_cluster = []

    for i in unique_compartments:

        start_end = [i for i in itertools.permutations(ds_neuron.nodes[ds_neuron.nodes.node_cluster == i].node_id.tolist(), 2)]

        nodes_of_cluster = cluster_to_all_nodes(original_neuron, start_end)

        node_to_cluster_dictionary = dict(zip(nodes_of_cluster, [i] * len(nodes_of_cluster)))

        whole_neuron_node_to_cluster.append(node_to_cluster_dictionary)

    whole_neuron_node_to_cluster_dict = {k : v for d in whole_neuron_node_to_cluster for k, v in d.items()}

    #Fetching postsynapses
    ds_neuron_postsynapses = nvneu.fetch_synapse_connections(target_criteria=neuron_id)

    ds_neuron_synapse_to_node = match_connectors_to_nodes(ds_neuron_postsynapses,
                                                          original_neuron,
                                                          synapse_type = 'post')

    #Which nodes are in the CA?
    skeleton_in_roi = navis.in_volume(original_neuron.nodes[['x','y','z']].values, roi, inplace = False)
    roi_syn_con = ds_neuron_synapse_to_node[ds_neuron_synapse_to_node.node.isin(original_neuron.nodes[skeleton_in_roi].node_id.tolist())].copy()

    a, b = nvneu.fetch_neurons(roi_syn_con.bodyId_pre.unique())
    bid_to_instance = dict(zip(a.bodyId.tolist(), a.instance.tolist()))
    roi_syn_con['instance'] = [bid_to_instance[i] for i in roi_syn_con.bodyId_pre]

    nodes_to_query = roi_syn_con[~roi_syn_con.node.isin(whole_neuron_node_to_cluster_dict.keys())].node.tolist()
    comp_of_missing_nodes = find_compartments_of_missing_nodes(roi_syn_con,
                                                               list(whole_neuron_node_to_cluster_dict.keys()),
                                                               original_neuron,
                                                               ds_neuron)

    whole_neuron_node_to_cluster_dict = {**whole_neuron_node_to_cluster_dict, **comp_of_missing_nodes}

    roi_syn_con['compartment'] = [whole_neuron_node_to_cluster_dict[i] for i in roi_syn_con.node.tolist()]

    original_neuron = node_to_compartment_full_neuron(original_neuron, ds_neuron)

    return(original_neuron, ds_neuron, roi_syn_con)
