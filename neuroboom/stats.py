# INSERT LICENSE

# This script contains statistical functions
import navis
from collections import Counter
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import copy
import random
import navis.interfaces.neuprint as nvneu


def presynapse_focality(x,
                        heal_fragmented_neuron=True,
                        confidence_threshold=(0.9, 0.9),
                        num_synapses_threshold=1):

    """
    Finds the connections that are downstream of 'x', where the presynpases of 'x' are focalised

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

    if not isinstance(x, (navis.TreeNeuron, navis.neuronlist.NeuronList)):
        raise ValueError('Need to pass a Navis Tree Neuron type')
    elif isinstance(x, navis.neuronlist.NeuronList):
        if len(x) > 1:
            raise ValueError('Need to pass a SINGLE CatmaidNeuron')
        else:
            x = x[0]

    if heal_fragmented_neuron is True:

        x = navis.heal_fragmented_neuron(x)

    # Getting the connector table of synapses where x.id is the source
    synapse_connections = navis.interfaces.neuprint.fetch_synapse_connections(source_criteria=x.id)
    synapse_connections.astype(object)
    synapse_connections['node_id'] = object

    for i, j in tqdm(enumerate(synapse_connections[['x_pre', 'y_pre', 'z_pre']].values)):

        equals = np.equal(j, x.presynapses[['x', 'y', 'z']].values)
        all_equals = [np.all(k) for k in equals]
        node_id = x.presynapses[all_equals].node_id.tolist()
        # node_id = x.presynapses[[np.all(k) for k in np.equal(j, x.presynapses[['x', 'y', 'z']].values)]].node_id.tolist()
        synapse_connections.at[i, 'node_id'] = node_id

    truth_list = [True if len(np.unique(i)) > 1 else False for i in synapse_connections.node_id.values]
    if synapse_connections[truth_list].shape[0] == 0:

        synapse_connections.node_id = [np.unique(k)[0] for k in synapse_connections.node_id.tolist()]

    else:

        return('There are synapses associated with multiple nodes!!!!')

    
    synapse_connections = synapse_connections[synapse_connections.confidence_pre > confidence_threshold[0]][synapse_connections.confidence_post > confidence_threshold[1]].copy()
    count = Counter(synapse_connections.bodyId_post.tolist())
    count = {k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)}
    truth_list = [True if count[i] > num_synapses_threshold else False for i in synapse_connections.bodyId_post]
    synapse_connections = synapse_connections[truth_list].copy()

    df = pd.DataFrame()
    df['partner_neuron'] = synapse_connections.bodyId_post.unique().tolist()
    df['gT'] = ""
    df['significance_val'] = ""
    df['p_val'] = ""
    df['num_syn'] = [count[i] for i in df.partner_neuron]

    return(synapse_connections, df)


def postsynapse_focality(
    x,
    heal_fragmented_neuron=True,
    split_neuron=False,
    confidence_threshold=(0.9, 0.9),
    num_synapses_threshold=1):

    if not isinstance(x, (navis.TreeNeuron, navis.neuronlist.NeuronList)):
        raise ValueError('Need to pass a Navis Tree Neuron type')
    elif isinstance(x, navis.neuronlist.NeuronList):
        if len(x) > 1:
            raise ValueError('Need to pass a SINGLE CatmaidNeuron')
        else:
            x = x[0]

    if heal_fragmented_neuron is True:

        x = navis.heal_fragmented_neuron(x)

    # Getting the connector table of synapses where x.id is the source
    synapse_connections = navis.interfaces.neuprint.fetch_synapse_connections(target_criteria=x.id)
    synapse_connections.astype(object)
    synapse_connections['node_id'] = object

    for i, j in tqdm(enumerate(synapse_connections[['x_post', 'y_post', 'z_post']].values)):

        node_id = x.postsynapses[[np.all(k) for k in np.equal(j, x.postsynapses[['x', 'y', 'z']].values)]].node_id.tolist()
        synapse_connections.at[i, 'node_id'] = node_id

    the_truth = [True if len(np.unique(i)) > 1 else False for i in synapse_connections.node_id.values]
    if synapse_connections[the_truth].shape[0] == 0:
    # if synapse_connections[[True if len(np.unique(i)) > 1 else False for i in synapse_connections.node_id.values]].shape[0] == 0:
        synapse_connections.node_id = [np.unique(k)[0] for k in synapse_connections.node_id.tolist()]

    else:
        return('There are synapses associated with multiple nodes!!!!')

    synapse_connections = synapse_connections[synapse_connections.confidence_pre > confidence_threshold[0]][synapse_connections.confidence_post > confidence_threshold[1]].copy()
    count = Counter(synapse_connections.bodyId_pre.tolist())
    count = {k:v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)}
    truth_list = [True if count[i] > num_synapses_threshold else False for i in synapse_connections.bodyId_pre]
    synapse_connections = synapse_connections[truth_list].copy()

    df = pd.DataFrame()
    df['partner_neuron'] = synapse_connections.bodyId_pre.unique().tolist()
    df['gT'] = ""
    df['significance_val'] = ""
    df['p_val'] = ""
    df['num_syn'] = [count[i] for i in df.partner_neuron]

    return(synapse_connections, df)


def permut_test(x,
                measuring_node,
                synapse_connections,
                relation=['presyn', 'postsyn'],
                num_iter=10,
                df=None,
                count=None):

    geo_mat = navis.geodesic_matrix(x, tn_ids=measuring_node)
    geo_mat = geo_mat.T
    geo_mat.sort_values(by=[measuring_node], ascending=True, inplace=True)

    for k, j in tqdm(enumerate(df.partner_neuron)):

        if relation == 'presyn':

            total_distribution = geo_mat[~geo_mat.index.isin(synapse_connections[synapse_connections.bodyId_post == j].node_id.tolist())]
            specific_distribution = geo_mat[geo_mat.index.isin(synapse_connections[synapse_connections.bodyId_post == j].node_id.tolist())]

        elif relation == 'postsyn':

            total_distribution = geo_mat[~geo_mat.index.isin(synapse_connections[synapse_connections.bodyId_pre == j].node_id.tolist())]
            specific_distribution = geo_mat[geo_mat.index.isin(synapse_connections[synapse_connections.bodyId_pre == j].node_id.tolist())]

        total_mean = np.average(total_distribution.values)
        specific_mean = np.average(specific_distribution.values)

        gT = np.abs(total_mean - specific_mean)
        df.iloc[k, 1] = gT

        pV = list(total_distribution.values) + list(specific_distribution.values)
        pV = [i[0] for i in pV]

        pS = copy.copy(pV)
        pD = []

        for i in range(0, num_iter):

            random.shuffle(pS)
            pD.append(np.abs(np.average(pS[0:int(len(pS) / 2)]) - np.average(pS[int(len(pS) / 2):])))

        p_val = len(np.where(pD >= gT)[0]) / num_iter
        df.iloc[k, 2] = p_val

        ttest = stats.ttest_ind(total_distribution.values, specific_distribution.values)
        df.iloc[k, 3] = ttest[1][0]
        # if np.isnan(ttest[1]):
        #    df.iloc[k, 3] = np.nan
        # else:
        #    df.iloc[k, 3] = ttest[1][0]
    # df['num_syn'] = [count[i] for i in df.partner_neuron]
    df.sort_values(by=['p_val'], ascending=True, inplace=True)
    df.reset_index(inplace=True)
    return(df)

def permutation_test_complete(
                            x,
                            n_iter=10,
                            remove_fragments=True):

    a_pre, a_df = presynapse_focality(x, heal_fragmented_neuron=False, confidence_threshold=(0.9, 0.9))
    b_post, b_df = postsynapse_focality(x, heal_fragmented_neuron=False, confidence_threshold=(0.9, 0.9))

    presyn_pt = permut_test(x, x.root[0], synapse_connections=a_pre, relation='presyn', num_iter=n_iter, df=a_df)
    postsyn_pt = permut_test(x, x.root[0], synapse_connections=b_post, relation='postsyn', num_iter=n_iter, df=b_df)

    a_partner_neurons, a_roi = nvneu.fetch_neurons(presyn_pt.partner_neuron.tolist())
    b_partner_neurons, b_roi = nvneu.fetch_neurons(postsyn_pt.partner_neuron.tolist())

    a_type_dict = dict(zip(a_partner_neurons.bodyId.tolist(), a_partner_neurons.type.tolist()))
    b_type_dict = dict(zip(b_partner_neurons.bodyId.tolist(), b_partner_neurons.type.tolist()))

    partner_dict = {**a_type_dict, **b_type_dict}

    presyn_pt['partner_type'] = presyn_pt.partner_neuron.map(partner_dict, na_action='ignore')
    postsyn_pt['partner_type'] = postsyn_pt.partner_neuron.map(partner_dict, na_action='ignore')

    if remove_fragments:

        presyn_pt = presyn_pt[~presyn_pt.partner_type.isnull()].copy()
        postsyn_pt = postsyn_pt[~postsyn_pt.partner_type.isnull()].copy()

    return(presyn_pt, postsyn_pt)
