# INSERT LICENSE

# This script contains statistical functions
import copy
import random
from collections import Counter
from typing import Optional, Union, Tuple

import navis
import navis.interfaces.neuprint as nvneu
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp
from tqdm import tqdm
import itertools
import seaborn as sns
from matplotlib.lines import Line2D

from neuroboom.utils import check_valid_neuron_input
from neuroboom import morphoelectro as nbm


def presynapse_focality(
    x: Union[navis.TreeNeuron, navis.NeuronList],
    heal_fragmented_neuron: bool = False,
    confidence_threshold: tuple((float, float)) = (0.9, 0.9),
    num_synapses_threshold: int = 1,
):

    """
    Finds the connections that are downstream of 'x', where the presynpases of 'x' are focalised

    Parameters
    --------
    x:                       A matrix to perform DBSCAN on

    heal_fragmented_neuron:  bool
                             Whether to heal the neuron or not.
                             N.B. Its better to heal neurons during
                             import to save time in this function.

    connector_confidence:    tuple of floats
                             The confidence value used to threshold the synapses.
                             The first value (connector_confidence[0]) will be used to threshold presynapses
                             The second value (connector_confidence[1]) will be used to threshold postsynapses

    num_samples_threshold:   int
                             The minimum number of synapses a partner must have
                             to be included in the permutation test

    Returns
    --------
    synapse_connections:     A dataframe detailing the presynaptic connections
    df:                      A dataframe to be populated by the permutation test function

    Examples
    --------

    """

    x = check_valid_neuron_input(x)

    if heal_fragmented_neuron is True:

        x = navis.heal_fragmented_neuron(x)

    # Getting the connector table of synapses where x.id is the source
    synapse_connections = navis.interfaces.neuprint.fetch_synapse_connections(
        source_criteria=x.id
    )
    synapse_connections.astype(object)
    synapse_connections = nbm.match_connectors_to_nodes(synapse_connections,
                                                        x,
                                                        synapse_type = 'pre')

    truth_list = [
        True if len(np.unique(i)) > 1 else False
        for i in synapse_connections.node.values
    ]
    if synapse_connections[truth_list].shape[0] == 0:

        synapse_connections.node = [
            np.unique(k)[0] for k in synapse_connections.node.tolist()
        ]

    else:

        return "There are synapses associated with multiple nodes!!!!"

    synapse_connections = synapse_connections[
        synapse_connections.confidence_pre > confidence_threshold[0]
    ][synapse_connections.confidence_post > confidence_threshold[1]].copy()
    count = Counter(synapse_connections.bodyId_post.tolist())
    count = {
        k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)
    }
    truth_list = [
        True if count[i] > num_synapses_threshold else False
        for i in synapse_connections.bodyId_post
    ]
    synapse_connections = synapse_connections[truth_list].copy()

    df = pd.DataFrame()
    df["partner_neuron"] = synapse_connections.bodyId_post.unique().tolist()
    df["gT"] = ""
    df["significance_val"] = ""
    df["p_val"] = ""
    df["num_syn"] = [count[i] for i in df.partner_neuron]

    return (synapse_connections, df)


def postsynapse_focality(
    x: Union[navis.TreeNeuron, navis.NeuronList],
    heal_fragmented_neuron: bool = False,
    split_neuron: bool = False,
    confidence_threshold: tuple((float, float)) = (0.9, 0.9),
    num_synapses_threshold: int = 1,
):

    """
    Finds the connections that are downstream of 'x', where the presynpases of 'x' are focalised

    Parameters
    --------
    x:                       A matrix to perform DBSCAN on

    heal_fragmented_neuron:  bool
                             Whether to heal the neuron or not.
                             N.B. Its better to heal neurons during
                             import to save time in this function.

    connector_confidence:    tuple of floats
                             The confidence value used to threshold the synapses.
                             The first value (connector_confidence[0]) will be used to threshold presynapses
                             The second value (connector_confidence[1]) will be used to threshold postsynapses

    num_samples_threshold:   int
                             The minimum number of synapses a partner must have
                             to be included in the permutation test

    Returns
    --------
    synapse_connections:     A dataframe detailing the presynaptic connections
    df:                      A dataframe to be populated by the permutation test function

    Examples
    --------

    """

    x = check_valid_neuron_input(x)

    if heal_fragmented_neuron is True:

        x = navis.heal_fragmented_neuron(x)

    # Getting the connector table of synapses where x.id is the source
    synapse_connections = navis.interfaces.neuprint.fetch_synapse_connections(
        target_criteria=x.id
    )
    synapse_connections.astype(object)
    synapse_connections = nbm.match_connectors_to_nodes(synapse_connections,
                                                        x,
                                                        synapse_type = 'post')

    the_truth = [
        True if len(np.unique(i)) > 1 else False
        for i in synapse_connections.node.values
    ]

    if synapse_connections[the_truth].shape[0] == 0:

        synapse_connections.node = [
            np.unique(k)[0] for k in synapse_connections.node.tolist()
        ]

    else:
        return "There are synapses associated with multiple nodes!!!!"

    synapse_connections = synapse_connections[
        synapse_connections.confidence_pre > confidence_threshold[0]
    ][synapse_connections.confidence_post > confidence_threshold[1]].copy()
    count = Counter(synapse_connections.bodyId_pre.tolist())
    count = {
        k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)
    }
    truth_list = [
        True if count[i] > num_synapses_threshold else False
        for i in synapse_connections.bodyId_pre
    ]
    synapse_connections = synapse_connections[truth_list].copy()

    df = pd.DataFrame()
    df["partner_neuron"] = synapse_connections.bodyId_pre.unique().tolist()
    df["gT"] = ""
    df["significance_val"] = ""
    df["p_val"] = ""
    df["num_syn"] = [count[i] for i in df.partner_neuron]

    return (synapse_connections, df)


def permut_test(
    x: Union[navis.TreeNeuron, navis.NeuronList],
    measuring_node: int,
    synapse_connections: pd.DataFrame,
    relation: str = "presyn",
    num_iter: int = 10,
    df: Optional = None,
    count: Optional = None,
):

    """
    Runs a permutation test on the geodesic distances for connections

    Parameters
    --------
    x:                       navis.TreeNeuron

    measuring_node:          int
                             Node ID for which to measure the geodesic distance of synapses

    synapse_connections:     pandas.DataFrame
                             A DataFrame containin the synaptic connections
                             upon which the permutation test will be executed.

    relation:                str
                             Whether the synaptic connections included in
                             synapse_connections are presynapses or postsynapses

    num_iter:                int
                             Number of iterations to run the permutation test for

    df:                      pandas.DataFrame
                             A DataFrame to record the p_values of the permutation test

    count:                   dict
                             The total number of synaptic connections the
                             partner neuron has onto x


    Returns
    --------
    df:                      A dataframe populated by the permutation test function
                             containing the p_values of the partners, determined by
                             the locations of their connections.


    Examples
    --------

    """

    x = check_valid_neuron_input(x)

    geo_mat = navis.geodesic_matrix(x, node_ids=measuring_node)
    geo_mat = geo_mat.T
    geo_mat.sort_values(by=[measuring_node], ascending=True, inplace=True)

    for k, j in tqdm(enumerate(df.partner_neuron)):

        if relation == "presyn":

            total_distribution = geo_mat[
                ~geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_post == j
                    ].node.tolist()
                )
            ]
            specific_distribution = geo_mat[
                geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_post == j
                    ].node.tolist()
                )
            ]

        elif relation == "postsyn":

            total_distribution = geo_mat[
                ~geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_pre == j
                    ].node.tolist()
                )
            ]
            specific_distribution = geo_mat[
                geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_pre == j
                    ].node.tolist()
                )
            ]

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
            pD.append(
                np.abs(
                    np.average(pS[0: int(len(pS) / 2)])
                    - np.average(pS[int(len(pS) / 2):])
                )
            )

        p_val = len(np.where(pD >= gT)[0]) / num_iter
        df.iloc[k, 2] = p_val

        ttest = stats.ttest_ind(total_distribution.values, specific_distribution.values)
        df.iloc[k, 3] = ttest[1][0]

    df.sort_values(by=["p_val"], ascending=True, inplace=True)
    df.reset_index(inplace=True)
    return df


def permutation_test_complete(
    x: Union[navis.TreeNeuron, navis.NeuronList],
    n_iter: int = 10,
    remove_fragments: bool = True,
    confidence_threshold: tuple = (0.9, 0.9)
):

    """
    A wrapper function for the presynaptic and postsynaptic permutation test
    functions, so that both can be performed with minimal code writing
    by the user

    Parameters
    --------
    x:                       navis.TreeNeuron

    n_iter:                  int
                             Number of iterations to run the permutation test for

    remove_fragments:        bool
                             Whether to remove partners that are fragments/
                             have not been traced to completion/
                             do not have a soma.


    Returns
    --------
    presyn_pt:               A dataframe populated by the permutation test function
                             containing the p_values of the presynaptic partners,
                             determined by the locations of their connections.

    postsyn_pt:              A dataframe populated by the permutation test function
                             containing the p_values of the postsynaptic partners,
                             determined by the locations of their connections.


    Examples
    --------

    """

    a_pre, a_df = presynapse_focality(
        x, heal_fragmented_neuron=False, confidence_threshold=confidence_threshold
    )
    b_post, b_df = postsynapse_focality(
        x, heal_fragmented_neuron=False, confidence_threshold=confidence_threshold
    )

    presyn_pt = permut_test(
        x,
        x.root[0],
        synapse_connections=a_pre,
        relation="presyn",
        num_iter=n_iter,
        df=a_df,
    )
    postsyn_pt = permut_test(
        x,
        x.root[0],
        synapse_connections=b_post,
        relation="postsyn",
        num_iter=n_iter,
        df=b_df,
    )

    a_partner_neurons, a_roi = nvneu.fetch_neurons(presyn_pt.partner_neuron.tolist())
    b_partner_neurons, b_roi = nvneu.fetch_neurons(postsyn_pt.partner_neuron.tolist())

    a_type_dict = dict(
        zip(a_partner_neurons.bodyId.tolist(), a_partner_neurons.type.tolist())
    )
    b_type_dict = dict(
        zip(b_partner_neurons.bodyId.tolist(), b_partner_neurons.type.tolist())
    )

    partner_dict = {**a_type_dict, **b_type_dict}

    presyn_pt["partner_type"] = presyn_pt.partner_neuron.map(
        partner_dict, na_action="ignore"
    )
    postsyn_pt["partner_type"] = postsyn_pt.partner_neuron.map(
        partner_dict, na_action="ignore"
    )

    if remove_fragments:

        presyn_pt = presyn_pt[~presyn_pt.partner_type.isnull()].copy()
        postsyn_pt = postsyn_pt[~postsyn_pt.partner_type.isnull()].copy()

    return (presyn_pt, postsyn_pt)


def prefocality_to_dendrogram_coloring(
    x: pd.DataFrame,
    p_val: float,
    neuron: navis.TreeNeuron
):

    """
    Function to take the results of synaptic focality tests and create colour dict for highlighting connectors
    """

    x_thresh = x[x.p_val < p_val].copy()
    partner_dict = dict(zip(x_thresh.partner_neuron, x_thresh.partner_type))

    # fetching synapse connections
    conn = nvneu.fetch_synapse_connections(source_criteria=neuron.id, target_criteria=x_thresh.partner_neuron.tolist())

    # filtering for highly probably synapses
    conn_thresh = conn[(conn.confidence_pre > 0.9) & (conn.confidence_post > 0.9)].copy()

    pal = sns.color_palette('turbo', len(partner_dict))
    pal_dict = dict(zip(partner_dict.keys(), pal))

    nodes_matched = nbm.match_connectors_to_nodes(conn_thresh, neuron, synapse_type='pre')

    c2n = dict(zip(nodes_matched.connector, nodes_matched.bodyId_post))
    c2color = {i: pal_dict[c2n[i]] for i in c2n.keys()}


    return(c2color, c2n, conn_thresh, partner_dict)


def postfocality_to_dendrogram_coloring(
    x: pd.DataFrame,
    p_val: float,
    neuron: navis.TreeNeuron
):

    """
    Function to take the results of synaptic focality tests and create colour dict for plotting
    """

    x_thresh = x[x.p_val < p_val].copy()
    partner_dict = dict(zip(x_thresh.partner_neuron, x_thresh.partner_type))

    # fetching synapse connections
    conn = nvneu.fetch_synapse_connections(target_criteria=neuron.id,
                                           source_criteria=x_thresh.partner_neuron.tolist())

    # filtering for highly probably synapses
    conn_thresh = conn[(conn.confidence_pre > 0.9) & (conn.confidence_post > 0.9)].copy()

    pal = sns.color_palette('turbo', len(partner_dict))
    pal_dict = dict(zip(partner_dict.keys(), pal))

    nodes_matched = nbm.match_connectors_to_nodes(conn_thresh, neuron, synapse_type = 'post')

    c2n = dict(zip(nodes_matched.connector, nodes_matched.bodyId_pre))
    c2color = {i : pal_dict[c2n[i]] for i in c2n.keys()}


    return(c2color, c2n, conn_thresh, partner_dict)



def make_legend_elements(connector_to_color, connector_to_neuron, partner_dict):

    neuron_to_color = {connector_to_neuron[i]: connector_to_color[i] for i in connector_to_color.keys()}

    legend_elements = []

    for i in range(len(neuron_to_color)):

        neuron = list(neuron_to_color.keys())[i]

        legend_elements.append(Line2D([i], [0], marker='o',
                                        color=neuron_to_color[neuron],
                                        label=f'{neuron} : {partner_dict[neuron]}',
                                        markerfacecolor=neuron_to_color[neuron],
                                        markersize=60))

    return(legend_elements)

#####################
## All By All synaptic Focality
#####################

# KS test

def synaptic_focality_KS_test(
    x: navis.TreeNeuron,
    synapse_type: str = 'pre',
    confidence_threshold: Tuple = (0.9, 0.9)
):

    if synapse_type == 'pre':

        g_mat = navis.geodesic_matrix(x)

        syn = nvneu.fetch_synapse_connections(source_criteria=x.id)
        syn = syn[(syn.confidence_pre > confidence_threshold[0]) & (syn.confidence_post > confidence_threshold[1])].copy()
        syn = nbm.match_connectors_to_nodes(syn, x, synapse_type=synapse_type)

        df = pd.DataFrame()
        df['partner_id'] = syn.bodyId_post.unique()
        partner_gt = {}
        partner_statistic = {}
        partner_pval = {}

        for i, j in enumerate(df.partner_id):

            nodes = syn[syn.bodyId_post == j].node.tolist()

            truth_array = np.isin(g_mat.index, nodes)

            partner_geo_dist_vals = g_mat[truth_array].values.mean(axis=1)

            total_geo_dist_vals = g_mat[~truth_array].values.mean(axis=1)

            partner_gt[j] = partner_geo_dist_vals

            KS_test = ks_2samp(partner_geo_dist_vals, total_geo_dist_vals)

            partner_statistic[j] = KS_test.statistic

            partner_pval[j] = KS_test.pvalue

        df['gT'] = df.partner_id.map(partner_gt)
        df['KS statistic'] = df.partner_id.map(partner_statistic)
        df['KS pval'] = df.partner_id.map(partner_pval)
        df['n_syn'] = [len(i) for i in df.gT]


    elif synapse_type == 'post':

        g_mat = navis.geodesic_matrix(x)

        syn = nvneu.fetch_synapse_connections(target_criteria=x.id)
        syn = syn[(syn.confidence_pre > confidence_threshold[0]) & (syn.confidence_post > confidence_threshold[1])].copy()
        syn = nbm.match_connectors_to_nodes(syn, x, synapse_type=synapse_type)

        df = pd.DataFrame()
        df['partner_id'] = syn.bodyId_pre.unique()
        partner_gt = {}
        partner_statistic = {}
        partner_pval = {}

        for i, j in enumerate(df.partner_id):

            nodes = syn[syn.bodyId_pre == j].node.tolist()

            truth_array = np.isin(g_mat.index, nodes)

            partner_geo_dist_vals = g_mat[truth_array].values.mean(axis=1)

            total_geo_dist_vals = g_mat[~truth_array].values.mean(axis=1)

            partner_gt[j] = partner_geo_dist_vals

            KS_test = ks_2samp(partner_geo_dist_vals, total_geo_dist_vals)

            partner_statistic[j] = KS_test.statistic

            partner_pval[j] = KS_test.pvalue


        df['gT'] = df.partner_id.map(partner_gt)
        df['KS statistic'] = df.partner_id.map(partner_statistic)
        df['KS pval'] = df.partner_id.map(partner_pval)
        df['n_syn'] = [len(i) for i in df.gT]


    return(df)

# permutation (enrichment analysis)


def calculate_T_obs(
    neuron_id: int,
    neuron_to_node_dict: dict,
    gmat: pd.DataFrame,
    two_sample: bool = True):

    Anodes_to_query = neuron_to_node_dict[neuron_id]
    An = len(Anodes_to_query)

    Bnodes_to_query = gmat.index[~np.isin(gmat.index, Anodes_to_query)].to_numpy()
    Bn = len(Bnodes_to_query)

    A_mean = gmat.loc[Anodes_to_query, :].mean().mean()
    B_mean = gmat.loc[Bnodes_to_query, :].mean().mean()

    T_obs = A_mean - B_mean

    if two_sample:

        T_obs = abs(T_obs)

    return(T_obs, An, Bn)


def random_draw_sample_dist(
    n_iter: int,
    gmat: pd.DataFrame,
    T_obs: float,
    An: int,
    Bn: int
):

    A_draws = []
    B_draws = []

    for i in range(n_iter):

        rc_A = np.random.choice(gmat.index.to_numpy(), size=An, replace=False)
        rc_B = np.random.choice(gmat.index.to_numpy(), size=Bn, replace=False)

        sample_mean_A = gmat.loc[rc_A, :].mean().mean()
        sample_mean_B = gmat.loc[rc_B, :].mean().mean()

        A_draws.append(sample_mean_A)
        B_draws.append(sample_mean_B)

    cart_prod = [i for i in itertools.product(A_draws, B_draws)]

    sampled_permut_diff = abs(np.array([i - j for i, j in cart_prod]))

    p_value = sum(sampled_permut_diff >= T_obs) / len(sampled_permut_diff)

    return(p_value)


def aba_presyn_focality(
    neuron: navis.TreeNeuron,
    confidence_threshold: Tuple = (0.0, 0.0),
    n_iter: int = 100
):

    syn = nvneu.fetch_synapse_connections(source_criteria=neuron.id)
    syn = syn[(syn.confidence_pre > confidence_threshold[0]) & (syn.confidence_post > confidence_threshold[1])].copy()

    # syn_wmc = synaptic connections with connectors matched
    syn_wmc = nbm.match_connectors_to_nodes(syn, neuron, synapse_type='pre')

    connector2node = dict(zip(neuron.connectors.connector_id, neuron.connectors.node_id))

    syn_wmc['node'] = syn_wmc.connector.map(connector2node).to_numpy()

    unique_usns = syn_wmc.bodyId_post.unique()

    neuron_to_uNodes = {i: syn_wmc[syn_wmc.bodyId_post == i].node.unique() for i in unique_usns}

    g_mat = navis.geodesic_matrix(neuron)

    df = pd.DataFrame()
    df['unique_ids'] = unique_usns
    T_obs_list = []
    An_list = []
    Bn_list = []
    rdsd_list = []

    for i in unique_usns:

        T_obs, An, Bn = calculate_T_obs(neuron_id=i,
                                        neuron_to_node_dict=neuron_to_uNodes,
                                        gmat=g_mat)

        rdsd = random_draw_sample_dist(n_iter, g_mat, T_obs, An, Bn)

        T_obs_list.append(T_obs)
        An_list.append(An)
        Bn_list.append(Bn)
        rdsd_list.append(rdsd)

    df['T_obs'] = T_obs_list
    df['An'] = An_list
    df['Bn'] = Bn_list
    df['rdsd'] = rdsd_list

    return(df)


def aba_postsyn_focality(
    neuron: navis.TreeNeuron,
    confidence_threshold: Tuple = (0.0, 0.0),
    n_iter: int = 100
):

    syn = nvneu.fetch_synapse_connections(target_criteria=neuron.id)
    syn = syn[(syn.confidence_pre > confidence_threshold[0]) & (syn.confidence_post > confidence_threshold[1])].copy()

    # syn_wmc = synaptic connections with connectors matched
    syn_wmc = nbm.match_connectors_to_nodes(syn, neuron, synapse_type='post')

    connector2node = dict(zip(neuron.connectors.connector_id, neuron.connectors.node_id))

    syn_wmc['node'] = syn_wmc.connector.map(connector2node).to_numpy()

    unique_usns = syn_wmc.bodyId_pre.unique()

    neuron_to_uNodes = {i: syn_wmc[syn_wmc.bodyId_pre == i].node.unique() for i in unique_usns}

    g_mat = navis.geodesic_matrix(neuron)

    df = pd.DataFrame()
    df['unique_ids'] = unique_usns
    T_obs_list = []
    An_list = []
    Bn_list = []
    rdsd_list = []

    for i in unique_usns:

        T_obs, An, Bn = calculate_T_obs(neuron_id=i,
                                        neuron_to_node_dict=neuron_to_uNodes,
                                        gmat=g_mat)

        rdsd = random_draw_sample_dist(n_iter, g_mat, T_obs, An, Bn)

        T_obs_list.append(T_obs)
        An_list.append(An)
        Bn_list.append(Bn)
        rdsd_list.append(rdsd)

    df['T_obs'] = T_obs_list
    df['An'] = An_list
    df['Bn'] = Bn_list
    df['rdsd'] = rdsd_list

    return(df)
