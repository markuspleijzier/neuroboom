# INSERT LICENSE

# This script contains statistical functions
import copy
import random
from collections import Counter
from typing import Optional, Union

import navis
import navis.interfaces.neuprint as nvneu
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from neuroboom.utils import check_valid_neuron_input


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
    synapse_connections["node_id"] = object

    for i, j in tqdm(
        enumerate(synapse_connections[["x_pre", "y_pre", "z_pre"]].values)
    ):

        equals = np.equal(j, x.presynapses[["x", "y", "z"]].values)
        all_equals = [np.all(k) for k in equals]
        node_id = x.presynapses[all_equals].node_id.tolist()
        synapse_connections.at[i, "node_id"] = node_id

    truth_list = [
        True if len(np.unique(i)) > 1 else False
        for i in synapse_connections.node_id.values
    ]
    if synapse_connections[truth_list].shape[0] == 0:

        synapse_connections.node_id = [
            np.unique(k)[0] for k in synapse_connections.node_id.tolist()
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
    synapse_connections["node_id"] = object

    for i, j in tqdm(
        enumerate(synapse_connections[["x_post", "y_post", "z_post"]].values)
    ):

        node_id = x.postsynapses[
            [np.all(k) for k in np.equal(j, x.postsynapses[["x", "y", "z"]].values)]
        ].node_id.tolist()
        synapse_connections.at[i, "node_id"] = node_id

    the_truth = [
        True if len(np.unique(i)) > 1 else False
        for i in synapse_connections.node_id.values
    ]

    if synapse_connections[the_truth].shape[0] == 0:

        synapse_connections.node_id = [
            np.unique(k)[0] for k in synapse_connections.node_id.tolist()
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

    geo_mat = navis.geodesic_matrix(x, tn_ids=measuring_node)
    geo_mat = geo_mat.T
    geo_mat.sort_values(by=[measuring_node], ascending=True, inplace=True)

    for k, j in tqdm(enumerate(df.partner_neuron)):

        if relation == "presyn":

            total_distribution = geo_mat[
                ~geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_post == j
                    ].node_id.tolist()
                )
            ]
            specific_distribution = geo_mat[
                geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_post == j
                    ].node_id.tolist()
                )
            ]

        elif relation == "postsyn":

            total_distribution = geo_mat[
                ~geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_pre == j
                    ].node_id.tolist()
                )
            ]
            specific_distribution = geo_mat[
                geo_mat.index.isin(
                    synapse_connections[
                        synapse_connections.bodyId_pre == j
                    ].node_id.tolist()
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
                    np.average(pS[0 : int(len(pS) / 2)])
                    - np.average(pS[int(len(pS) / 2) :])
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
        x, heal_fragmented_neuron=False, confidence_threshold=(0.9, 0.9)
    )
    b_post, b_df = postsynapse_focality(
        x, heal_fragmented_neuron=False, confidence_threshold=(0.9, 0.9)
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
