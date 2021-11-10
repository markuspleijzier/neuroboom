# INSERT LICENSE
from collections import Counter
from typing import List, Optional

import navis
import navis.interfaces.neuprint as nvneu
import pandas as pd
from neuroboom import morphoelectro as nbm

# Create an adjacency matrix from synapse connections (neuprint)


def adjx_from_syn_conn(
    x: List[int],
    presyn_postsyn: str = "pre",
    roi: Optional = None,
    ct: tuple = (0.0, 0.0),
    rename_index: bool = False,
):

    """
    Creates an adjacency matrix from synapse connections

    Parameters
    ----------
    presyn_postsyn:
    x :                 list
                        a list of bodyIds of which you want to find their synaptic (pre/post) connections

    presyn_postsyn :    str
                        a string of either 'pre' or 'post'
                        If 'pre', the function will search for the
                        presynaptic connections / downstream neurons of x

                        If 'post', the function will search for the
                        postsynaptic connections / upstream neurons of x

    roi :               navis.Volume
                        A region of interest within which you are filtering the connections for


    ct :                tuple
                        Confidence threshold tuple containing the confidence value to filter above
                        The first value is presynaptic confidence and the second value postsynaptic confidence
                        e.g. (0.9, 0.8) will filter for connections where the presynaptic confidence > 0.9
                        and the postsynaptic confidence > 0.8

    rename_index :      bool
                        Whether to rename the index using the type of the connected neuron


    Returns
    -------
    df :                a DataFrame where x are the columns, the connection type (pre/post) are the rows
                        and the values the number of connections

    partner_type_dict : a dictionary where the keys are bodyIds of the
                        upstream/downstream neurons and the values are their types

    Examples
    --------
    """

    if presyn_postsyn == "pre":

        con = nvneu.fetch_synapse_connections(source_criteria=x)

        if roi:

            tt = navis.in_volume(con[["x_pre", "y_pre", "z_pre"]].values, roi)
            con = con[tt].copy()

        if ct[0] or ct[1] > 0.0:

            con = con[(con.confidence_pre > ct[0]) & (con.confidence_post > ct[1])].copy()

        neurons = con.bodyId_post.unique()
        n, _ = nvneu.fetch_neurons(neurons)
        partner_type_dict = dict(zip(n.bodyId.tolist(), n.type.tolist()))

        count = Counter(con.bodyId_post)
        count = count.most_common()
        count_dict = dict(count)

        df = pd.DataFrame(columns=[x], index=[i for i, j in count], data=[count_dict[i] for i, j in count])

    elif presyn_postsyn == "post":

        con = nvneu.fetch_synapse_connections(target_criteria=x)

        if roi:

            tt = navis.in_volume(con[["x_post", "y_post", "z_post"]].values, roi)
            con = con[tt].copy()

        if ct[0] or ct[1] > 0.0:

            con = con[(con.confidence_pre > ct[0]) & (con.confidence_post > ct[1])].copy()

        neurons = con.bodyId_pre.unique()
        n, _ = nvneu.fetch_neurons(neurons)
        partner_type_dict = dict(zip(n.bodyId.tolist(), n.type.tolist()))

        count = Counter(con.bodyId_pre)
        count = count.most_common()
        count_dict = dict(count)

        df = pd.DataFrame(index=[i for i, j in count], columns=[x], data=[count_dict[i] for i, j in count])
        df = df.T.copy()

    # df = pd.DataFrame(
    #     columns=[x], index=[i for i, j in count], data=[count_dict[i] for i, j in count])

    if rename_index:

        df.index = [partner_type_dict[i] for i in df.index]

    return (df, partner_type_dict)

#### Labelling connection types (e.g. Axo-dendritic etc.)

def node_to_compartment_type(
    x: navis.TreeNeuron,
    split_neuron: navis.NeuronList
):

    for i, j in enumerate(x.nodes.node_id):

        if np.isin(j, split_neuron[0].nodes.node_id):

            compartment = split_neuron[0].compartment

            x.nodes.loc[i, 'compartment'] = compartment

        elif np.isin(j, split_neuron[1].nodes.node_id):

            compartment = split_neuron[1].compartment

            x.nodes.loc[i, 'compartment'] = compartment

        elif np.isin(j, split_neuron[2].nodes.node_id):

            compartment = split_neuron[2].compartment

            x.nodes.loc[i, 'compartment'] = compartment

        elif np.isin(j, split_neuron[3].nodes.node_id):

            compartment = split_neuron[3].compartment

            x.nodes.loc[i, 'compartment'] = compartment

        else:

            x.nodes.loc[i, 'compartment'] = 'missing'

    if any(x.nodes.compartment.isnull()):

        assert "There are nodes which are not assigned to a compartment!!"

    else:

        return(x)



def find_connection_types(
    n: navis.TreeNeuron,
    split: navis.NeuronList,
    syn_con: pd.DataFrame,
    synapse_type: str = 'pre',
    metric: str = 'flow_centrality',
    disable_progress: bool = False
):

    syn_con = nbm.match_connectors_to_nodes(syn_con, n, synapse_type = synapse_type)

    n_copy = node_to_compartment_type(n, split)

    n2comp = dict(zip(n_copy.nodes.node_id, n_copy.nodes.compartment))

    if synapse_type == 'pre':

        syn_con['pre_node_type'] = syn_con.node.map(n2comp)

        ind_to_compart_post = {}

        for i in tqdm(syn_con.bodyId_post.unique(), disable = disable_progress):

            post_n = nvneu.fetch_skeletons(i, heal = True)[0]
            post_split = navis.split_axon_dendrite(n, metric = metric)
            post_n_copy = node_to_compartment_type(post_n, post_split)

            post_n2comp = dict(zip(post_n_copy.nodes.node_id, post_n_copy.nodes.compartment))
            sub = syn_con[syn_con.bodyId_post == i].copy()
            sub_matched = nbm.match_connectors_to_nodes(sub, post_n_copy, synapse_type = 'post')

            tmp_comp = [post_n2comp[i] for i in sub_matched.node]
            ind_comp = dict(zip(list(sub_matched.index), tmp_comp))

            ind_to_compart_post.update(ind_comp)

        syn_con['post_node_type'] = list(syn_con.index.map(ind_to_compart_post))

    elif synapse_type == 'post':

        syn_con['post_node_type'] = syn_con.node.map(n2comp)

        ind_to_compart_pre = {}

        for i in tqdm(syn_con.bodyId_pre.unique(), disable = disable_progress):

            pre_n = nvneu.fetch_skeletons(i, heal = True)[0]
            pre_split = navis.split_axon_dendrite(n, metric = metric)
            pre_n_copy = node_to_compartment_type(pre_n, pre_split)

            pre_n2comp = dict(zip(pre_n_copy.nodes.node_id, pre_n_copy.nodes.compartment))
            sub = syn_con[syn_con.bodyId_pre == i].copy()
            sub_matched = nbm.match_connectors_to_nodes(sub, pre_n_copy, synapse_type = 'pre')

            tmp_comp = [pre_n2comp[i] for i in sub_matched.node]
            ind_comp = dict(zip(list(sub_matched.index), tmp_comp))

            ind_to_compart_pre.update(ind_comp)

        syn_con['pre_node_type'] = list(syn_con.index.map(ind_to_compart_pre))

    syn_con['connection_type'] = syn_con[['pre_node_type','post_node_type']].agg('-'.join, axis = 1)

    return(syn_con)
