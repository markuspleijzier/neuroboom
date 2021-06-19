# INSERT LICENSE
from collections import Counter
from typing import List, Optional

import navis
import navis.interfaces.neuprint as nvneu
import pandas as pd

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

        df = pd.DataFrame(columns = [x], index = [i for i, j in count], data = [count_dict[i] for i, j in count])

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

        df = pd.DataFrame(index = [i for i, j in count], columns = [x], data = [count_dict[i] for i, j in count])
        df = df.T.copy()

    # df = pd.DataFrame(
    #     columns=[x], index=[i for i, j in count], data=[count_dict[i] for i, j in count])

    if rename_index:

        df.index = [partner_type_dict[i] for i in df.index]

    return (df, partner_type_dict)


def match_connectors_to_nodes(
    synapse_connections: pd.DataFrame,
    neuron: navis.TreeNeuron,
    synapse_type: str = 'post'
):
    """
    Matches connections to skeleton nodes

    Parameters
    ----------
    synapse_connections :       pandas.DataFrame
                                A pandas dataframe containing the synapse connections of the neuron of interest
                                synapse connections is created using nvneu.fetch_synapse_connections()

    neuron :                    navis.TreeNeuron
                                A navis TreeNeuron skeleton of the neuron of interest

    synapse_type :              str
                                A string of ['pre' or 'post'] determining whether presynapses or postsynapses are matched to nodes

    """

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
