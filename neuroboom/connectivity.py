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

        neurons = con.bodyId_post.unique()
        n, _ = nvneu.fetch_neurons(neurons)
        partner_type_dict = dict(zip(n.bodyId.tolist(), n.type.tolist()))

        count = Counter(con.bodyId_post)
        count = count.most_common()
        count_dict = dict(count)

    elif presyn_postsyn == "post":

        con = nvneu.fetch_synapse_connections(source_criteria=x)

        if roi:

            tt = navis.in_volume(con[["x_post", "y_post", "z_post"]].values, roi)
            con = con[tt].copy()

        neurons = con.bodyId_pre.unique()
        n, _ = nvneu.fetch_neurons(neurons)
        partner_type_dict = dict(zip(n.bodyId.tolist(), n.type.tolist()))

        count = Counter(con.bodyId_pre)
        count = count.most_common()
        count_dict = dict(count)

    df = pd.DataFrame(
        columns=[x], index=[i for i, j in count], data=[count_dict[i] for i, j in count]
    )

    if rename_index:

        df.index = [partner_type_dict[i] for i in df.index]

    return (df, partner_type_dict)
