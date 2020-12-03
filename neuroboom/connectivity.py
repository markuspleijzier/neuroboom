# INSERT LICENSE
import navis.interfaces.neuprint as nvneu
from collections import Counter
import pandas as pd
import navis

# Create an adjacency matrix from synapse connections (neuprint)


def adjx_from_syn_conn(x, roi, rename_index=False):
    """
    Creates an adjacency matrixx from synapse connections

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

    presyn_con = nvneu.fetch_synapse_connections(source_criteria=x)
    postsyn_con = nvneu.fetch_synapse_connections(target_criteria=x)

    pre_tt = navis.in_volume(presyn_con[['x_pre', 'y_pre', 'z_pre']].values, roi)
    post_tt = navis.in_volume(postsyn_con[['x_pre', 'y_pre', 'z_pre']].values, roi)

    roi_presyn_con = presyn_con[pre_tt].copy()
    roi_postsyn_con = postsyn_con[post_tt].copy()

    postsyn_roi_neurons = roi_presyn_con.bodyId_post.unique()

    n, _ = nvneu.fetch_neurons(postsyn_roi_neurons)
    partner_type_dict = dict(zip(n.bodyId.tolist(), n.type.tolist()))

    count = Counter(roi_presyn_con.bodyId_post)
    count = count.most_common()
    count_dict = dict(count)

    df = pd.DataFrame(columns=[x], index=[i for i, j in count], data=[count_dict[i] for i, j in count])

    if rename_index:

        df.index = [partner_type_dict[i] for i in df.index]

    return(df, partner_type_dict)
