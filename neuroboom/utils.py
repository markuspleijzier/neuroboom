# INSERT LICENSE


# This script contains utility functions

# Importing dependencies

import navis
import numpy as np
import networkx as nx
import pymaid

# Calculating cable length between nodes - this function does not exist for navis neurons


def calc_cable(x, return_skdata=False):

    """
    Calculates cable length of navis neurons

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

    nodes = x.nodes[~x.nodes.parent_id.isnull()]
    tn_coords = nodes[['x', 'y', 'z']].values

    this_tn = nodes.set_index('node_id')
    parent_coords = this_tn.reindex(nodes.parent_id.values, ['x', 'y', 'z']).values

    w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1))

    if return_skdata:
        nodes.loc[~nodes.parent_id.isnull(), 'parent_dist'] = w / 1000
        x.nodes = nodes
        return(x)

    return np.sum(w[np.logical_not(np.isnan(w))]) / 1000


def topological_sorting_of_nodes(x, return_object=['list', 'dict']):

    """
    Topological sorting of treenodes. Nodes and their parents are adjacent in the treenode table

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

    g = nx.DiGraph()
    g.add_node(x.nodes[x.nodes.type == 'root'].treenode_id.values[0])
    root = x.nodes[x.nodes.type == 'root'].treenode_id.values[0]
    nodes_to_add = [i for i in x.nodes.treenode_id.tolist() if i != root]
    g.add_nodes_from(nodes_to_add)

    for e in x.nodes[['treenode_id', 'parent_id']].values:

        if e[1] is None:

            continue

        else:

            g.add_edge(e[1], e[0])

    topological_sort = [i for i in nx.topological_sort(g)]

    if return_object == 'list':

        return(topological_sort)

    elif return_object == 'dict':

        topological_sort = dict(zip(topological_sort, range(0, len(topological_sort))))

        return(topological_sort)


######################################################################

def prepare_neuron(x, change_units=True, factor=1e3):
    """
    Prepares neuron for electrotonic modelling

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
    node_sort = dict([(i, k) for i, k in zip(range(len(x.nodes)), pymaid.node_label_sorting(x))])
    node_sort_rev = {i: j for j, i in node_sort.items()}
    # x = pymaid.downsample_neuron(x, resampling_factor = float('inf'))
    # x = pymaid.guess_radius(x)
    x.nodes['rank'] = x.nodes.treenode_id.map(node_sort_rev).tolist()
    x.nodes.sort_values(by=['rank'], ascending=True, inplace=True)
    x.nodes.reset_index(drop=True, inplace=True)

    x = pymaid.calc_cable(x, return_skdata=True)

    if not change_units:
        return(x)
    else:
        x.nodes['x'] = x.nodes['x'] / factor
        x.nodes['y'] = x.nodes['y'] / factor
        x.nodes['z'] = x.nodes['z'] / factor
        x.nodes['radius'] = x.nodes['radius'] / factor
        x.nodes['parent_dist'] = x.nodes['parent_dist'] / factor
        return(x)


def pymaid_to_navis(x):
    """
    Converts pymaid neuron types to navis


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

    if isinstance(x, pymaid.core.CatmaidNeuron):
        pass
    else:
        return('Need to pass a Catmaid Neuron!')

    if isinstance(x, pymaid.core.CatmaidNeuronList):
        if len(x) > 1:
            return('Need to pass a single Catmaid Neuron')
        else:
            x = x[0]

    # Getting topological sort of pymaid neuron
    x.nodes['rank'] = x.nodes.treenode_id.map(topological_sorting_of_nodes(x, return_object='dict'))
    x.nodes.sort_values(by=['rank'], ascending=True, inplace=True)

    # Getting the graph object of pymaid neuron and passing that to an empty navis TreeNeuron object
    x_graph = x.graph
    navis_neuron = navis.TreeNeuron(x_graph)

    # Populating the xyz columns of nodes
    for i, j in enumerate(x.nodes[['x', 'y', 'z']].values):

        navis_neuron.nodes.loc[i, 'x'] = j[0]
        navis_neuron.nodes.loc[i, 'y'] = j[1]
        navis_neuron.nodes.loc[i, 'z'] = j[2]

    # Adding type column
    navis_neuron.nodes['type'] = x.nodes.type.copy()

    # Adding connectors
    navis_neuron.connectors = x.connectors.copy()
    navis_neuron.connectors = ['pre' if i == 0 else 'post' for i in navis_neuron.connectors.type]

    # Adding soma
    navis_neuron.soma = x.soma

    # Adding name
    navis_neuron.name = x.neuron_name

    return(navis_neuron)
