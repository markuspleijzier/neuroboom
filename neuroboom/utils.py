# INSERT LICENSE
# This script contains utility functions
# Importing dependencies

from typing import Optional, Union, Any

import navis
import networkx as nx
import numpy as np
import pymaid

# Calculating cable length between nodes - this function does not exist for navis neurons


def check_valid_neuron_input(x: Any) -> Optional[navis.TreeNeuron]:
    """
    Takes an object and checks whether it is a navis TreeNeuron object

    Parameters
    --------
    x: a python object: hopefully a navis.TreeNeuron object

    Returns
    --------
    x: returns x if it is a navis.TreeNeuron object, otherwise will raise an assertion error

    """
    assert isinstance(
        x, (navis.TreeNeuron, navis.NeuronList)
    ), f"Need to pass a Navis Tree Neuron type. You have passed: {x}"

    if isinstance(x, navis.NeuronList):
        assert len(x) < 1, "Need to pass a SINGLE Neuron"
        x = x[0]
    return x


def calc_cable(
    x: Union[navis.TreeNeuron, navis.NeuronList], return_skdata: bool = False
):

    """
    # Calculate cable length between nodes - this function does not exist for navis neurons
    :param x:
    :param return_skdata:
    :return:
    """

    x = check_valid_neuron_input(x)

    nodes = x.nodes[~x.nodes.parent_id.isnull()]
    tn_coords = nodes[["x", "y", "z"]].values

    this_tn = nodes.set_index("node_id")
    parent_coords = this_tn.reindex(nodes.parent_id.values, ["x", "y", "z"]).values

    w = np.sqrt(np.sum((tn_coords - parent_coords) ** 2, axis=1))

    if return_skdata:
        nodes.loc[~nodes.parent_id.isnull(), "parent_dist"] = w / 1000
        x.nodes = nodes
        return x

    return np.sum(w[np.logical_not(np.isnan(w))]) / 1000


def check_valid_pymaid_input(x: Any) -> Optional[pymaid.core.CatmaidNeuron]:

    """
    Takes an object and checks whether it is a navis TreeNeuron object

    Paramters
    --------
    x: a python object: hopefully a navis.TreeNeuron object

    Returns
    --------
    x: returns x if it is a navis.TreeNeuron object, otherwise will raise an assertion error

    """
    assert isinstance(
        x, (pymaid.core.CatmaidNeuron, pymaid.core.CatmaidNeuronList)
    ), f"Need to pass a Navis Tree Neuron type. You have passed: {x}"

    if isinstance(x, pymaid.core.CatmaidNeuron):
        assert len(x) < 1, "Need to pass a SINGLE Neuron"
        x = x[0]
    return x


def pymaid_topological_sort(
    x: Union[pymaid.core.CatmaidNeuron, pymaid.core.CatmaidNeuronList],
    return_object: str = "list",
):

    """
    Takes pymaid/CatmaidNeuron and topologically sorts the nodes

    Paramters
    ---------
    x:                a pymaid/Catmaid neuron object
    return_skdata:    bool
                    whether to return a list of node_ids topologically sorted
                    or to return a dict where the keys are treenodes and the values
                    are ranking in the topological sort

    Returns
    --------
    x: list or dict

    Examples
    --------



    """

    x = check_valid_pymaid_input(x)

    g = nx.DiGraph()
    g.add_node(x.nodes[x.nodes.type == "root"].treenode_id.values[0])
    g.add_nodes_from(
        [
            i
            for i in x.nodes.node_id.tolist()
            if i != x.nodes[x.nodes.type == "root"].treenode_id.values[0]
        ]
    )

    for e in x.nodes[["treenode_id", "parent_id"]].values:

        if e[1] is not None:

            g.add_edge(e[1], e[0])

        else:

            continue

    topological_sort = [i for i in nx.topological_sort(g)]

    if return_object == "list":
        return topological_sort
    elif return_object == "dict":
        topological_sort = dict(zip(topological_sort, range(0, len(topological_sort))))
        return topological_sort


def pymaid_to_navis(x: Union[pymaid.core.CatmaidNeuron, pymaid.core.CatmaidNeuronList]):

    """
    Takes pymaid/CatmaidNeuron and topologically sorts the nodes

    Paramters
    ---------
    x:                a pymaid/Catmaid neuron object

    return_skdata:    bool
                    whether to return a list of node_ids topologically sorted
                    or to return a dict where the keys are treenodes and the values
                    are ranking in the topological sort

    Returns
    --------
    x: list or dict

    Examples
    --------
    """

    x = check_valid_pymaid_input(x)

    x.nodes["rank"] = x.nodes.treenode_id.map(
        pymaid_topological_sort(x, return_object="dict")
    )
    x.nodes.sort_values(by=["rank"], ascending=True, inplace=True)

    # Getting the topological sort of the pymaid neuron
    x_graph = x.graph
    navis_neuron = navis.TreeNeuron(x_graph)

    # Populating the xyz columns of nodes

    for i, j in enumerate(x.nodes[["x", "y", "z"]].values):

        navis_neuron.nodes.loc[i, "x"] = j[0]
        navis_neuron.nodes.loc[i, "y"] = j[1]
        navis_neuron.nodes.loc[i, "z"] = j[2]

    # adding type column
    navis_neuron.nodes["type"] = x.nodes.type.copy()

    # adding connectors
    navis_neuron.connectors = x.connectors.copy()
    navis_neuron.connectors = [
        "pre" if i == 0 else "post" for i in navis_neuron.connectors.type
    ]

    # adding soma & name

    navis_neuron.soma = x.soma
    navis_neuron.name = x.neuron_name

    return navis_neuron
