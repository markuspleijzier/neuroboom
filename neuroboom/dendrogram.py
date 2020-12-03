# INSERT LICENSE

import time
import navis
# from typing import Tuple, Optional, List, Union
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from neuroboom.utils import calc_cable
# from neuroboom.utils import check_valid_neuron_input
from itertools import chain
# import plotly as py
# from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.offline import plot, iplot


# from logging import
# This script contains functions for plotting dendrograms, static and interactive

# dendro

VALID_PROGS = ["fdp", "dot", "neato"]


# def plot_dendrogram(
#     x: Union[navis.TreeNeuron, navis.neuronlist.NeuronList],
#     heal_neuron: bool = True,
#     downsample_neuron: float = 0.0,
#     plot_connectors: bool = True,
#     connector_confidence: Tuple[float, float] = (0.0, 0.0),
#     highlight_connectors: Optional = None,
#     fragment: bool = False,
#     presyn_color: List[List[float]] = [[0.9, 0.0, 0.0]],
#     postsyn_color: List[List[float]] = [[0.0, 0.0, 0.9]],
#     highlight_connector_color: List[List[float]] = [[0.0, 0.9, 0.0]],
#     highlight_connector_size: int = 20,
#     presyn_size: float = 0.1,
#     postsyn_size: float = 0.1,
#     prog: str = "dot",
# ) -> None:
#
#     # Typing & Input sanity checking
#     x = check_valid_neuron_input(x)
#
#     assert isinstance(
#         connector_confidence, tuple
#     ), f"Need to pass a tuple for confidence values. You have passed: {connector_confidence}"
#     assert (
#         len(connector_confidence) == 2
#     ), """Need to pass a tuple containing two values for confidence. \n
#            The first value is the confidence threshold for presynapses. \n
#           The second value is the confidence threshold for postsynapses
#        """
#
#     assert (
#         prog in VALID_PROGS
#     ), f"Unknown program parameter!. Please specify one of: {VALID_PROGS}"
# =======

# This script contains functions for plotting dendrograms, static and interactive
# Create Graph Structure

def create_graph_structure(x, returned_object='graph', prog='dot'):
    """
    Takes a navis neuron and creates a graph layout

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
            raise ValueError('Need to pass a SINGLE neuron')
        else:
            x = x[0]

    valid_objects = ['graph', 'positions', 'graph_and_positions']
    if returned_object not in valid_objects:
        raise ValueError('Unknown object type to return!')

    valid_progs = ['fdp', 'dot', 'neato']
    if prog not in valid_progs:
        raise ValueError('Unknown program parameter!')

    print('Creating Graph Structure...')
    g = nx.DiGraph()
    g.add_nodes_from(x.nodes.node_id)
    for e in x.nodes[['node_id', 'parent_id', 'parent_dist']].values:
        # Skip root node
        if e[1] == -1:
            continue
        g.add_edge(int(e[0]), int(e[1]), len=e[2])

    if returned_object == 'graph':
        print('Returning graph only')
        return(g)

    elif returned_object == 'positions':
        print('Calculating layout...')
        pos = nx.nx_agraph.graphviz_layout(g, prog=prog)
        print('Returning positions only')
        return(pos)

    elif returned_object == 'graph_and_positions':
        print('Calculating layout...')
        pos = nx.nx_agraph.graphviz_layout(g, prog=prog)
        print('Returning graph and positions')
        return(g, pos)

# dendrogram


def plot_dendrogram(x, heal_neuron=True,
                    downsample_neuron=0.0,
                    plot_connectors=True,
                    connector_confidence=(0.0, 0.0),
                    highlight_connectors=None,
                    fragment=False,
                    presyn_color=[[.9, .0, .0]],
                    postsyn_color=[[.0, .0, .9]],
                    highlight_connector_color=[[.0, .9, .0]],
                    highlight_connector_size=20,
                    presyn_size=.1, postsyn_size=.1,
                    prog='dot'):

    """
    Takes a navis neuron and creates a graph layout

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
            raise ValueError('Need to pass a SINGLE Neuron')
        else:
            x = x[0]

    if not isinstance(connector_confidence, tuple):
        raise ValueError('Need to pass a tuple for confidence values')
    elif isinstance(connector_confidence, tuple):
        if len(connector_confidence) != 2:
            raise ValueError('''Need to pass a tuple containing two values for confidence.
                             \n The first value is the confidence threshold for presynapses.
                             \n The second value is the confidence threshold for postsynapses''')

    valid_progs = ['fdp', 'dot', 'neato']
    if prog not in valid_progs:
        raise ValueError('Unknown program parameter!')

    start = time.time()

    if heal_neuron:
        print("Healing Neuron...")
        navis.heal_fragmented_neuron(x, inplace=True)

    if any(connector_confidence) > 0.0:
        print(
            """Thresholding synapses: only considering
        presynapses above {} confidence and postsynapses above {}""".format(
                connector_confidence[0], connector_confidence[1]
            )
        )
        presyn_included = x.connectors[x.connectors.type == "pre"][
            x.connectors.confidence > connector_confidence[0]
        ].connector_id.tolist()
        postsyn_included = x.connectors[x.connectors.type == "post"][
            x.connectors.confidence > connector_confidence[1]
        ].connector_id.tolist()
        connectors_included = list(
            chain.from_iterable([presyn_included, postsyn_included])
        )
        x.connectors = x.connectors[x.connectors.connector_id.isin(connectors_included)]

    if downsample_neuron > 0:
        print("Downsampling neuron, factor = {}".format(downsample_neuron))
        x = navis.downsample_neuron(
            x,
            downsampling_factor=downsample_neuron,
            preserve_nodes=x.connectors.node_id.unique().tolist(),
        )

    if "parent_dist" not in x.nodes:
        print("Calculating cable length...")
        x = calc_cable(x, return_skdata=True)

    g, pos = create_graph_structure(x, returned_object="graph_and_positions", prog=prog)

    # Plotting tree with the above layout
    print("Plotting Tree...")
    nx.draw(g, pos, node_size=0, arrows=False, width=0.25)

    # Whether to add soma or not
    if not fragment:
        print("Plotting soma")
        plt.scatter([pos[x.soma][0]], [pos[x.soma][1]], s=40, c=[[0, 0, 0]], zorder=1)

    if plot_connectors is not False:
        print("Plotting connectors...")
        plt.scatter(
            [
                pos[tn][0]
                for tn in x.connectors[x.connectors.type == "pre"].node_id.values
            ],
            [
                pos[tn][1]
                for tn in x.connectors[x.connectors.type == "pre"].node_id.values
            ],
            c=presyn_color,
            zorder=2,
            s=presyn_size,
            linewidths=1,
        )

        plt.scatter(
            [
                pos[tn][0]
                for tn in x.connectors[x.connectors.type == "post"].node_id.values
            ],
            [
                pos[tn][1]
                for tn in x.connectors[x.connectors.type == "post"].node_id.values
            ],
            c=postsyn_color,
            zorder=2,
            s=postsyn_size,
            linewidths=1,
        )

    if highlight_connectors is not None:

        if isinstance(highlight_connectors, (list, np.ndarray)) is True:
            hl_cn_coords = np.array(
                [
                    pos[tn]
                    for tn in x.connectors[
                        x.connectors.connector_id.isin(highlight_connectors)
                    ].treenode_id
                ]
            )
            plt.scatter(
                hl_cn_coords[:, 0],
                hl_cn_coords[:, 1],
                s=highlight_connector_size,
                c=highlight_connector_color,
                zorder=3,
                linewidths=1,
            )

        elif isinstance(highlight_connectors, dict) is True:
            for cn in highlight_connectors:
                if cn in highlight_connectors:
                    if cn is None:
                        continue
                    if cn not in x.connectors.connector_id.values:
                        print(
                            "Connector {} is not present in the neuron / neuron fragment".format(
                                cn
                            )
                        )
                    hl_cn_coords = np.array(
                        [
                            pos[tn]
                            for tn in x.connectors[
                                x.connectors.connector_id == cn
                            ].treenode_id
                        ]
                    )
                    plt.scatter(
                        hl_cn_coords[:, 0],
                        hl_cn_coords[:, 1],
                        s=highlight_connector_size,
                        c=highlight_connectors[cn],
                        zorder=3,
                    )

        else:

            raise TypeError(
                "Unable to highlight connectors from data of type {}".format(
                    type(highlight_connectors)
                )
            )

    print("Completed in %is" % int(time.time() - start))


def interactive_dendrogram(
    z,
    heal_neuron=True,
    plot_connectors=True,
    highlight_connectors=None,
    in_volume=None,
    prog="dot",
    inscreen=True,
    filename=None,
):
    """
    Takes a navis neuron and creates a graph layout

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
    z = check_valid_neuron_input(z)
    assert prog in VALID_PROGS, "Unknown program parameter!"

    if heal_neuron:
        z = navis.heal_fragmented_neuron(z)

    # save start time
    start = time.time()

    # Necessary for neato layouts for preservation of segment lengths

    if "parent_dist" not in z.nodes:
        z = calc_cable(z, return_skdata=True)

    # Generation of networkx diagram

    g, pos = create_graph_structure(z, returned_object="graph_and_positions", prog=prog)

    print("Now creating plotly graph...")

    # Convering networkx nodes for plotly
    # NODES

    x = []
    y = []
    node_info = []

    for n in g.nodes():
        x_, y_ = pos[n]

        x.append(x_)
        y.append(y_)
        node_info.append("Node ID: {}".format(n))

    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers",
        text=node_info,
        hoverinfo="text",
        marker=go.scatter.Marker(showscale=False),
    )

    # EDGES

    xe = []
    ye = []

    for e in g.edges():

        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]

        xe += [x0, x1, None]
        ye += [y0, y1, None]

    edge_trace = go.Scatter(
        x=xe,
        y=ye,
        line=go.scatter.Line(width=1.0, color="#000"),
        hoverinfo="none",
        mode="lines",
    )

    # SOMA

    xs = []
    ys = []

    for n in g.nodes():
        if n != z.soma:
            continue
        elif n == z.soma:

            x__, y__ = pos[n]
            xs.append(x__)
            ys.append(y__)

        else:
            break

    soma_trace = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        hoverinfo="text",
        marker=dict(size=20, color="rgb(0,0,0)"),
        text="Soma, treenode_id:{}".format(z.soma),
    )

    # CONNECTORS:
    # RELATION  = 0 ARE PRESYNAPSES, RELATION = 1 ARE POSTSYNAPSES

    if plot_connectors is False:
        presynapse_connector_trace = go.Scatter()
        postsynapse_connector_trace = go.Scatter()

    elif plot_connectors is True:
        presynapse_connector_list = list(
            z.connectors[z.connectors.type == "pre"].node_id.values
        )

        x_pre = []
        y_pre = []
        presynapse_connector_info = []

        for node in g.nodes():
            for tn in presynapse_connector_list:

                if node == tn:

                    x, y = pos[node]

                    x_pre.append(x)
                    y_pre.append(y)
                    presynapse_connector_info.append(
                        "Presynapse, connector_id: {}".format(tn)
                    )

        presynapse_connector_trace = go.Scatter(
            x=x_pre,
            y=y_pre,
            text=presynapse_connector_info,
            mode="markers",
            hoverinfo="text",
            marker=dict(size=10, color="rgb(0,255,0)"),
        )

        postsynapse_connectors_list = list(
            z.connectors[z.connectors.type == "post"].node_id.values
        )

        x_post = []
        y_post = []
        postsynapse_connector_info = []

        for node in g.nodes():
            for tn in postsynapse_connectors_list:

                if node == tn:

                    x, y = pos[node]

                    x_post.append(x)
                    y_post.append(y)

                    postsynapse_connector_info.append(
                        "Postsynapse, connector id: {}".format(tn)
                    )

        postsynapse_connector_trace = go.Scatter(
            x=x_post,
            y=y_post,
            text=postsynapse_connector_info,
            mode="markers",
            hoverinfo="text",
            marker=dict(size=10, color="rgb(0,0,255)"),
        )

    if highlight_connectors is None:

        HC_trace = go.Scatter()

    elif highlight_connectors is not None:

        HC_nodes = []

        for i in highlight_connectors:

            HC_nodes.append(
                z.connectors[z.connectors.connector_id == i].node_id.values[0]
            )

            HC_x = []
            HC_y = []

            HC_info = []

            for node in g.nodes():

                for tn in HC_nodes:

                    if node is tn:

                        x, y = pos[node]

                        HC_x.append(x)
                        HC_y.append(y)

                        HC_info.append(
                            "Connector of Interest, connector_id: {}, treenode_id: {}".format(
                                z.connectors[
                                    z.connectors.node_id == node
                                ].connector_id.values[0],
                                node,
                            )
                        )

            HC_trace = go.Scatter(
                x=HC_x,
                y=HC_y,
                text=HC_info,
                mode="markers",
                hoverinfo="text",
                marker=dict(size=15, color="rgb(238,0,255)"),
            )

    # Highlight the nodes that are in a particular volume

    if in_volume is not None:

        # Volume of interest

        res = navis.in_volume(z.nodes, volume=in_volume, mode="IN")

        z.nodes["IN_VOLUME"] = res

        x_VOI = []
        y_VOI = []

        VOI_info = []

        for nodes in g.nodes():

            for tn in list(z.nodes[z.nodes.IN_VOLUME == True].node_id.values):

                if nodes == tn:

                    x, y = pos[tn]

                    x_VOI.append(x)
                    y_VOI.append(y)

                    VOI_info.append("Treenode {} is in {} volume".format(tn, in_volume))

        in_volume_trace = go.Scatter(
            x=x_VOI,
            y=y_VOI,
            text=VOI_info,
            mode="markers",
            hoverinfo="text",
            marker=dict(size=5, color="rgb(35,119,0)"),
        )

    print("Creating Plotly Graph")

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace,
            soma_trace,
            presynapse_connector_trace,
            postsynapse_connector_trace,
            HC_trace,
            in_volume_trace,
        ],
        layout=go.Layout(
            title="Plotly graph of {} with {} layout".format(z.name, prog),
            titlefont=dict(size=16),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=50, r=5, t=40),
            annotations=[
                dict(showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
            ],
            xaxis=go.layout.XAxis(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=go.layout.YAxis(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    print("Time taken: {}".format(time.time() - start))

    if inscreen is True:

        return iplot(fig)

    else:

        return plot(fig, filename=filename)
