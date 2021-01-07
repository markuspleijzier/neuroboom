import time
from itertools import chain
from typing import List, Optional, Tuple, Union

import navis
import networkx as nx
import numpy as np
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from plotly.offline import iplot, plot

from neuroboom.utils import calc_cable, check_valid_neuron_input

# from logging import
# This script contains functions for plotting dendrograms, static and interactive


def create_graph_structure(
    x: Union[navis.TreeNeuron, navis.NeuronList],
    returned_object: str = "graph",
    prog: str = "dot",
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

    x = check_valid_neuron_input(x)

    valid_objects = ["graph", "positions", "graph_and_positions"]

    assert isinstance(
        returned_object, str
    ), f"You need to pass a string for the returned object. You have passed: {returned_object}"

    assert (
        returned_object in valid_objects
    ), f"You need to pass a valid object to return. These include: {valid_objects}"

    valid_progs = ["fdp", "dot", "neato"]

    assert (
        prog in valid_progs
    ), f"Invalid program parameter. You need to pass one of {valid_progs}"

    print("Creating Graph Structure...")

    g = nx.DiGraph()
    g.add_nodes_from(x.nodes.node_id)
    for e in x.nodes[["node_id", "parent_id", "parent_dist"]].values:
        # Skip root node
        if e[1] == -1:
            continue
        g.add_edge(int(e[0]), int(e[1]), len=e[2])

    if returned_object == "graph":
        print("Returning graph only")
        return g

    elif returned_object == "positions":
        print("Calculating layout...")
        pos = nx.nx_agraph.graphviz_layout(g, prog=prog)
        print("Returning positions only")
        return pos

    elif returned_object == "graph_and_positions":
        print("Calculating layout...")
        pos = nx.nx_agraph.graphviz_layout(g, prog=prog)
        print("Returning graph and positions")
        return (g, pos)


# dendrogram


def plot_dendrogram(
    x: Union[navis.TreeNeuron, navis.NeuronList],
    heal_neuron: bool = False,
    downsample_neuron: float = 0.0,
    plot_connectors: bool = True,
    connector_confidence: Tuple[float, float] = (0.0, 0.0),
    highlight_connectors: Optional = None,
    fragment: bool = False,
    presyn_color: List[List[float]] = [[0.9, 0.0, 0.0]],
    postsyn_color: List[List[float]] = [[0.0, 0.0, 0.9]],
    highlight_connector_color: List[List[float]] = [[0.0, 0.9, 0.0]],
    highlight_connector_size: int = 20,
    presyn_size: float = 0.1,
    postsyn_size: float = 0.1,
    prog: str = "dot",
):

    """
    This function creates a 2-dimensional dendrogram, a 'flattened' version of a neuron.
    Dendrograms can be used to visualise the locations of specific partner synapses.

    Parameters
    ----------
    x :                navis.TreeNeuron

                        A single navis tree neuron object

    heal_neuron :      bool

                        Whether you want to heal the neuron or not. N.B. Navis neurons
                        should be healed on import, i.e. navis.fetch_skeletons(bodyid, heal = True)
                        see navis.fetch_skeletons and navis.heal_fragmented_neuron for more details

    downsample_neuron: float

                        A float specifying the downsampling factor used by navis.downsample_neuron()
                        If 0.0, then no downsampling will occur. If float('inf') then this will reduce
                        the neuron to branch and end points.

                        It is recommended to downsample very large neurons when testing out this code
                        for the first time.

    plot_connectors:   bool

                        Whether to plot presynapses and postsynapses on the dendrogram or not.

    connector_confidence: tuple

                        The confidence value used to threshold the synapses.
                        The first value (connector_confidence[0]) will be used to threshold presynapses
                        The second value (connector_confidence[1]) will be used to threshold postsynapses

    highlight_connectors: optional | np.array | dict

                        If a numpy array, then this should be an array of the treenodes
                        connected to the connectors that you want to highlight.
                        The single color and size will be specified in
                        highlight_connector_color and highlight_connector_size

                        If a dictionary, then the key values should be
                        treenode ids of the connectors you want to highlight
                        and their values should be the color you want to colour them.

                        Passing dictionaries to this parameter allow for synapses to be coloured differently

    fragment:           bool

                        Whether the neuron object you are passing
                        is a fragment or not (i.e. does it have a soma or not)

    presyn_color:       list

                        A list containing the rgb values that you want to colour the presynapses.
                        All presynapses will be coloured this color

    postsyn_color:      list

                        A list containing the rgb values that you want to colour the postsynapses.
                        All postsynapses will be coloured this color

    highlight_connector_color: list

                        A list containing the rgb values that you want to color your special synapses

    highlight_connector_size: int

                        The size of the synapses you want to highlight on the dendrogram

    presyn_size:         int

                        The size of all presynapses on the dendrogram

    postsyn_size:        int

                        The size of all postsynapses on the dendrogrm

    prog :             str

                       The layout type used by navis.nx_agraph.graphviz_layout()
                       Valid programs include [dot, neato or fdp].
                       The dot program provides a hierarchical layout, this is the fastest program
                       The neato program creates edges between nodes proportional to their real length.
                       The neato program takes the longest amount of time, can be ~2hrs for a single neuron!


    Returns
    -------
    fig: a figure of containing the dendrogram


    Example
    --------
    from neuroboom.utils import create_graph_structure
    from neuroboom.dendrogram import plot_dendrogram
    import navis.interfaces.neuprint as nvneu
    from matplotlib import pyplot as plt

    test_neuron = nvneu.fetch_skeletons(722817260)

    plt.clf()
    fig, ax = plt.subplots(figsize = (20,20))
    plot_dendrogram(test_neuron, prog = 'dot')
    plt.show()

    """

    x = check_valid_neuron_input(x)

    assert isinstance(
        connector_confidence, tuple
    ), f"Need to pass a tuple for confidence values. You have passed a {type(connector_confidence)}"

    assert (
        len(connector_confidence) == 2
    ), """
    Need to pass a tuple containing two values for confidence. \n
    The first value is the confidence threshold for presynapses. \n
    The second value is the confidence threshold for postsynapses. """

    valid_progs = ["fdp", "dot", "neato"]

    assert (
        prog in valid_progs
    ), f"Invalid program parameter. You need to pass one of {valid_progs}"

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
        plt.scatter([pos[x.soma][0]], [pos[x.soma][1]], s=80, c=[[0, 0, 0]], zorder=1)

    if plot_connectors:
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

        if isinstance(highlight_connectors, (list, np.ndarray)):
            hl_cn_coords = np.array(
                [
                    pos[tn]
                    for tn in x.connectors[
                        x.connectors.connector_id.isin(highlight_connectors)
                    ].node_id
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

        elif isinstance(highlight_connectors, dict):
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
                            ].node_id
                        ]
                    )
                    plt.scatter(
                        hl_cn_coords[:, 0],
                        hl_cn_coords[:, 1],
                        s=highlight_connector_size,
                        color=highlight_connectors[cn],
                        zorder=3,
                    )

        else:

            raise TypeError(
                "Unable to highlight connectors from data of type {}".format(
                    type(highlight_connectors)
                )
            )

    print("Completed in %is" % int(time.time() - start))


# Plot an interactive dendrogram
def interactive_dendrogram(
    z: Union[navis.TreeNeuron, navis.NeuronList],
    heal_neuron: bool = True,
    plot_nodes: bool = True,
    plot_connectors: bool = True,
    highlight_connectors: Optional = None,
    in_volume: Optional = None,
    prog: str = "dot",
    inscreen: bool = True,
    filename: Optional = None,
):

    """
    Takes a navis neuron and returns an interactive 2D dendrogram.
    In this dendrogram, nodes or connector locations can be highlighted


    Parameters
    ----------
    x:                    A navis neuron object

    heal_neuron: bool

                    Whether you want to heal the neuron or not. N.B. Navis neurons
                    should be healed on import, i.e. navis.fetch_skeletons(bodyid, heal = True)
                    see navis.fetch_skeletons and navis.heal_fragmented_neuron for more details

    plot_nodes:           bool

                    Whether treenodes should be plotted

    plot_connectors:      bool

                    Whether connectors should be plotted

    highlight_connectors: dict

                    A dictionary containing the treenodes of
                    the connectors you want to highlight as keys
                    and the colours you want to colour them as values.
                    This allows for multiple colours to be plotted.

                    N.B. Plotly colours are in the range of 0 - 255
                    whereas matplotlib colours are between 0-1. For the
                    interactive dendrogram colours need to be in the
                    plotly range, whereas in the static dendrogram
                    the colours need to be in the matplotlib range.

    in_volume:            navis.Volume object

                    A navis.Volume object corresponding to an ROI in the brain.
                    This will then highlight the nodes of
                    the neuron which are in that volume


    prog:                 str

                    The layout type used by navis.nx_agraph.graphviz_layout()
                    Valid programs include [dot, neato or fdp].
                    The dot program provides a hierarchical layout, this is the fastest program
                    The neato program creates edges between nodes proportional to their real length.
                    The neato program takes the longest amount of time, can be ~2hrs for a single neuron!

    inscreen:             bool

                    Whether to plot the graph inscreen (juptyer notebooks) or to plot it as a
                    separate HTML file that can be saved to file, opened in the browser and opened any time

    filename:             str

                    The filename of your interactive dendrogram html file. This parameter is only appropriate
                    when inscreen = False.



    Returns
    -------
    plotly.fig object containing the dendrogram - this can be either inscreen or as a separate html file
    with the filename specified by the filename parameter


    Examples
    --------
    from neuroboom.utils import create_graph_structure
    from neuroboom.dendrogram import interactive_dendrogram
    import navis.interfaces.neuprint as nvneu
    from matplotlib import pyplot as plt


    test_neuron = nvneu.fetch_skeletons(722817260)


    interactive_dendrogram(test_neuron, prog = 'dot', inscreen = True)


    """

    z = check_valid_neuron_input(z)

    if heal_neuron:
        z = navis.heal_fragmented_neuron(z)

    valid_progs = ["neato", "dot"]
    if prog not in valid_progs:
        raise ValueError("Unknown program parameter!")

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

    if plot_nodes:

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
    else:

        node_trace = go.Scatter()

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
        text="Soma, node:{}".format(z.soma),
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

    elif isinstance(highlight_connectors, dict):

        HC_nodes = []
        HC_color = []

        for i in list(highlight_connectors.keys()):

            HC_nodes.append(
                z.connectors[z.connectors.connector_id == i].node_id.values[0]
            )
            HC_color.append(
                "rgb({}, {}, {})".format(
                    int(highlight_connectors[i][0]),
                    int(highlight_connectors[i][1]),
                    int(highlight_connectors[i][2]),
                )
            )

            HC_x = []
            HC_y = []

            HC_info = []

            for node in g.nodes():

                for tn in HC_nodes:

                    if node == tn:

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
                marker=dict(size=12.5, color=HC_color),
            )

    # Highlight the nodes that are in a particular volume

    if in_volume is None:

        in_volume_trace = go.Scatter()

    elif in_volume is not None:

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

    if inscreen is True:
        print("Finished in {} seconds".format(time.time() - start))
        return iplot(fig)

    else:
        print("Finished in {} seconds".format(time.time() - start))
        return plot(fig, filename=filename)
