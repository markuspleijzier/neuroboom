=====
Usage
=====

Start by importing neuroboom.

.. code-block:: python

    import neuroboom

Dendrograms
-----------
Dendrograms are 2D graph representation of 3D reconstructed neurons that preserve the local and global topology of the neuron.

The dendrogram functions within neuroboom recover treenode and connector information of a presented neuron and renders it as a NetworkX graph / network.

* **Treenodes** are the nodes used to represent the neuron's (skeletonised) morphology
* **Connectors** are synapses (*N.B.* a single connector/synapse can have multiple connections)

As every connector (both presynaptic & postsynaptic) is associated with a specific treenode id (both on the presynaptic and postsynaptic neuron),
dendrograms allows one to visualise which treenodes of a neuron are connected by specific neurons/ specific connectors.

**This allows one to visualise the placement of synapses from specific neurons or lineage.**

There are two options for dendrograms:

1. Static

Static dendrograms are Matplotlib plots containing the 2D representation.
These are best for scientific paper figures / presentations.

2. Interactive

Interactive dendrograms use plotly to present the 2D representation.
This allows for node / connector identification within the dendrogram.
These are best for *in progress* analyses, however could also be used for
scientific presentations.

For static dendrograms execute:

.. code-block:: python

    from neuroboom.dendrogram import plot_dendrogram

OR for interactive dendrograms:

.. code-block:: python

    from neuroboom.dendrogram import interactive_dendrogram
