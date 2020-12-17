=========
neuroboom
=========

.. image:: https://img.shields.io/travis/markuspleijzier/neuroboom.svg
        :target: https://travis-ci.org/markuspleijzier/neuroboom

.. image:: https://img.shields.io/pypi/v/neuroboom.svg
        :target: https://pypi.python.org/pypi/neuroboom


.. image:: /docs/source/_static/logos/nb_large_landing.png
        :width: 1200
        :class: with-shadow



Neuroboom is a suite of Python 3 tools for analysing neuron reconstructions within Connectomic initiatives.
This package is currently designed to operate on neuron recontructions within the *CATMAID* or *Neuprint* platforms.

These two different platforms reflect the two available connectomic efforts in the fruit fly, Drosophila melanogaster:

1. the `Full Adult Female Brain (FAFB) <https://www.sciencedirect.com/science/article/pii/S0092867418307876?via%3Dihub>`_ Zheng *et al.*, 2018, *Cell*
2. the `Hemibrain <https://elifesciences.org/articles/57443>`_, Scheffer *et al.*, 2020, *eLife*

Have I seen this before?
----------------------------

Some code incorporated within the neuroboom package has been presented in scientific papers, most notably the **dendrogram** functions:

1. `Integration of Parallel Opposing Memories Underlies Memory Extinction <https://www.sciencedirect.com/science/article/pii/S0092867418310377?via%3Dihub>`_, J. Felsenberg *et al.*, 2018, *Cell*
2. `Input Connectivity Reveals Additional Heterogeneity of Dopaminergic Reinforcement in Drosophila <https://www.cell.com/current-biology/fulltext/S0960-9822(20)30764-8>`_, N. Otto, **MW Pleijzier** *et al.*, 2018, *Current Biology*
3. `Convergence of distinct subpopulations of mechanosensory neurons onto a neural circuit that elicits grooming <https://www.biorxiv.org/content/10.1101/2020.06.08.141341v1>`_, S. Hampel *et al.*, 2020, *biorXiv*

Documentation
-------------

* neuroboom is on `readthedocs! <>`_


Features
--------

* Dendrograms: 2D graph representations of neurons.
  * Available in static renderings for scientific papers / representations
  * Interactive renderings for exploratory analyses

Coming soon:
------------
* synaptic focalisation
* electrotonic modelling
