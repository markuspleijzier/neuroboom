============
Installation
============

neuroboom can be installed in two main ways:

1. A quick installation using pip
2. Building from source


Quick Installation
------------------

**neuroboom** requires **Python 3.7**

If you don't already have it, get the Python package manager `PIP <https://pip.pypa.io/en/stable/installing/>`_.

To install the most recent version available on PyPI, run

::

  $ pip install neuroboom


in the command line.

To get the most recent development version,
from `Github <https:://github.com/markuspleijzier/neuroboom>`_ use:

    $ pip install git::git://github.com/markuspleijzier/neuroboom@master


Instead of using PIP to install from Github, you can also install manually:

    1. Download the source (e.g a ``tar.gz`` file from
       https://github.com/markuspleijzier/neuroboom/tree/master/dist)

    2. Unpack and change directory to the source directory
       (the one with ``setup.py``).

    3. Run ``python setup.py install`` to build and install


**Important Dependency Configuration**
--------------------------------------
Installing the graph rendering engines is a little more complicated and one cannot rely on automatic installation (yet).

Installing the graph drawing engines is a little more complicated and one cannot rely on automatic installation (yet).

To install Graphviz, one must first install homebrew:

**Homebrew ( see https://brew.sh/ )**

::

    $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Simply paste the above into the terminal.

Then to install Graphviz:

**Graphviz (see http://www.graphviz.org/)**

::
    $ brew install Graphviz

**PyGraphviz ( see http://pygraphviz.github.io/)**

To install PyGraphviz, we need to direct pip (this time using pip3) to where Graphviz is located.
To do this, simply paste this snippet into the terminal:

::
    pip3 install --install-option="--include-path=/usr/local/include/" --install-option="--library-path=/usr/local/lib/" pygraphviz


Once these are installed, import them into your environment.
Graphviz is a dependency of NetworkX, so there is no need to import Graphviz/PyGraphviz.


Other (non-issue) Dependencies
-------------------------------
`pymaid <https://pymaid.readthedocs.io/en/latest/source/install.html/>`_

`navis  <https://navis.readthedocs.io/en/latest/index.html>`_

`matplotlib <http://matplotlib.sourceforge.net/>`_

`numpy <http://www.numpy.org/>`_

`networkx <https://networkx.github.io>`_

`plotly <https://plot.ly/python/getting-started/>`_

`pandas <http://pandas.pydata.org/>`_

`phate <https://github.com/KrishnaswamyLab/PHATE>`_

`seaborn <https://seaborn.pydata.org>`_

`sklearn <https://scikit-learn.org/stable/install.html>`_

`scprep <https://github.com/KrishnaswamyLab/scprep`_

`scipy <http://scipy.org>`_

`tqdm  <https://tqdm.github.io/>`_
