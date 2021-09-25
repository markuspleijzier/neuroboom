from typing import Optional, Union, Any

import seaborn as sns
import random


def define_color_palette(cmap, n_clusters, shuffle = False):

    pal = sns.color_palette(cmap, n_clusters)

    if shuffle:

        random.shuffle(pal)

    label_to_col = dict(zip([i for i in range(0, n_clusters)], pal))
    label_to_col[-1] = (0.0, 0.0, 0.0)

    return(label_to_col)
