# | Copyright Matthias Mayr October 2021
# |
# | Code repository: https://github.com/matthias-mayr/behavior-tree-policy-learning
# | Preprint: https://arxiv.org/abs/2109.13050
# |
# | This software is governed by the CeCILL-C license under French law and
# | abiding by the rules of distribution of free software.  You can  use,
# | modify and/ or redistribute the software under the terms of the CeCILL-C
# | license as circulated by CEA, CNRS and INRIA at the following URL
# | "http://www.cecill.info".
# |
# | As a counterpart to the access to the source code and  rights to copy,
# | modify and redistribute granted by the license, users are provided only
# | with a limited warranty  and the software's author,  the holder of the
# | economic rights,  and the successive licensors  have only  limited
# | liability.
# |
# | In this respect, the user's attention is drawn to the risks associated
# | with loading,  using,  modifying and/or developing or reproducing the
# | software by the user in light of its specific status of free software,
# | that may mean  that it is complicated to manipulate,  and  that  also
# | therefore means  that it is reserved for developers  and  experienced
# | professionals having in-depth computer knowledge. Users are therefore
# | encouraged to load and test the software's suitability as regards their
# | requirements in conditions enabling the security of their systems and/or
# | data to be ensured and,  more generally, to use and operate it in the
# | same conditions as regards security.
# |
# | The fact that you are presently reading this means that you have had
# | knowledge of the CeCILL-C license and that you accept its terms.
# |
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd

colors = ['#ff7f0e', '#5c88b4', '#2ca02c', '#d62728',
        '#bcbd22', '#8c564b', '#e377c2']

real_success_data = [100, 53.3, 100, 100, 100, 100]
sim_success_data = [100, 100, 0, 100, 100, 100, 100, 0]
no_search_success_data = [20, 20, 20, 20, 20]
random_search_success_data = [46, 67, 46, 67, 53, 33, 53, 33, 0, 100, 100, 100, 53,
                              33, 100]

names = ['No Search', 'Random\nParameters', 'Simulation\nParameters', 'Real\nParameters']
legend_handles = list()
for it in range(len(names)):
    legend_handles.append(mlines.Line2D([], [], color=colors[it], label=names[it]))

fig1, ax = plt.subplots()
ax.set_title('Insertion Rates for the Peg Task')
data = [no_search_success_data, random_search_success_data, sim_success_data, real_success_data]
pos = np.array(range(len(data))) + 1
bp = ax.boxplot(data, sym='o', patch_artist=True, positions=pos, notch=0)
# ax.legend(loc='upper center', bbox_to_anchor=(-0, 0.0),
#   ncol=3, fancybox=True, shadow=True, handles=legend_handles)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

labels = [item.get_text() for item in ax.get_xticklabels()]
for i in range(len(names)):
    labels[i] = names[i]

ax.set_xticklabels(labels)
ax.set_ylabel("Successful insertions / %")

plt.savefig('peg_success_rates.png', dpi=300)
plt.savefig('peg_success_rates.svg')