"""For plotting trajectories.
Adapted from https://github.com/drphilmarshall/SpaceWarps with minor adjustments

"""

def trajectory_plot(swap, path=None, subjects=200, logy=True):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np

    # get subjects
    # max_seen is set by subject with max number of classifications
    max_seen = 1
    subjects_final = []
    subjects=[20865737,20865736,20865735,20865734,20865733,20865732,20865731,20865730,20865729,20865728]
    if type(subjects) == int:
        # draw random numbers
        while len(subjects_final) < subjects:
            # note that there can be duplication.
            indx = np.random.choice(len(swap.subjects.keys()))
            subject = swap.subjects[swap.subjects.keys()[indx]]

            # if subject.seen < 3:
            #     continue

            subjects_final.append(subject)
            if subject.seen > max_seen:
                max_seen = subject.seen
    else:
        # assume the subjects are their IDs, so get it from there
        for i in subjects:
            subject = swap.subjects[i]
            subjects_final.append(subject)
            if subject.seen > max_seen:
                max_seen = subject.seen
    subjects = subjects_final

    fig, ax = plt.subplots(figsize=(5,5), dpi=300)

    #####
    # pretty up the figure
    #####
    color_test = 'gray'
    color_bogus = 'red'
    color_real = 'blue'
    colors = [color_bogus, color_real, color_test]

    linewidth_test = 1.0
    linewidth_bogus = 1.5
    linewidth_real = 1.5
    linewidths = [linewidth_bogus, linewidth_real, linewidth_test]

    size_test = 20
    size_bogus = 40
    size_real = 40
    sizes = [size_bogus, size_real, size_test]

    alpha_test = 0.1
    alpha_bogus = 0.3
    alpha_real = 0.3
    alphas = [alpha_bogus, alpha_real, alpha_test]


    # axes and labels
    p_min = 5e-8
    p_max = 1
    ax.set_xlim(p_min, p_max)
    ax.set_xscale('log')
    ax.set_ylim(max_seen, 1)
    if logy:
        ax.set_yscale('log')

    if swap.thresholds is None:
        p_bogus = 1.1 * p_min
        p_real = 0.9 * p_max
    else:
        p_bogus = swap.thresholds.thresholds[0]
        p_real = swap.thresholds.thresholds[1]
        p_bogus = 1.e-7
        p_real = 0.95
    ax.axvline(x=subjects[0].p0, color=color_test, linestyle='dotted')
    ax.axvline(x=p_bogus, color=color_bogus, linestyle='dotted')
    ax.axvline(x=p_real, color=color_real, linestyle='dotted')

    ax.set_xlabel('Posterior Probability Pr(LENS|d)')
    ax.set_ylabel('No. of Classifications')

    # plot history trajectories
    for subject in subjects:
        score, history = subject.update_score(history=True)
        # clip history
        history = np.array(history)
        history = np.where(history < p_min, p_min, history)
        history = np.where(history > p_max, p_max, history)
        # trajectory
        y = np.arange(len(history)) + 1

        # add initial y=0.5, history=subject.p0
        history = np.append(subject.p0, history)
        y = np.append(0.5, y)

        ax.plot(history, y, color=colors[subject.gold], alpha=alphas[subject.gold], linewidth=linewidths[subject.gold], linestyle='-')

        # a point at the end
        ax.scatter(history[-1:], y[-1:], s=sizes[subject.gold], edgecolors=colors[subject.gold], facecolors=colors[subject.gold], alpha=1.0)

    # add legend
    patches = []
    for color, alpha, label in zip(colors, alphas, ['Bogus', 'Real', 'Test']):
        patch = mpatches.Patch(color=color, alpha=alpha, label=label)
        patches.append(patch)
    ax.legend(handles=patches, loc='lower center', prop={'size':10}, framealpha=1.0)

    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=300)

    return fig
