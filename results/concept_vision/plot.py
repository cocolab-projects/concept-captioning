import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def plot(split):
    results = pd.read_csv('/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/results/concept_vision/results_{}.tsv'.format(split), sep='\t')
    cat_mapping = {
        0: 'Single Feature',
        1: 'Conjunction',
        2: 'Disjunction',
        3: 'Conjunctive Disjunction',
        4: 'Disjunctive Conjunction',
    }
    rule_types = []
    teacher_student_accuracies = []
    teacher_y_hat_accuracies = []
    groups = []
    for name, group in results.groupby(['gameid', 'rule_idx']):
        print ('Processing: {}'.format(name))
        groups.append(name)
        teacher_y_hat_agreement = (group['y_hat'] == group['teacher']).tolist()
        teacher_y_hat_acc = (np.sum(np.array(teacher_y_hat_agreement)) * 1.0) / len(teacher_y_hat_agreement)
        teacher_y_hat_accuracies.append(teacher_y_hat_acc)

        teacher_student_agreement = (group['student'] == group['teacher']).tolist()
        teacher_student_acc = (np.sum(np.array(teacher_student_agreement)) * 1.0) / len(teacher_student_agreement)
        teacher_student_accuracies.append(teacher_student_acc)

        rule_types.append(cat_mapping[int(int(name[1]) / 10)])

    # for i, group in enumerate(groups):
    #     if teacher_student_accuracies[i] == 1.0:
    #         print (group, teacher_student_accuracies[i], teacher_y_hat_accuracies[i])

    df = pd.DataFrame(dict(t_s_acc=teacher_student_accuracies, t_m_acc = teacher_y_hat_accuracies, rule=rule_types))
    lm = sns.lmplot('t_s_acc', 't_m_acc', data=df, hue='rule', fit_reg=True)
    axes = lm.axes
    axes[0,0].set_ylim(0,1.1)
    axes[0,0].set_xlim(0,1.1)
    axes[0,0].set_xticks(np.arange(0.0, 1.1, 0.1))
    axes[0,0].set_yticks(np.arange(0.0, 1.1, 0.1)) 

    r, p = stats.pearsonr(teacher_student_accuracies, teacher_y_hat_accuracies)
    axes[0,0].text(0.2, 0.2, "r = {:.2f}".format(r, horizontalalignment='left', size='medium', color='black', weight='semibold'))
    axes[0,0].text(0.2, 0.15, "p = {:.2f}".format(p, horizontalalignment='left', size='medium', color='black', weight='semibold'))

    plt.xlabel('Teacher-Student Agreement')
    plt.ylabel('Teacher-Model Agreement')
    plt.savefig('/Users/Sahil/Desktop/Research/cultural_ratchet/cultural-ratchet-lfl/results/concept_vision/{}.png'.format(split))

plot('val')


