import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

sns.set()

import matplotlib
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

# 20190327_180830 is the corner constrained hypergrid shape01 (2 vals excluded)
# 20190327_200126 is the corner constrained hypergrid shape02 (3 vals excluded)
# 20190328_150736 is the corner constrained hypergrid shape03 (4 vals excluded)

def plotErrorGrad(path, proj_func='mean'):
    for file in os.listdir(path):
        if file.endswith('.csv'):
            if file[:9]=='test_feat':
                feats = pd.read_csv(os.path.join(path, file), header=None, sep=' ').values
                print('got feats for path {}'.format(path))
            elif file[:9]=='test_pred':
                preds = pd.read_csv(os.path.join(path, file), header=None, sep=' ').values
                print('got test for path {}'.format(path))
            elif file[:10]=='test_truth':
                truths = pd.read_csv(os.path.join(path, file), header=None, sep=' ').values
                print('got truths for path {}'.format(path))
    print(' ')
    assert len(feats)==len(preds)==len(truths)

    MSEs = [np.mean(np.square(truth-pred)) for truth, pred in zip(truths, preds)]

    # project features onto 2D space
    feats_prj = []
    for feat in feats[:, :8]:
        if proj_func=='mean':
            feats_prj.append([np.mean(feat[:4]), np.mean(feat[4:8])])
        if proj_func=='max':
            feats_prj.append([np.max(feat[:4]), np.max(feat[4:8])])
        if proj_func=='min':
            feats_prj.append([np.min(feat[:4]), np.min(feat[4:8])])
        elif proj_func not in ['mean', 'max', 'min']:
            print('proj_func was not a valid choice of mean, max, or min')
            return
    feats_prj = np.array(feats_prj)
    unique_pnts = np.unique(feats_prj, axis=0)

    # calculate the mean MSE value for overlapping points
    if proj_func=='max':
        mse_agg = []
        for pnt in unique_pnts:
            mse_holder = []
            for feat, mse in zip(feats_prj, MSEs):
                if np.array_equal(feat, pnt):
                    mse_holder.append(mse)
            mse_agg.append(np.mean(mse_holder))

        f, ax = plt.subplots()
        colors = [perc for perc in mse_agg/max(mse_agg)]
        print('Projection is with max function, MSE normalized to {}'.format(max(mse_agg)))
        scatter = ax.scatter(x=unique_pnts[:, 0], y=unique_pnts[:, 1], c=colors, cmap='plasma')
        f.colorbar(scatter)

    if proj_func=='min':
        mse_agg = []
        for pnt in unique_pnts:
            mse_holder = []
            for feat, mse in zip(feats_prj, MSEs):
                if np.array_equal(feat, pnt):
                    mse_holder.append(mse)
            mse_agg.append(np.mean(mse_holder))

        f, ax = plt.subplots()
        colors = [perc for perc in mse_agg/max(mse_agg)]
        print('Projection is with min function, MSE normalized to {}'.format(max(mse_agg)))
        scatter = ax.scatter(x=unique_pnts[:, 0], y=unique_pnts[:, 1], c=colors, cmap='plasma')
        f.colorbar(scatter)

    elif proj_func=='mean':
        f, ax = plt.subplots()
        colors = [perc for perc in MSEs/max(MSEs)]
        print('Projection is with meanfunction, MSE normalized to {}'.format(max(MSEs)))
        scatter = ax.scatter(x=feats_prj[:, 0], y=feats_prj[:, 1], c=colors, alpha=.2, cmap='plasma')
        # get mappable for use with colorbar
        f.colorbar(scatter)
        # plot a rectangle with clipped edges to show the boundary of the space
        ax.add_patch(Rectangle((48, 49.5), width=55-48, height=5, fill=False, linewidth=2,
                               linestyle = 'dashed', edgecolor='black'))

    plt.savefig('sp{}_{}.png'.format(proj_func, dir))


if __name__=="__main__":
    # dirs = ['20190410_135122', '20190410_140657', '20190410_142248']
    # 20190422_153723 is the bottom left corner cross-val experiment
    dirs = ['20190410_142248']
    for dir in dirs:
        plotErrorGrad(dir, 'mean')
    plt.show()
    print('done plotting')