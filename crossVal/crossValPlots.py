import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

#enable seaborn defaults for matplotlib
sns.set()

import matplotlib

# change matplotlib ticksize settings
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

# function for reading medium-dimensional data, projecting onto a 2D space, and then plotting
# using a third dimension (here loss, or MSE) for coloring the plotted points
def plotErrorGrad(path, proj_func='mean'):
    for file in os.listdir(path):
        if file.endswith('.csv'):
            # get the input features, predicted output, and true (labeled) output
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

    # make sure the lengths of these match
    assert len(feats)==len(preds)==len(truths)

    # calculate MSE between the true labels and the predictions
    MSEs = [np.mean(np.square(truth-pred)) for truth, pred in zip(truths, preds)]

    # project features onto 2D space
    # makes assumptions about which columns should be mapped together
    # will need to be changed based on how input data is organized
    # here for example, the first 4 cols are metamaterial unit cell heights, second 4 are radii
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
    # change to np array so we can slice as we please
    feats_prj = np.array(feats_prj)
    # select the points which are unique
    unique_pnts = np.unique(feats_prj, axis=0)

    # calculate the mean MSE value for overlapping points
    # in my case the geometries where sampled on a grid, so many overlapped
    # when projecting with the max function
    if proj_func=='max':
        mse_agg = []
        for pnt in unique_pnts:
            mse_holder = []
            for feat, mse in zip(feats_prj, MSEs):
                # if the projected input vectors (here geometries) are the same, then grab all the mse values
                # so that we can average later
                if np.array_equal(feat, pnt):
                    mse_holder.append(mse)
            # calculate mean
            mse_agg.append(np.mean(mse_holder))

        f, ax = plt.subplots()
        # normalize the error values to the maximum and print
        colors = [perc for perc in mse_agg/max(mse_agg)]
        print('Projection is with max function, MSE normalized to {}'.format(max(mse_agg)))
        # plot with nice plasma color
        scatter = ax.scatter(x=unique_pnts[:, 0], y=unique_pnts[:, 1], c=colors, cmap='plasma')
        f.colorbar(scatter)

    # same thing here with min. Could eliminate some of this redundant code later
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

    # if we're taking the mean many points won't overlap, so there's no need to average the MSE of
    # overlapping ones
    elif proj_func=='mean':
        f, ax = plt.subplots()
        colors = [perc for perc in MSEs/max(MSEs)]
        print('Projection is with mean function, MSE normalized to {}'.format(max(MSEs)))
        scatter = ax.scatter(x=feats_prj[:, 0], y=feats_prj[:, 1], c=colors, alpha=.2, cmap='plasma')
        # get mappable for use with colorbar
        f.colorbar(scatter)
        # plot a rectangle with clipped edges to show the boundary of the space
        # this was to show the values of cross-validation set in my case
        ax.add_patch(Rectangle((48, 49.5), width=55-48, height=5, fill=False, linewidth=2,
                               linestyle = 'dashed', edgecolor='black'))

    plt.savefig(os.path.join('plotsOut', 'sp{}_{}.png'.format(proj_func, dir)))


if __name__=="__main__":
    # specify list of directories to pull and plot data from
    dirs = ['20190410_142248']
    for dir in dirs:
        plotErrorGrad(dir, 'mean')

    #show the plots generated
    plt.show()
    print('done plotting')
