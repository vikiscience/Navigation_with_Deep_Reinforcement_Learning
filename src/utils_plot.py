import const

import pandas as pd
from matplotlib import pyplot as plt


def plot_history_rolling_mean(hist, N=const.rolling_mean_N):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # prepare data
    x = pd.Series(hist)
    y = x.rolling(window=N).mean().iloc[N - 1:]

    # plot
    plt.plot(hist, c='darkorchid', marker='.', markevery=[-1])
    plt.plot(y, c='blue', marker='.', markevery=[-1])

    # annotate last_rolling point
    last_points = [(len(hist) - 1, hist[-1]), (len(hist) - 1, y.iloc[-1])]
    for (i, j) in last_points:
        ax.annotate('{:.2f}'.format(j), xy=(i, j), xytext=(i + 0.1, j))

    plt.xlabel('Episodes')
    plt.ylabel('Score (Sum of Rewards)')
    plt.title('Online Performance')
    plt.legend(['score', 'rolling_score (N={})'.format(N)], loc='best')
    plt.savefig(const.file_path_img_score)
    plt.close()
