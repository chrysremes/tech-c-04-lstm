import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def data_plot_multindex(df, n_tickers):
    # Plot all line charts
    df_plot = df.copy()

    ncols = 2
    nrows = int(round(df_plot.shape[1] / (ncols*n_tickers), 0))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
    for i, ax in enumerate(fig.axes):
        for tk in range(n_tickers):
            sns.lineplot(data=df_plot.iloc[:, tk+i*n_tickers], ax=ax)
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    # plt.show()
    plt.show(block=False)

def data_plot(df):
    # Plot line charts
    df_plot = df.copy()

    ncols = 2
    nrows = int(round(df_plot.shape[1] / ncols, 0))

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(14, 7))
    for i, ax in enumerate(fig.axes):
        sns.lineplot(data=df_plot.iloc[:, i], ax=ax)
        ax.tick_params(axis="x", rotation=30, labelsize=10, length=0)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    plt.show(block=False)