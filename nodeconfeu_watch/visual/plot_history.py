
import numpy as np
import pandas as pd

def plot_history(history, autoshow=True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    df = pd.DataFrame(history.history)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'epoch'}, inplace=True)

    df = pd.melt(df, id_vars=['epoch'])
    df.loc[:,'loss.type'] = df.loc[:,'variable'].map({
        'loss': 'cross entropy',
        'sparse_categorical_accuracy': 'accuracy',
        'val_loss': 'cross entropy',
        'val_sparse_categorical_accuracy': 'accuracy'
    })
    df.loc[:,'dataset'] = df.loc[:,'variable'].map({
        'loss': 'training',
        'sparse_categorical_accuracy': 'training',
        'val_loss': 'validation',
        'val_sparse_categorical_accuracy': 'validation'
    })

    g = sns.relplot(x="epoch", y="value", row="loss.type", hue="dataset",
                    facet_kws=dict(sharey=False),
                    height=3, aspect=2.5,
                    kind="line", legend="full", data=df)
    g.set(ylim=(0, None))

    if autoshow:
        plt.show()
