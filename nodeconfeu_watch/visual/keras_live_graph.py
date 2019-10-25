
import tensorflow.keras as keras

import uuid
import json
import os.path as path

try:
    import IPython
except:
    pass

thisdir = path.dirname(path.realpath(__file__))

class KerasLivePlot(keras.callbacks.Callback):
    """Makes an live-updating graph of the accuracy and loss.

    Arguments:
        plot_interval: backlog `plot_interval` epochs and then update the plot.
        width: the width of the plot in pixels.
        height: the height of the plot in pixels.
    """
    def __init__(self, plot_interval=5, width=600, height=400):
        super().__init__()
        self.width = width
        self.height = height
        self.plot_interval = plot_interval

    def on_train_begin(self, logs=None):
        self.graph_id = uuid.uuid4()
        self.epochs = self.params['epochs']
        self.backlog = []

        with open(path.join(thisdir, 'keras_live_graph.css')) as css_fp, \
             open(path.join(thisdir, 'keras_live_graph.js')) as js_fp:
            display(IPython.display.HTML((
                f'<style>{css_fp.read()}</style>'
                f'<script src="https://d3js.org/d3.v5.min.js"></script>'
                f'<script>{js_fp.read()}</script>'
                f'<svg id="{self.graph_id}"></svg>'
                f'<script>'
                f' window.setupLearningGraph({{'
                f'  id: "{self.graph_id}",'
                f'  epochs: {self.epochs},'
                f'  height: {self.height},'
                f'  width: {self.width}'
                f' }});'
                f'</script>'
            )))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = { k: float(v) for k, v in logs.items() }
        row['epoch'] = epoch
        self.backlog.append(row)

        if (epoch % self.plot_interval == 0 or epoch == self.epochs - 1):
          display(IPython.display.Javascript(
            f'window.appendLearningGraphData({json.dumps(self.backlog)})'
          ))
          self.backlog = []
