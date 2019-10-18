
import numpy as np
import sklearn.metrics

def classification_report(model, dataset, subset='validation'):
    return sklearn.metrics.classification_report(
        getattr(dataset, subset).y,
        np.argmax(model.predict(getattr(dataset, subset).x), -1),
        target_names=dataset.classnames,
        digits=4)
