
import numpy as np
import sklearn.metrics

def classification_report(model, dataset, subset='validation'):
    """Returns a printable classificaion report.

    Arguments:
        dataset: an dataset with each data subset as attributes.
        subset: the dataset subset to use, e.g. `train`, `validation`, or
            `test`.
    """
    return sklearn.metrics.classification_report(
        getattr(dataset, subset).y,
        np.argmax(model.predict(getattr(dataset, subset).x), -1),
        target_names=dataset.classnames,
        digits=4)
