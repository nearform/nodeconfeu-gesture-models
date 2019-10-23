
import tensorflow as tf
import tensorflow.keras as keras

from nodeconfeu_watch.reader import AccelerationReader

conor_dataset = AccelerationReader('./data/conor-v2', test_ratio=0.2, validation_ratio=0.2,
                              classnames=['swiperight', 'swipeleft', 'upup', 'waggle', 'clap2'])
conor_dataset.savecsv('./exports/conor.csv')

conor_dataset = AccelerationReader('./data/james-v2', test_ratio=0.2, validation_ratio=0.2,
                              classnames=['swiperight', 'swipeleft', 'upup', 'waggle', 'clap2'])
conor_dataset.savecsv('./exports/james.csv')
