import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# The listed features in the index variables were selected using a dimensionality reduction algorithm based on feature
# importance. You can adopt any algorithm to reduce the dimension of your dataset or choose to all.

index = pd.Index(['fwd_packet_length_max', 'flow_packet_length_max', 'bwd_packet_length_max', 'down_up_ratio',
                  'flow_packet_length_mean', 'bwd_packet_length_mean', 'fwd_total_length_packets',
                  'flow_packet_length_std',
                  'bwd_total_length_packets', 'fwd_ack_flags', 'fwd_packet_length_mean', 'flow_packet_length_min',
                  'fwd_packet_length_min', 'fwd_avg_num_packets', 'bwd_packet_length_std', 'fwd_num_packets',
                  'flow_avg_num_packets', 'flow_total_length_packets', 'label'])

base = '../classifier/final-experiment/data/selected/'


def get_train_data(data_name):
    """
    This returns any of three training dataset or a global dataset that contains samples from each datasets.
    :param data_name: this is the name of the dataset to be returned. It takes either 'caida', 'dos', 'ids' or 'global'
    as input.
    :return: a tuple of the training samples and the classification is returned
    """
    df = None
    if data_name == 'caida':
        df = pd.read_csv(base + 'caida_train_selected.csv')
    elif data_name == 'dos':
        df = pd.read_csv(base + 'cicdos2017_train_selected.csv')
    elif data_name == 'ids':
        df = pd.read_csv(base + 'ids2017_train_selected.csv')
    elif data_name == 'global':
        df1 = pd.read_csv(base + 'caida_train_selected.csv')
        df2 = pd.read_csv(base + 'cicdos2017_train_selected.csv')
        df3 = pd.read_csv(base + 'ids2017_train_selected.csv')
        g1_sample = df1.groupby('label', group_keys=False).apply(lambda x: x.sample(15000))
        g2_sample = df2.groupby('label', group_keys=False).apply(lambda x: x.sample(15000))
        g3_sample = df3.groupby('label', group_keys=False).apply(lambda x: x.sample(15000))
        df = pd.concat([g1_sample, g2_sample, g3_sample], axis=0, ignore_index=True)
        df = shuffle(df, random_state=10)

    x_train = df[index].drop(['label'], axis=1)
    y_train = df['label']
    y_train = LabelEncoder().fit_transform(y_train)
    return x_train, y_train


def get_test_data(data_name):
    """
    This returns any of three testing dataset or a global dataset that contains samples from each datasets.
    :param data_name: this is the name of the dataset to be returned. It takes either 'caida', 'dos', 'ids' or 'global'
    as input.
    :return: a tuple of the testing samples and the classification is returned
    """
    df = None
    if data_name == 'caida':
        df = pd.read_csv(base + 'caida_test_selected.csv', low_memory=False)
    elif data_name == 'dos':
        df = pd.read_csv(base + 'cicdos2017_test_selected.csv', low_memory=False)
    elif data_name == 'ids':
        df = pd.read_csv(base + 'ids2017_test_selected.csv', low_memory=False)
    elif data_name == 'global':
        df1 = pd.read_csv(base + 'caida_test_selected.csv', low_memory=False)
        df2 = pd.read_csv(base + 'cicdos2017_test_selected.csv', low_memory=False)
        df3 = pd.read_csv(base + 'ids2017_test_selected.csv', low_memory=False)
        g1_sample = df1.groupby('label', group_keys=False).apply(lambda x: x.sample(10000))
        g2_sample = df2.groupby('label', group_keys=False).apply(lambda x: x.sample(10000))
        g3_sample = df3.groupby('label', group_keys=False).apply(lambda x: x.sample(10000))
        df = pd.concat([g1_sample, g2_sample, g3_sample], axis=0, ignore_index=True)
        df = shuffle(df, random_state=10)

    if not data_name == 'global':
        df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(10000))
    x_test = df[index].drop(['label'], axis=1)
    y_test = df['label']
    y_test = LabelEncoder().fit_transform(y_test)
    return x_test, y_test
