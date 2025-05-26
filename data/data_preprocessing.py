import matplotlib.pyplot as plt
import numpy as np
import deeptrack as dt
import torch
from torch.utils.data import DataLoader

def retrive_and_extract_relevant_data_values(df, feature_cols, chosen_ticker, no_plotting=False):
    df = df[df["Ticker"] == chosen_ticker]
    df = df.iloc[20:]

    prices = df["Close"].values    
    market_cap = df["MarketCap"].values
    rsi = df["RSI14"].values

    dataset = df[feature_cols]
    print('Shape dataset', dataset.shape)

    values = dataset.values

    means = dataset.mean(axis=0)
    stds = dataset.std(axis=0)

    print('means: ',means)
    print('stds: ', stds)

    volvo_dataset_norm = (dataset - means)/stds 
    volvo_values_norm = volvo_dataset_norm.values

    norm_close = volvo_values_norm[:,0]
    norm_mark_cap = volvo_values_norm[:,2]
    norm_rsi = volvo_values_norm[:, 10]

    if not no_plotting:
        plt.plot(prices, label= 'close')
        plt.plot(market_cap, label='market cap')
        plt.plot(rsi, label='rsi')
        plt.legend()
        plt.title(chosen_ticker)
        plt.show()

        plt.plot(norm_close, label= 'close')
        plt.plot(norm_mark_cap, label='market cap')
        plt.plot(norm_rsi, label='rsi')
        plt.legend()
        plt.title(chosen_ticker)

        plt.show()

        for i in range(len(feature_cols)):
            plt.plot(volvo_values_norm[:,i], label= feature_cols[i])

        plt.legend()
        plt.title(chosen_ticker)

    return values, dataset

def generate_loaders_and_benchmark(raw_numpy_data, window_step_length, num_test_points, window_size, horizon, target_idx, minimal_plotting=False):

    data = raw_numpy_data[:-num_test_points]
    test_real_data = raw_numpy_data[-num_test_points:]
    #print(data[-1])
    #print(test_real_data[0])
    if not minimal_plotting:
        plt.plot(test_real_data[:,0])
        plt.title('Test data plot')
        plt.show()

    n_samples = data.shape[0]
    past_seq = window_size
    lag = horizon

    in_sequences, targets = [], []
    for i in range(past_seq, n_samples - lag, window_step_length):
        in_sequences.append(data[i - past_seq:i, :])
        targets.append(data[i + lag:i + lag + 1, target_idx])
    in_sequences, targets = np.asarray(in_sequences), np.asarray(targets)
    #print(in_sequences[10][horizon,target_idx])
    #print(targets[9])

    print('in sequences shape: ',in_sequences.shape)
    print('targets shape: ',targets.shape)

    sources = dt.sources.Source(inputs=in_sequences, targets=targets)
    train_sources, val_sources = dt.sources.random_split(sources, [0.8, 0.2])

    

    train_mean = np.mean([src["inputs"] for src in train_sources], axis=(0, 1))
    train_std = np.std([src["inputs"] for src in train_sources], axis=(0, 1))

    inputs_pipeline = (dt.Value(sources.inputs - train_mean) / train_std
                       >> dt.pytorch.ToTensor(dtype=torch.float))
    targets_pipeline = (dt.Value(sources.targets - train_mean[target_idx]) 
                        / train_std[target_idx])

    train_dataset = dt.pytorch.Dataset(inputs_pipeline & targets_pipeline,
                                       inputs=train_sources)
    val_dataset = dt.pytorch.Dataset(inputs_pipeline & targets_pipeline,
                                     inputs=val_sources)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print('train loader length: ', len(train_loader))
    print('val loader length: ', len(val_loader))
    #for x, y in train_loader:
    #    print(y[0])
    #    print(x[0])
    target_data = data[:, target_idx]
    indices = np.arange(0, len(target_data)-lag)
    diffs = np.abs(target_data[indices + lag] - target_data[indices])
    benchmark_raw = np.mean(diffs)
    benchmark = benchmark_raw / train_std[target_idx]


    print(f"Benchmark Celsius: {benchmark_raw}")
    print(f"Normalized Benchmark: {benchmark}")

    return train_loader, val_loader, benchmark, train_mean, train_std

def plot_correlation_matrix(dataframe, feature_cols):
    corr = dataframe.corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, aspect='auto')
    plt.colorbar(im)
    plt.xticks(ticks=np.arange(len(feature_cols)), labels=feature_cols, rotation=90)
    plt.yticks(ticks=np.arange(len(feature_cols)), labels=feature_cols)
    plt.title("Correlation Matrix of Volvo Features")
    plt.tight_layout()
    plt.show()