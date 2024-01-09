import pandas as pd

df = pd.read_csv('data/mushrooms.csv')


def onehot(df):
    X = pd.get_dummies(df)
    return X


if __name__ == "__main__":
    import timeit

    iterations = 1000
    seconds = timeit.timeit(lambda: onehot(df), number=iterations) / iterations
    print(f"{seconds * 1000} ms")
