import pandas as pd

def onehot():
    df = pd.read_csv('data/mushrooms.csv')
    X = pd.get_dummies(df)
    return X


if __name__ == "__main__":
    import timeit

    iterations = 1000
    seconds = timeit.timeit(lambda: onehot(), number=iterations) / iterations
    print(f"{seconds * 1000} ms")
