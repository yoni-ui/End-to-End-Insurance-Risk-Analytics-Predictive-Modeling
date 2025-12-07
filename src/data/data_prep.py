import pandas as pd

def main():
    df = pd.read_csv("data/MachineLearningRating_v3.txt", sep="\t")
    df = df.dropna()

    df.to_csv("data/clean_data.csv", index=False)

if __name__ == "__main__":
    main()
