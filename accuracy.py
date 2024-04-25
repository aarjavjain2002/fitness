"""Calculate the accuracy of a rep."""
import pandas as pd
import math
DECAY_RATE = 0.03193 # (-math.log(60/100)) / 16

# We assume a residual of 16 corresponds to a 60% accuracy

def get_accuracy(userdf: pd.DataFrame, modeldf: pd.DataFrame) -> float:
    """Retrieve the accuracy by comparing the user angles with the model angles."""
    num_rows, num_columns = userdf.shape
    res = 0

    for j in range(num_columns):
        for i in range(num_rows):
            res += ((userdf.iloc[i, j] - modeldf.iloc[i, j])**2)
    res = res*(1/(2*num_rows))
    return exp_decay(res)

def exp_decay(res: float) -> float:
    return 100 * math.exp(-DECAY_RATE*res)

def main():
    modeldf = pd.read_csv("data/bicep_curl_model.csv")
    userdf = pd.read_csv("data/bicep_curl_user.csv")
    print(get_accuracy(userdf, modeldf))

if __name__ == "__main__":
    main()