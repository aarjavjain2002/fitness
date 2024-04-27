"""Generate a fake model for bicep curl"""

import pandas as pd
import random
NOISE = 2.5


def generate_test_bicep_curl_model():
    """Outputs the data within the csv file in the data directory."""
    temp = [150 - 1.4*i for i in range(50)]
    temp.append(80)
    temp.extend([80 + 1.4*i for i in range(1, 51)])
    tricep = [60 for i in range(101)]

    data = {"tricep angle" : tricep, "bicep angle" : temp}
    bicep_curl = pd.DataFrame(data = data)
    bicep_curl.to_csv("data/bicep_curl_model.csv")


def generate_test_bicep_curl_user():
    temp = []
    for i in range(50):
        noise = random.uniform(-NOISE, NOISE)
        temp.append(150 - 1.4*i + noise)
    temp.append(80 + random.uniform(-NOISE, NOISE))
    for i in range(50):
        noise = random.uniform(-NOISE, NOISE)
        temp.append(80 + 1.4*i + noise)

    tricep = [60 for i in range(101)]

    data = {"tricep angle" : tricep, "bicep angle" : temp}
    bicep_curl = pd.DataFrame(data = data)
    bicep_curl.to_csv("data/bicep_curl_user.csv")


def main():
    
    generate_test_bicep_curl_model()
    generate_test_bicep_curl_user()

if __name__ == "__main__":
    main()