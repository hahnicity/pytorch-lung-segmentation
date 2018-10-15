import pandas as pd
import matplotlib.pyplot as plt

losses = []
with open("log.txt") as lossFile:
    for line in lossFile:
        losses.append(float(line))
        print(line)
losses = pd.DataFrame(losses)
rolling_mean = losses.rolling(window=10).mean()

itr = range(len(rolling_mean))

plt.plot(itr, rolling_mean)
plt.show()
