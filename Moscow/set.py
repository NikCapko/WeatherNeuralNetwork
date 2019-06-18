import pandas as pd

data = pd.read_csv('Moscow_test.csv', sep=',', error_bad_lines=False, header=5)[::-1]
s = set()
close_price = data.ix[:, 9].tolist()
for i in close_price:
	s.add(i)
for i in s:
    print(i)
#print(s, len(s))
