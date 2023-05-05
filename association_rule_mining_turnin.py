#-------------------------------------------------------------------------
# AUTHOR: Joey Cindass
# FILENAME: association_rule_mining.py
# SPECIFICATION: finding strong rules associated with supermarket products
# FOR: CS 4210- Assignment #5
# TIME SPENT: 3.5 hours
#-----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('retail_dataset.csv', sep=',')
# print(df)

# grabbing unique items and putting them into a set
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

itemset.remove(np.nan)

encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for item in itemset:
        labels[item] = 0
    for element in row:
        if element in itemset:
            labels[element] = 1
    encoded_vals.append(labels)

ohe_df = pd.DataFrame(encoded_vals)

freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)

for i in range(len(rules)):
    antecedents = list(rules.loc[i]['antecedents'])
    consequents = list(rules.loc[i]['consequents'])
    support = rules.loc[i]['support']
    confidence = rules.loc[i]['confidence']

    prior = support / len(encoded_vals)
    gain_in_conf = str(100*((confidence-prior) / prior))

    print("{} --> {}".format(antecedents, consequents))
    print("Support: {}".format(support))
    print("Confidence: {}".format(confidence))
    print("Prior: {}".format(prior)) # how many people ordered beer compared to the whole dataset --> 4 out of 10 ordered beer
    print("Gain in Confidence: {}".format(gain_in_conf))
    print()

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title("Support vs Confidence")
plt.show()




# for i in range(len(rules)):
#     print(rules.iloc[i][:2])
# print(rules.iloc[6][:2])
# print(rules.columns)




# import pandas as pd
#
# mydict = {'Bread': [1, 0, 1], 'Wine': [0, 1, 1], 'Eggs': [1, 1, 1]}
# print(mydict)
# mydf = pd.DataFrame(mydict)
# print(mydf)

