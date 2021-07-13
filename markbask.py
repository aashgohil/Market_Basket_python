#Git Change
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


df1 = pd.read_csv('/home/gamooga/data/market basket/BreadBasket.csv')
#print (df1.head())

df = df1.groupby(['Transaction','Item']).size().reset_index(name='Count')

basket = (df.groupby(['Transaction','Item'])['Count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Transaction'))


#print (basket)

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
#print (frequent_itemsets)
rules = association_rules(frequent_itemsets, metric = 'lift')
rules.sort_values('lift', ascending = False, inplace = True)
print (rules.head(10))

