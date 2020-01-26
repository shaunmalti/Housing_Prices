# Housing_Prices

Heavily inspired from kaggle notebooks, namely:

https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
And
https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard#Data-Processing

Other to these, inflation was added using CPI index seen in `adjustPrice()` method:

<pre><code>def adjustPrice(data): 
    map = {
        2010: 218.056,
        2009: 214.537,
        2008: 215.303,
        2007: 207.342,
        2006: 201.6
    }
    data['AdjustedPrice'] = data.apply(lambda x: x['SalePrice'] * (map[2010]/map[x['YrSold']]), axis=1)
    return data.drop(['SalePrice'], axis=1)
</code></pre>

Map contains CPI index used for inflation calculation.