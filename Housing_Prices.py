import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from scipy.special import boxcox1p

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# feature engineering heavily inspired from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python-her-buddies-and-her-interests

def main():
    train = pd.read_csv('~/Repos/Housing_Prices/house-prices-advanced-regression-techniques/train.csv')
    # print(train.head(10))
    # describe(train)
    # correlationMatrix(train)
    # scatter(train)
    # getEmpty(train)

    train = unskewPrice(train)
    train = adjustPrice(train)
    train = fillEmpty(train)
    converNumericaltoCategorical(train)
    train = labelEncode(train)
    train = boxCoxTransform(train)

    elasticReg(train)


def elasticReg(data):
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=.8, random_state=3))

    Y_res = data['AdjustedPrice']
    data = data.drop(['AdjustedPrice'], axis=1)

    ENet.fit(data, Y_res)

    score = crossVal(ENet, data, Y_res)
    print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


def crossVal(model, train, y_res):
    folds = 4
    kf = KFold(folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_res, scoring="neg_mean_squared_error", cv=kf))
    return rmse

def boxCoxTransform(train):
    # get numeric features
    numeric_feats = train.dtypes[train.dtypes != "object"].index
    # check skew
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    lda = 0.1
    for feat in skewed_feats.index:
        train[feat] = boxcox1p(train[feat], lda)
    return train

def unskewPrice(train):
    train["SalePrice"] = np.log1p(train["SalePrice"])
    return train

def converNumericaltoCategorical(data):
    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['OverallCond'] = data['OverallCond'].astype(str)
    data['YrSold'] = data['YrSold'].astype(str)
    data['MoSold'] = data['MoSold'].astype(str)

def labelEncode(data):
    for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType',
                'MSSubClass', 'OverallCond', 'YrSold', 'MoSold', 'LotShape', 'PavedDrive', 'Street', 'CentralAir',
                'Functional', 'MSZoning', 'SaleCondition', 'KitchenQual', 'Electrical', 'HeatingQC',
                'Heating', 'ExterQual', 'ExterCond', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                'HouseStyle', 'BldgType', 'Condition2', 'Condition1', 'Neighborhood', 'LandSlope', 'LotConfig',
                'LandContour', 'Foundation', 'SaleType'):
        lEncoder = LabelEncoder()
        lEncoder.fit(data[col])
        data[col] = lEncoder.transform(data[col])
    return data

def getEmpty(data):
    data_na = (data.isnull().sum() / len(data)) * 100
    data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]
    # get list of col + % empty
    print(data_na)
    # get list of col + sum of nas
    print(data.isnull().sum())

def fillEmpty(train):
    # fill lot frontage with median of neighborhood in question since they are closely related
    train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    # fill nas with None
    for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
                'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType', 'MSSubClass'):
        train[col] = train[col].fillna('None')
    # fill nas with 0s
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF',
                'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
        train[col] = train[col].fillna(0)
    # fill na with typical
    train["Functional"] = train["Functional"].fillna("Typical")
    # fill these in with the mode
    for col in ('MSZoning', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType'):
        train[col] = train[col].fillna(train[col].mode()[0])
    # remove since same throughout apart from 1/2 rows
    train = train.drop(['Utilities'], axis=1)
    print(train.isnull().sum())
    return train


# adjusting price for inflation
def adjustPrice(data):
    # map contains CPI values needed to adjust prices
    map = {
        2010: 218.056,
        2009: 214.537,
        2008: 215.303,
        2007: 207.342,
        2006: 201.6
    }
    # apply inflation correction to each individual cost
    data['AdjustedPrice'] = data.apply(lambda x: x['SalePrice'] * (map[2010]/map[x['YrSold']]), axis=1)
    # remove original price
    return data.drop(['SalePrice'], axis=1)

def addTotSqFoot(data):
    data['TotSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

# def scatter(data):
    # sns.scatterplot(x=data['Id'], y=data['YrSold'])
    # plt.show()
    # unique values for year sold show that data only contains houses that were sold from 2006-2010
    # so take 2010 as the base year
    # print(data['YrSold'].unique())

def describe(train):
    # for (col in train.columns.values):
    # print(train.columns.values)
    # print(train['SalePrice'].corr(train['YearBuilt']))

    # get correlation of all cols wrt sale price
    # print(train[train.columns[1:]].corr()['SalePrice'][:])
    # print(train.head(10))

    # print(train['BldgType'].unique())
    # train['BldgType'] = train['BldgType'].astype('category')
    # print(train['BldgType'].unique())
    # print(train[train.columns[1:]].corr()['SalePrice'][:])
    print(train.isnull().sum())
    print(train['MiscFeature'].head(20))


def correlationMatrix(train):
    # corrMatrix = train.corr()
    # sns.heatmap(corrMatrix, annot=True)
    # plt.show()
    # saleprice correlation matrix

    # takes largest correlations with respect to sales price
    k = 15  # number of variables for heatmap
    corrMatrix = train.corr()
    cols = corrMatrix.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


if __name__ == '__main__':
    main()