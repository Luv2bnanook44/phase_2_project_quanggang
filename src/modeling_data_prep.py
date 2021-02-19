
def add_dummies(data):
    import pandas as pd

    in_city_series = pd.Series(data['in_city']).astype('category')

    in_city = pd.get_dummies(in_city_series, prefix='incity', drop_first=True)
    
    inc_series = pd.Series(data['unincorporated']).astype('category')
    inc = pd.get_dummies(inc_series, prefix='inc', drop_first=True)
    wf_series = pd.Series(data['waterfront'])

    cat_wf = wf_series.astype('category')

    wf = pd.get_dummies(cat_wf, prefix='wf', drop_first=True)
    bedroom_series = pd.Series(data['bedrooms']).astype('category')
    bedrooms = pd.get_dummies(bedroom_series, prefix='bed', drop_first=True)
    bathrooms_series = pd.Series(data['bathrooms'])

    cat_bathrooms = bathrooms_series.astype('category')
    bathrooms = pd.get_dummies(cat_bathrooms, prefix='bath', drop_first=True)
    floor_dummies = pd.get_dummies(data['floors'].astype(int), prefix='floor', drop_first=True)
    cond_dummies = pd.get_dummies(data['condition'].astype(int), prefix='cond', drop_first=True)
    grade_dummies = pd.get_dummies(data['grade'].astype(int), prefix='grade', drop_first=True)
    season_dummies = pd.get_dummies(data['season_sold'], prefix='month', drop_first=True)
    lc_series = pd.Series(data['location_cost']).astype('category')

    lc = pd.get_dummies(lc_series, drop_first=True)
    yrbuilt_dummies = pd.get_dummies(data['yr_built'].astype(int), prefix='built', drop_first=True)
    
    fortyyr_series = pd.Series(data['40yr_section']).astype('category')
    fortyyr = pd.get_dummies(fortyyr_series, prefix='fortyyr', drop_first=True)
    decade_series = pd.Series(data['decade_built']).astype('category')

    decades = pd.get_dummies(decade_series, prefix='decb', drop_first=True)
    
    continuous = data[['price', 'sqft_living', 'sqft_lot', 'sqft_above', 
                         'sqft_living15', 'sqft_lot15', 'price_per_sqft',
                         'zip_psqft', 'price_per_lot_sqft', 'yard_size', 'grade', 'condition']]
    mod_df = pd.concat([continuous, inc, wf, in_city, bedrooms,
                       bathrooms, floor_dummies, cond_dummies, grade_dummies,
                       season_dummies, lc, yrbuilt_dummies, decades, fortyyr], axis=1)
    return mod_df

def highest_rsquared(data):
    from statsmodels.formula.api import ols
    import scipy.stats as stats
    import pandas as pd
    import numpy as np
    x_cols = ['sqft_living', 'price_per_sqft',
              'wf_Y','zpsft500plus','price_per_lot_sqft',
              'grade_13','zpsft200_300',
              'zpsft200_300','zpsft300_400','zip_psqft',
              'sqft_living15','incity_Y','grade_12']
    predictors = '+'.join(x_cols)
                      
    formula = 'price' + '~' + predictors

    model = ols(formula = formula, data = data).fit() 
    
    return model, data[x_cols]

def continuous_dists(data, cols):
    import matplotlib.pyplot as plt

    continuous_vars = cols
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,12))
    for i in range(0,3):
        ax[0][i].hist(data[continuous_vars[i]], bins='auto')
        ax[0][i].set_title(continuous_vars[i]+' Distribution')
    for i in range(0,2):
        ax[1][i].hist(data[continuous_vars[i+3]], bins='auto')
        ax[1][i].set_title(continuous_vars[i+3]+' Distribution')
    return ax
    
def normalizing(data):
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    
    data = data[(data['grade']>=6)&(data['condition']>=3)]
    data = data[~(np.abs(stats.zscore(data['price'])) > 3)]
    data = data[~(np.abs(stats.zscore(data['sqft_living'])) > 3)]
    data = data[~(np.abs(stats.zscore(data['sqft_living15'])) > 3)]
    
    log_price = np.log(data['price'])
    log_sqft_living = np.log(data['sqft_living'])
    log_sqft_lot = np.log(data['sqft_lot'])
    log_sqft_living15 = np.log(data['sqft_living15'])
    log_sqft_lot15 = np.log(data['sqft_lot15'])
    log_yard_size = np.log(data['yard_size'])

    plt.hist(log_price, bins = 'auto', color = 'r', alpha = .7);
    plt.hist(log_sqft_living, bins = 'auto', color = 'b', alpha = .7);
    plt.hist(log_sqft_lot, bins = 'auto', color = 'g', alpha = .7);
    plt.hist(log_sqft_living15, bins = 'auto', color = 'pink', alpha = .7);
    plt.hist(log_sqft_lot15, bins = 'auto', color = 'y', alpha = .7);
    plt.title('Normalized Variables: Price, sqft_living, sqft_lot, sqft_living15, sqft_lot15, yard_size')
    logged_vars = [log_price, log_sqft_living, log_sqft_lot, 
                   log_sqft_living15, log_sqft_lot15, log_yard_size]
    
    return logged_vars, plt.show()
    
    
    
def power_transform(logged=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    
    if logged==None:
        return print("Error: must have list-like logged argument that contains 6 elements.")
    
    power = PowerTransformer()

    power_price = power.fit_transform(np.array(logged[0]).reshape(-1, 1))
    power_sqft_living = power.fit_transform(np.array(logged[1]).reshape(-1, 1))
    power_sqft_lot = power.fit_transform(np.array(logged[2]).reshape(-1, 1))
    power_sqft_living15 = power.fit_transform(np.array(logged[3]).reshape(-1, 1))
    power_sqft_lot15 = power.fit_transform(np.array(logged[4]).reshape(-1, 1))
    power_yard_size = power.fit_transform(np.array(logged[5]).reshape(-1, 1))

    plt.hist(power_price, bins = 'auto', color = 'r', alpha = .7);
    plt.hist(power_sqft_living, bins = 'auto', color = 'b', alpha = .7);
    plt.hist(power_sqft_lot, bins = 'auto', color = 'g', alpha = .7);
    plt.hist(power_sqft_living15, bins = 'auto', color = 'pink', alpha = .7);
    plt.hist(power_sqft_lot15, bins = 'auto', color = 'y', alpha = .7);
    plt.title('Power Transformed Variables (Centered around Zero)')
    powered_vars = [power_price, power_sqft_living, power_sqft_living15,
                    power_sqft_lot, power_sqft_lot15, power_yard_size]
    return powered_vars, plt.show()

def final_model_all_homes(data, powered=None):
    from statsmodels.formula.api import ols
    import scipy.stats as stats
    import pandas as pd
    import numpy as np
    
    data = data[(data['grade']>=6)&(data['condition']>=3)]
    data = data[~(np.abs(stats.zscore(data['price'])) > 3)]
    data = data[~(np.abs(stats.zscore(data['sqft_living'])) > 3)]
    data = data[~(np.abs(stats.zscore(data['sqft_living15'])) > 3)]
    
    grade_dummies = pd.get_dummies(data['grade'].astype(int),
                                   prefix='grade', drop_first=True)
    lc_series = pd.Series(data['location_cost']).astype('category')

    lc = pd.get_dummies(lc_series, drop_first=True)
    if powered==None:
        return print("Error: Must have list-like powered argument")
    data['price'] = powered[0]
    data['sqft_living'] = powered[1]
    data['sqft_living15'] = powered[2]
    
    final_data = pd.concat([data[['price','sqft_living',
                                 'sqft_living15']], 
                           grade_dummies, lc], axis=1)
    
    x_cols = final_data.drop('price', axis = 1).columns
    predictors = '+'.join(x_cols)
                      
    formula = 'price' + '~' + predictors

    model = ols(formula = formula, data = final_data).fit()                  
    return model, final_data

def final_model_luxury(data):
    from statsmodels.formula.api import ols
    import pandas as pd
    import scipy.stats as stats
    import numpy as np
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    
    ## Log and power transform
    
    data = data[(data['grade']>=10)&(data['condition']>=3)]
    
    log_price = np.log(data['price'])
    log_sqft_living = np.log(data['sqft_living'])
    log_sqft_lot = np.log(data['sqft_lot'])
    log_sqft_living15 = np.log(data['sqft_living15'])
    log_sqft_lot15 = np.log(data['sqft_lot15'])
    log_yard_size = np.log(data['yard_size'])
    
    logged_vars = [log_price, log_sqft_living, log_sqft_lot, 
                   log_sqft_living15, log_sqft_lot15, log_yard_size]
    
    logged = logged_vars
    
    power = PowerTransformer()

    power_price = power.fit_transform(np.array(logged[0]).reshape(-1, 1))
    power_sqft_living = power.fit_transform(np.array(logged[1]).reshape(-1, 1))
    power_sqft_lot = power.fit_transform(np.array(logged[2]).reshape(-1, 1))
    power_sqft_living15 = power.fit_transform(np.array(logged[3]).reshape(-1, 1))
    power_sqft_lot15 = power.fit_transform(np.array(logged[4]).reshape(-1, 1))
    power_yard_size = power.fit_transform(np.array(logged[5]).reshape(-1, 1))
    
    powered_vars = [power_price, power_sqft_living, power_sqft_living15,
                    power_sqft_lot, power_sqft_lot15, power_yard_size]
    
    
    ##Prepare model
    
    grade_dummies = pd.get_dummies(data['grade'].astype(int),
                                   prefix='grade', drop_first=True)
    lc_series = pd.Series(data['location_cost']).astype('category')
    lc = pd.get_dummies(lc_series, drop_first=True) 
    cond_dummies = pd.get_dummies(data['condition'].astype(int), 
                                  prefix='cond', drop_first=True)
    inc_series = pd.Series(data['unincorporated']).astype('category')
    inc = pd.get_dummies(inc_series, prefix='inc', drop_first=True)
    
    wf_series = pd.Series(data['waterfront'])

    cat_wf = wf_series.astype('category')

    wf = pd.get_dummies(cat_wf, prefix='wf', drop_first=True)
    
    powered = powered_vars
    data['price'] = powered[0]
    data['sqft_living'] = powered[1]
    data['sqft_living15'] = powered[2]
    
    final_data = pd.concat([data[['price','sqft_living', 'sqft_living15']],
                           wf, cond_dummies,
                           grade_dummies, lc, inc], axis=1)
    
    x_cols = final_data.drop('price', 
                             axis = 1).columns
    predictors = '+'.join(x_cols)
                      
    formula = 'price' + '~' + predictors

    model = ols(formula = formula, data = final_data).fit() 
    
    return model, final_data

def final_model_non_lux(data):
    from statsmodels.formula.api import ols
    import pandas as pd
    import scipy.stats as stats
    import numpy as np
    from sklearn.preprocessing import StandardScaler, PowerTransformer
    # subsetting the model
    data = data[(data['grade']>6)&(data['grade']<=9)&(data['condition']>=3)]
    data = data[~(np.abs(stats.zscore(data['price'])) > 3)]
    data = data[~(np.abs(stats.zscore(data['sqft_living'])) > 3)]
    data = data[~(np.abs(stats.zscore(data['sqft_living15'])) > 3)]
    ## Log and power transform
    
    log_price = np.log(data['price'])
    log_sqft_living = np.log(data['sqft_living'])
    log_sqft_lot = np.log(data['sqft_lot'])
    log_sqft_living15 = np.log(data['sqft_living15'])
    log_sqft_lot15 = np.log(data['sqft_lot15'])
    log_yard_size = np.log(data['yard_size'])
    
    logged_vars = [log_price, log_sqft_living, log_sqft_lot, 
                   log_sqft_living15, log_sqft_lot15, log_yard_size]
    
    logged = logged_vars
    
    power = PowerTransformer()

    power_price = power.fit_transform(np.array(logged[0]).reshape(-1, 1))
    power_sqft_living = power.fit_transform(np.array(logged[1]).reshape(-1, 1))
    power_sqft_lot = power.fit_transform(np.array(logged[2]).reshape(-1, 1))
    power_sqft_living15 = power.fit_transform(np.array(logged[3]).reshape(-1, 1))
    power_sqft_lot15 = power.fit_transform(np.array(logged[4]).reshape(-1, 1))
    power_yard_size = power.fit_transform(np.array(logged[5]).reshape(-1, 1))
    
    powered_vars = [power_price, power_sqft_living, power_sqft_living15,
                    power_sqft_lot, power_sqft_lot15, power_yard_size]
    
    # Get dummies and prepare model data
    grade_dummies = pd.get_dummies(data['grade'].astype(int),
                                   prefix='grade', drop_first=True)
    lc_series = pd.Series(data['location_cost']).astype('category')
    lc = pd.get_dummies(lc_series, drop_first=True) 
    cond_dummies = pd.get_dummies(data['condition'].astype(int), 
                                  prefix='cond', drop_first=True)
    powered = powered_vars
    data['price'] = powered[0]
    data['sqft_living'] = powered[1]
    data['sqft_living15'] = powered[2]
    data['sqft_lot'] = powered[3]
    
    final_data = pd.concat([data[['price','sqft_living', 'sqft_living15',
                                 'sqft_lot']], grade_dummies, lc, 
                            cond_dummies], axis=1)
    
    x_cols = final_data.drop('price', axis = 1).columns
    predictors = '+'.join(x_cols)
                      
    formula = 'price' + '~' + predictors

    model = ols(formula = formula, data = final_data).fit() 
    return model, final_data

