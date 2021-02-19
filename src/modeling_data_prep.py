
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

# Can concat to original data frame
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
                         'zip_psqft', 'price_per_lot_sqft', 'yard_size']]
    mod_df = pd.concat([continuous, inc, wf, in_city, bedrooms,
                       bathrooms, floor_dummies, cond_dummies, grade_dummies,
                       season_dummies, lc, yrbuilt_dummies, decades, fortyyr], axis=1)
    return mod_df

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
    
# def normalizing(df):
#     import numpy as np
#     import matplotlib.pyplot as plt
    
#     log_price = np.log(lux_ols_base['price'])
#     log_sqft_living = np.log(lux_ols_base['sqft_living'])
#     log_sqft_lot = np.log(lux_ols_base['sqft_lot'])
#     log_sqft_living15 = np.log(lux_ols_base['sqft_living15'])
#     log_sqft_lot15 = np.log(lux_ols_base['sqft_lot15'])

#     plt.hist(log_price, bins = 'auto', color = 'r', alpha = .7);
#     plt.hist(log_sqft_living, bins = 'auto', color = 'b', alpha = .7);
#     plt.hist(log_sqft_lot, bins = 'auto', color = 'g', alpha = .7);
#     plt.hist(log_sqft_living15, bins = 'auto', color = 'pink', alpha = .7);
#     plt.hist(log_sqft_lot15, bins = 'auto', color = 'y', alpha = .7);
    
#     return plt.show()
    
    
    
# def power_transform():
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     power = PowerTransformer()

#     power_price = power.fit_transform(np.array(log_price).reshape(-1, 1))
#     power_sqft_living = power.fit_transform(np.array(log_sqft_living).reshape(-1, 1))
#     power_sqft_lot = power.fit_transform(np.array(log_sqft_lot).reshape(-1, 1))
#     power_sqft_living15 = power.fit_transform(np.array(log_sqft_living15).reshape(-1, 1))
#     power_sqft_lot15 = power.fit_transform(np.array(log_sqft_lot15).reshape(-1, 1))

#     plt.hist(power_price, bins = 'auto', color = 'r', alpha = .7);
#     plt.hist(power_sqft_living, bins = 'auto', color = 'b', alpha = .7);
#     plt.hist(power_sqft_lot, bins = 'auto', color = 'g', alpha = .7);
#     plt.hist(power_sqft_living15, bins = 'auto', color = 'pink', alpha = .7);
#     plt.hist(power_sqft_lot15, bins = 'auto', color = 'y', alpha = .7);
#     return plt.show(), 

# def 
