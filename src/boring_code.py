####### TABLE OF CONTENTS #######
## 1. Data Cleaning
## 2. GRAPHS


### Data Cleaning ###

def cleaning(df, column_names, drop=None, date_col=None, date_suffix=None, date_split=None):
    """
    Replaces NaN/Unknown values for a set of given columns in a DataFrame, drops requested columns, and converts a date column into separate columns for month and year
    
    Arguments:
        df - Pandas dataframe
        columns - list of column names to adjust
        drop (optional) - list of columns names to drop
        dates - if you want a date column converted, specify the name of the column
        date_col - name of the date column name (str) you want converted
        date_split - character to split date value on (for example, for 2/3/2019, date_split='/')
        date_suffix - a list of strings (2 max) to be added to month and year column name (must be in order, ['month_suffix', 'year_suffix'])
    
    Returns: Dataframe with adjusted columns
    """ 
    import numpy as np
    
    for col in column_names:
        if '?' in df[col].value_counts().index:
            df[col] = df[col].replace('?', np.nan).astype('float64')
        else:
            df[col].fillna(np.nan, inplace=True)
            
    if drop != None:
        df.drop(labels=drop, axis=1, inplace=True)
        
    if date_col != None:
        col_name_month = 'month'+date_suffix[0]
        col_name_year = 'year'+date_suffix[1]
        df[col_name_month] = df[date_col].map(lambda x: int(x.split(date_split)[0]))
        df[col_name_year] = df[date_col].map(lambda x: int(x.split(date_split)[2]))
        df.drop(labels=date_col, axis=1, inplace=True)
    
    return df.head()

def new_features(df, new_column_names):
    """
    Replaces adds new columns to a dataframe (specifically for the KC Housing Data)
    Arguments:
        df - original pandas.DataFrame
        new_column_names - list of names of new columns
    
    Returns: Dataframe with new columns added
    """
    import numpy as np
    import pandas as pd
    
    for col in new_column_names:
        if col == 'season_sold':
            winter = [12,1,2]
            fall = [9,10,11]
            summer = [6,7,8]
            spring = [3,4,5]

            season_vals = []

            for month in df['month_sold']:
                if month in winter:
                    season_vals.append('winter')
                elif month in fall:
                    season_vals.append('fall')
                elif month in summer:
                    season_vals.append('summer')
                else:
                    season_vals.append('spring')

            df['season_sold'] = pd.DataFrame(season_vals)

        if col == 'yard_size':
            df['yard_size'] = df['sqft_lot']-df['sqft_above']-df['sqft_basement']
        if col == 'price_per_sqft':
            df['price_per_sqft'] = df['price'] / df['sqft_living']
        if col == 'decade_built':
            decades = [(df['yr_built'] >= 1900) & (df['yr_built'] < 1910),
               (df['yr_built'] >= 1910) & (df['yr_built'] < 1920),
               (df['yr_built'] >= 1920) & (df['yr_built'] < 1930),
               (df['yr_built'] >= 1930) & (df['yr_built'] < 1940),
               (df['yr_built'] >= 1940) & (df['yr_built'] < 1950),
               (df['yr_built'] >= 1950) & (df['yr_built'] < 1960),
               (df['yr_built'] >= 1960) & (df['yr_built'] < 1970),
               (df['yr_built'] >= 1970) & (df['yr_built'] < 1980),
               (df['yr_built'] >= 1980) & (df['yr_built'] < 1990),
               (df['yr_built'] >= 1990) & (df['yr_built'] < 2000),
               (df['yr_built'] >= 2000) & (df['yr_built'] < 2010),
               (df['yr_built'] >= 2010) & (df['yr_built'] < 2020)]

            decade_names = ['1900_1910', '1910_1920', '1920_1930', '1930_1940', 
                            '1940_1950', '1950_1960', '1960_1970', '1970_1980', 
                            '1980_1990', '1990_2000', '2000_2010', '2010_2020']

            df['decade_built'] = np.select(decades, decade_names)
            
            sections = [decade_names[:4], decade_names[4:8], decade_names[8:]]
            section_names = []
            for section in sections:
                for decade in df['decade_built']:
                    if decade in section:
                        section_names.append(section[0][:4]+"_"+section[3][5:])

            df['40yr_section'] = pd.DataFrame(section_names)
           
        if col == 'price_per_lot_sqft':
            df['price_per_lot_sqft'] = df['price'] / df['sqft_lot']
        if col == 'zip_psqft':
            zipcode_per_sqft = df.groupby('zipcode')['price_per_sqft'].mean()
            def zipcode_price(zipcode):
                try:
                    return zipcode_per_sqft.loc[zipcode]
                except:
                    return np.nan  
            df['zip_psqft'] = df['zipcode'].apply(zipcode_price)
            # location cost
            ranges = [(df['zip_psqft'] >= 100) & (df['zip_psqft'] < 200),
                      (df['zip_psqft'] >= 200) & (df['zip_psqft'] < 300),
                      (df['zip_psqft'] >= 300) & (df['zip_psqft'] < 400),
                      (df['zip_psqft'] >= 400) & (df['zip_psqft'] < 500),
                       df['zip_psqft'] >= 500]

            range_labels = ['zpsft100_200', 'zpsft200_300', 'zpsft300_400', 'zpsft400_500', 'zpsft500plus']

            df['location_cost'] = np.select(ranges, range_labels)
        if col == 'unincorporated':
            unincorporated_zipcodes = [98019, 98014, 98024, 98065, 98038, 98051, 98022, 98045, 
                                       98288, 98224, 98051, 98029, 98014, 98077, 98053, 98010, 
                                       98070]
            df['unincorporated'] = np.where(df['zipcode'].isin(unincorporated_zipcodes), 'Y', 'N')
        if col == 'in_city':
            seattle_zips = [98101, 98102, 98103, 98104, 98105, 98106, 98107, 98108, 98109, 98110, 98111, 98112, 98114, 98115, 
                            98116, 98117, 98118, 98119, 98121, 98122, 98124, 98125, 98126, 98129, 98131, 98132, 98133, 98134,
                            98136, 98138, 98144, 98145, 98146, 98148, 98151, 98154, 98155, 98158, 98160, 98161, 98164, 98166,
                            98168, 98170, 98171, 98174, 98177, 98178, 98181, 98184, 98185, 98188, 98190, 98191, 98195, 98198,
                            98199]
            df['in_city'] = np.where(df['zipcode'].isin(seattle_zips), 'Y', 'N')
        if col == 'waterfront':
            df['waterfront'][df['waterfront']==1] = 'Y'
            df['waterfront'][df['waterfront']==0] = 'N'
    return df.head()

def corr_table(df):
    """
    Creates a table of correlations and their corresponding variables (listed as a tuple)
    Arguments:
        df - pandas.DataFrame.corr()
    
    Returns: Dataframe with a list of variable pairs and their corresponding correlation coefficients
    """
    best_corrs = df.corr().abs().stack().reset_index().sort_values(0, ascending=False)

    best_corrs['pairs'] = list(zip(best_corrs.level_0, best_corrs.level_1))

    best_corrs.set_index(['pairs'], inplace = True)

    #d rop level columns
    best_corrs.drop(columns=['level_1', 'level_0'], inplace = True)
    
    best_corrs.columns = ['cc']
    
    best_corrs.drop_duplicates(inplace=True)
    best_corrs = best_corrs[best_corrs['cc']<1.0]
    
    return best_corrs

################################################################################
# GRAPHS


def price_by_grade(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.style.use('seaborn')
    
    fig, ax = plt.subplots(figsize=(10,6))

    grade_means = df.groupby('grade')['price'].mean().values

    bar1 = [i for i in range(3,14)]

    ax.bar(bar1, grade_means, width=.8, color='seagreen', label='Average House Price', align='center')

    ax.set_xticks([i for i in range(3,14)])
    ax.set_xlabel('Grade - King County Grading System (See Appendix)')
    ax.set_ylabel('Price (Millions)')
    ax.set_title('Comparing House Grade to Price', size=15)
    ax.legend()
    return ax

def price_distribution(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.histplot(df['price'], bins=20, kde=True)
    plt.title('Distribution of House Prices, King County')
    plt.axvline(x=df['price'].mean(), ymin=0, ymax=6000, lw=4, color='indigo', label='Average Price')
    plt.xlabel('Price (Millions)')
    plt.ylabel('Number of Houses')
    plt.legend()
    return plt.show()

def in_city_boxplots(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    
    in_city = df[['price', 'in_city']][df['in_city']=='Y']
    out_of_city = df[['price', 'in_city']][df['in_city']=='N']

    cdf = pd.concat([in_city, out_of_city])
    
    sns.boxplot(x='in_city', y='price', data=cdf, showfliers=False)
    plt.title('Price Distribution for City Houses vs Non-City Houses', size=15)  
    plt.ylabel('Price (Millions)')
    plt.xlabel('In City? Y or N')
    return plt.show()

def yr_built_dist(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.histplot(df.yr_built)
    return plt.show()

def zip_price_per_sqft_dist(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.histplot(df.zip_psqft)
    return plt.show()

def sqft_living_vs_price(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.regplot(x=df.sqft_living, y=df.price)
    return plt.show()

def waterfront_price_dist(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import matplotlib.pyplot as plt
    
    waterfront = df['price_per_sqft'][df['waterfront']=='Y']
    no_waterfront = df['price_per_sqft'][df['waterfront']=='N']
    fig, ax = plt.subplots(figsize=(10,6))
    ax.hist(waterfront, bins=20, color='blue', density=True, alpha=.7, label='Waterfront')
    ax.hist(no_waterfront, bins=20, color='springgreen', density=True, alpha=.7, label='No waterfront')
    ax.axvline(df['price_per_sqft'][df['waterfront']=='Y'].mean(), color='cyan', lw=4, label='Waterfront Homes MEAN Price')
    ax.axvline(df['price_per_sqft'][df['waterfront']=='N'].mean(), color='darkgreen', lw=4, label='Non-Waterfront Homes MEAN Price')
    ax.legend()
    ax.set_title('Waterfront or No Waterfront Property? Comparing Price Per Sqft Distribution')
    ax.set_xlabel('Price per Square Foot')
    ax.set_ylabel('Density')
    return ax

def incorp_vs_unincorp(df):
    """
    Plots the average price of a given grade of house:
    Arguments:
        df - pandas.DataFrame
    
    Returns: Bar graph plotting average price per house grade
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10,8))

    x = ['Incorporated', 'Unincorporated']
    y = [df['price_per_sqft'][(df['unincorporated']=='N')].mean(), df['price_per_sqft'][(df['unincorporated']=='Y')].mean()]
    ax.bar(x, y, color=['indigo', 'seagreen'])
    ax.set_title('Comparing Average Price per Sqft: Unincorporated vs Incorporated Houses', size=15)
    ax.set_ylabel('Average Price per Sqft')
    return ax

def ttest_dist_check(df):
    import matplotlib.pyplot as plt
    
    inc = df['price_per_sqft'][(df['unincorporated']=='N')]
    uninc = df['price_per_sqft'][(df['unincorporated']=='Y')]

    fig, ax = plt.subplots(figsize=(9,6))

    ax.hist(inc, bins=30, color='mediumseagreen', density=True, label='Incorporated Homes') # inc
    ax.hist(uninc, bins=30, color='blue', density=True, alpha=.7, label='Unincorporated Homes') # not inc
    
    ax.set_xlabel('Price per Sqft')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Price per Sqft: Incorporated vs Unincorporated Homes', size=15)

    ax.axvline(x=df['price_per_sqft'][(df['unincorporated']=='N')].mean(), color='orange', label='Incorporated Mean') # incorporated
    ax.axvline(x=df['price_per_sqft'][(df['unincorporated']=='Y')].mean(), color='black', label = 'Unincorporated Mean')
    ax.legend()
    return ax

def cloropleth(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    import geopandas as gpd
    
    kc = gpd.read_file('src/shapefile/Zipcodes_for_King_County_and_Surrounding_Area_(Shorelines)___zipcode_shore_area.shp')
    
    zip_psqft_df = df[['zipcode', 'zip_psqft']].drop_duplicates()
    cloropleth_info = kc.merge(zip_psqft_df, left_on = 'ZIP', right_on = 'zipcode')
    fig, ax = plt.subplots(figsize=(15,15))
    cloropleth_info.dropna().plot(column='zip_psqft',
                                       ax = ax,
                                       cmap = 'gist_earth',
                                       k=5, 
                                       legend = True);

    ax.set_title('Price per Square Foot per Zipcode', fontdict= 
                {'fontsize':25})
    ax.set_axis_off()
    return ax

def map_grades(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    import geopandas as gpd
    
    kc = gpd.read_file('src/shapefile/Zipcodes_for_King_County_and_Surrounding_Area_(Shorelines)___zipcode_shore_area.shp')
    
    zip_psqft_df = df[['zipcode', 'zip_psqft']].drop_duplicates()
    cloropleth_info = kc.merge(zip_psqft_df, left_on = 'ZIP', right_on = 'zipcode')
    fig, ax = plt.subplots(figsize=(15,15))
    cloropleth_info.dropna().plot(column='zip_psqft',
                                       ax = ax,
                                       cmap = 'gist_earth',
                                       k=5, 
                                       legend = True);
    ax.set_title('Distribution of Homes Using the King County Grading System', fontdict= 
            {'fontsize':25})
    ax.set_axis_off()
    ax.scatter(df['long'][df['grade']<6], df['lat'][df['grade']<6], color='mediumspringgreen', label='Grade 1-5 (Below code)')
    ax.scatter(df['long'][(df['grade']>=6)&(df['grade']<10)], df['lat'][(df['grade']>=6)&(df['grade']<10)], color='mediumslateblue', label='Grade 6-9 (Average)')
    ax.scatter(df['long'][df['grade']>=10], df['lat'][df['grade']>=10], color='fuchsia', label='Grade 10-13 (Above Average, higher quality)')
    ax.legend()
    
    return ax

def map_waterfront(df):
    import matplotlib.pyplot as plt
    import pandas as pd
    import geopandas as gpd
    
    kc = gpd.read_file('src/shapefile/Zipcodes_for_King_County_and_Surrounding_Area_(Shorelines)___zipcode_shore_area.shp')
    
    zip_psqft_df = df[['zipcode', 'zip_psqft']].drop_duplicates()
    cloropleth_info = kc.merge(zip_psqft_df, left_on = 'ZIP', right_on = 'zipcode')
    fig, ax = plt.subplots(figsize=(15,15))
    cloropleth_info.dropna().plot(column='zip_psqft',
                                       ax = ax,
                                       cmap = 'gist_earth',
                                       k=5, 
                                       legend = True);
    ax.set_title('Price per Square Foot per Zipcode', fontdict= 
            {'fontsize':25})
    ax.scatter(df['long'][(df['waterfront']=='Y')&(df['grade']>=9)], df['lat'][(df['waterfront']=='Y')&(df['grade']>=9)], color='firebrick', label='Waterfront Homes: House Grade 9+')
    ax.scatter(df['long'][(df['waterfront']=='Y')&(df['grade']<9)], df['lat'][(df['waterfront']=='Y')&(df['grade']<9)], color='darksalmon', label='Waterfront Homes: House Grade less than 9')
    ax.legend()
    ax.set_axis_off()
    
    return ax