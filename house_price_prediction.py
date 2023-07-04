#Real Estate Price Prediction
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)
df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\Bengaluru_House_Data.csv')
#To make the model simpler we will remove few columns from the dataset
df1=df.drop(['area_type','availability','society','balcony'],axis='columns')

#Data cleaning process
#Handling Na values
df1.isnull().sum() #Total values are not available
df2=df1.dropna()
df2.isnull().sum()
df2['bhk']=df2['size'].apply(lambda x: int(x.split(' ')[0])) #take only the first token also convert it into integer value

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df2[~df2['total_sqft'].apply(is_float)].head(10)

#Now to make the data uniform and remove outliers
#Following function takes range as input and return average as output
def conv_sqft_num(x):
    tokens=x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df3=df2.copy()
df3['total_sqft']=df3['total_sqft'].apply(conv_sqft_num)

df4=df3.copy()
df4['price_per_sqft']=df4['price']*100000 / df4['total_sqft']

df4.location= df4.location.apply(lambda x: x.strip())

location_stats= df4.groupby('location')['location'].agg('count')

location_stats_less_than_10 = location_stats[location_stats<=10]
#len(df4.location.unique())=1293

#For all the locations having flats less than 10 make it's location as other 
df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
#len(df4.location.unique())=242

#Outliers present in the dataset
df4[df4.total_sqft/df4.bhk<300].head()

df5 = df4[~(df4.total_sqft/df4.bhk<300)]
# df5.shape=(12502, 7)

#Outlier Detection and removal
df5.price_per_sqft.describe()

#Function to remove extreme cases using standard deviation
#Remove price per sqft
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6 = remove_pps_outliers(df5)
# df6.shape=(10241, 7)


# Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='red',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df6,"Rajaji Nagar")

#Data Cleanup
# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df6)
# df7.shape=(7329, 7)

plot_scatter_chart(df7,"Rajaji Nagar")

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df7.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")

# It is unusual to have 2 more bathrooms than number of bedrooms in a home
df7[df7.bath>df7.bhk+2]

df8 = df7[df7.bath<df7.bhk+2]
# df8.shape=(7251, 7)
df9 = df8.drop(['size','price_per_sqft'],axis='columns')

#Converting text into numeric
dummies = pd.get_dummies(df9.location)
dummies.head(5)

df10 = pd.concat([df9,dummies.drop('other',axis='columns')],axis='columns')
df11 = df10.drop('location',axis='columns')
#Build a Model
X = df11.drop(['price'],axis='columns') #Dependent variable is price so remove it
y = df11.price
print("X_shape", X.shape)   #X_shape (7251, 244)
print("y_shape", y.shape)   #y_shape (7251,)

from sklearn.model_selection import train_test_split
X_trin,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=10) #20% test sample and 80% model training 
#train_test_split
from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)   #0.8452277697874371

#Use K Fold cross validation to measure accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
#Randomize samples
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)

#We can see that in 5 iterations we get a score above 80% all the time.
#This is pretty good but we want to test few other algorithms for
#regression to see if we can get even better score. 
#We will use GridSearchCV for this purpose 
#Best algorithm selection for that particular algorithm it will also tell best parameter
#Hyper parameter tuning

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)

#Test the model for few properties
#Will return estimated price
def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns)) 
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


predict_price('1st Phase JP Nagar',1000,2,2)

