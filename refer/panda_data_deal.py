import pandas as pd
from sklearn import preprocessing
# https://vitobellini.github.io/posts/2018/01/03/how-to-build-a-recommender-system-in-tensorflow.html
# https://stackoverflow.com/questions/44898080/recommender-system-svd-with-tensorflow
# https://blog.csdn.net/m0_38045485/article/details/81174728

df = pd.read_csv('./data/ml-1m/ratings.dat', sep='\t', names=['user', 'item', 'rating', 'timestamp'], header=None)
df = df.drop('timestamp', axis=1)

num_items = df.item.nunique()
num_users = df.user.nunique()
print(len(df))
print("USERS: {} ITEMS: {}".format(num_users, num_items))

r = df['rating'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(r.reshape(-1,1))
df_normalized = pd.DataFrame(x_scaled)
df['rating'] = df_normalized

# Convert DataFrame in user-item matrix
matrix = df.pivot(index='user', columns='item', values='rating')
matrix.fillna(0, inplace=True)
# Users and items ordered as they are in matrix
print(matrix)
users = matrix.index.tolist()
items = matrix.columns.tolist()

matrix = matrix.iloc[:,:].values
print(type(matrix))

