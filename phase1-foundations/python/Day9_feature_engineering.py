# Types of feature engineering:

# 1. Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 2. Binning continuous variables
df['age_group'] = pd.cut(df['age'],
    bins=[0, 18, 35, 60, 100],
    labels=['child', 'young', 'adult', 'senior'])

# 3. Log transform (skewed features)
df['log_income'] = np.log1p(df['income'])

# 4. Target encoding (careful — data leakage!)
df['city_encoded'] = df.groupby('city')['target'].transform('mean')

# 5. Interaction features
df['height_weight'] = df['height'] * df['weight']
