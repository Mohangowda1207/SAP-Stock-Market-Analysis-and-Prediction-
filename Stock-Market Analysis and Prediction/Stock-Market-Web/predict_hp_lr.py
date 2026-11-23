import pandas as pd
import numpy as np
import os


class StandardScalerLite:
	def __init__(self):
		self.mean_ = None
		self.scale_ = None

	def fit(self, X):
		arr = np.asarray(X, dtype=float)
		if arr.ndim == 1:
			arr = arr.reshape(-1, 1)
		self.mean_ = arr.mean(axis=0)
		self.scale_ = arr.std(axis=0)
		self.scale_[self.scale_ == 0] = 1.0
		return self

	def transform(self, X):
		arr = np.asarray(X, dtype=float)
		original_shape = arr.shape
		if arr.ndim == 1:
			arr = arr.reshape(-1, 1)
		out = (arr - self.mean_) / self.scale_
		if len(original_shape) == 1:
			return out.reshape(-1)
		return out

	def fit_transform(self, X):
		return self.fit(X).transform(X)


class LinearRegressionLite:
	def __init__(self):
		self.coef_ = None
		self.intercept_ = 0.0

	def fit(self, X, y):
		X_arr = np.asarray(X, dtype=float)
		y_arr = np.asarray(y, dtype=float)
		if y_arr.ndim == 1:
			y_arr = y_arr.reshape(-1, 1)
		ones = np.ones((X_arr.shape[0], 1))
		X_bias = np.hstack([ones, X_arr])
		params, *_ = np.linalg.lstsq(X_bias, y_arr, rcond=None)
		self.intercept_ = params[0]
		self.coef_ = params[1:]
		return self

	def predict(self, X):
		X_arr = np.asarray(X, dtype=float)
		if X_arr.ndim == 1:
			X_arr = X_arr.reshape(1, -1)
		return X_arr @ self.coef_ + self.intercept_


# Load HP.csv from parent directory, skipping extra header rows
csv_path = os.path.join(os.path.dirname(__file__), '..', 'HP.csv')
# Skip the first two rows (extra headers), set correct column names
df = pd.read_csv(csv_path, skiprows=2)
df.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']


# Use multiple features for prediction
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'

# Prepare the dataset
df_features = df[features].copy()
df_features['Close after n days'] = df_features['Close'].shift(-1)
df_features = df_features.dropna()

X = df_features[features].values
y = df_features['Close after n days'].values.reshape(-1, 1)

# Train/test split (use last row for prediction, rest for training)
X_train, y_train = X[:-1], y[:-1]
X_to_be_forecasted = X[-1].reshape(1, -1)

# Feature Scaling
sc = StandardScalerLite()
X_train_scaled = sc.fit_transform(X_train)
X_to_be_forecasted_scaled = sc.transform(X_to_be_forecasted)

# Train Linear Regression
clf = LinearRegressionLite()
clf.fit(X_train_scaled, y_train)

# Predict tomorrow's closing price
forecast_set = clf.predict(X_to_be_forecasted_scaled)
tomorrow_pred = float(forecast_set[0, 0]) if forecast_set.size else 0.0
print(f"Tomorrow's HP Closing Price Prediction by Linear Regression: {tomorrow_pred:.2f}")
