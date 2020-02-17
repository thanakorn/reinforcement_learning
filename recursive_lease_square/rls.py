# %% Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %% Generate synthetic problem
num_data = 500
num_dim = 30

X = np.random.randn(num_data, num_dim)

w_true = np.random.randn(num_dim, 1)
y_true = X @ w_true + 0.8 * np.random.randn(num_data, 1)

# %% Linear regression using pseudoinverse method
w_est = np.linalg.inv(X.T @ X) @ X.T @ y_true
y_est = X @ w_est

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
fig.suptitle('Linear Regression using Pseudoinverse')
ax[0].set_title('Accuracy')
ax[0].scatter(y_true, y_est)
ax[0].grid(True)
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Prediction")

ax[1].set_title('Model')
ax[1].scatter(w_true, w_est)
ax[1].grid(True)
ax[1].set_xlabel("True Weights")
ax[1].set_ylabel("Estimated Weights")

print("Error : %.2f" %(np.linalg.norm(y_true - y_est)))
# plt.savefig("PI.png")

# %% Linear regression using gradient descent
w_grad = np.random.randn(num_dim, 1)
num_iteration = 300
learning_rate = 0.0001
errors = np.zeros((num_iteration, 1))

print("Error : %.2f" %(np.linalg.norm(y_true - (X @ w_grad))))

for i in range(num_iteration):
    errors[i] = np.linalg.norm(y_true - (X @ w_grad))
    gradient = X.T @ (X @ w_grad - y_true)
    w_grad -= learning_rate * gradient

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
fig.suptitle('Linear Regression using Gradient Descent')
ax[0].set_title('Accuracy')
ax[0].scatter(y_true, (X @ w_grad))
ax[0].grid(True)
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Prediction")

ax[1].set_title('Model')
ax[1].scatter(w_true, w_grad)
ax[1].grid(True)
ax[1].set_xlabel("True Weights")
ax[1].set_ylabel("Estimated Weights")

ax[2].set_title('Learning Curve')
ax[2].plot(range(num_iteration), errors)
ax[2].grid(True)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Error")
# plt.savefig("GD.png")

# %% Linear regression using stochastic gradient descent
w_sgd = np.random.randn(num_dim, 1)
num_iteration = 1000
learning_rate = 0.05
errors = np.zeros((num_iteration, 1))

for i in range(num_iteration):
    errors[i] = np.linalg.norm(y_true - (X @ w_sgd))
    index = np.floor(np.random.rand() * num_data).astype(int)
    x = np.reshape(X[index,:], (1, num_dim))
    y = y_true[index]
    predict = x @ w_sgd
    gradient = (predict - y) * x.T
    w_sgd -= learning_rate * gradient

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
fig.suptitle('Linear Regression using Stochastic Gradient Descent')
ax[0].set_title('Accuracy')
ax[0].scatter(y_true, (X @ w_sgd))
ax[0].grid(True)
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Prediction")

ax[1].set_title('Model')
ax[1].scatter(w_true, w_sgd)
ax[1].grid(True)
ax[1].set_xlabel("True Weights")
ax[1].set_ylabel("Estimated Weights")

ax[2].set_title('Learning Curve')
ax[2].plot(range(num_iteration), errors)
ax[2].grid(True)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Error")
# plt.savefig("SGD.png")

# %% Mini-batch SGD
w_msgd = np.random.randn(num_dim, 1)
num_iteration = 20
learning_rate = 0.03
batch_size = 30
num_epoch = 20
errors = list()

for i in range(num_iteration):
    for indices in np.array_split(np.array(range(num_data)), num_epoch):
        errors.append(np.linalg.norm(y_true - (X @ w_msgd)))
        # indices = (np.random.rand(batch_size) * num_data).astype(int)
        x = X[indices,:]
        y = y_true[indices]
        predict = x @ w_msgd
        # gradient = (x * (predict - y)).sum(axis=0)
        gradient = x.T @ (x @ w_msgd - y)
        w_msgd -= learning_rate * np.reshape(gradient, (num_dim, 1))

fig, ax = plt.subplots(figsize=(5,4))
ax.set_title('Learning Curve')
ax.plot(range(num_iteration * num_epoch), np.array(errors))
ax.grid(True)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
# plt.savefig("MSGD.png")

# %% Learning rate decay
w_dsgd = np.random.randn(num_dim, 1)
num_iteration = 1000
learning_rate = 0.05
errors = np.zeros((num_iteration, 1))
decay = 0.0001

for i in range(num_iteration):
    errors[i] = np.linalg.norm(y_true - (X @ w_dsgd))
    index = np.floor(np.random.rand() * num_data).astype(int)
    x = np.reshape(X[index,:], (1, num_dim))
    y = y_true[index]
    predict = x @ w_dsgd
    gradient = (predict - y) * x.T
    w_dsgd -= learning_rate * gradient
    learning_rate *= (1/(1 + decay * i))
    # print(learning_rate)

fig, ax = plt.subplots(figsize=(4,4))
ax.set_title('Learning Curve')
ax.plot(range(num_iteration), errors)
ax.grid(True)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
# plt.savefig("decay-SGD.png")

# %% Recursive Least Sqaure
w_rls = np.random.randn(num_dim, 1)
l = 1.0
num_iteration = num_data
errors = np.zeros((num_iteration, 1))
p = np.eye(num_dim)

for i in range(num_iteration):
    errors[i] = np.linalg.norm(y_true - (X @ w_rls))
    index = np.floor(np.random.rand() * num_data).astype(int)
    x = np.reshape(X[index,:], (num_dim, 1))
    y = y_true[index]
    k = (p @ x / l) / (1 + (x.T @ p @ x / l))
    e = y - (w_rls.T @ x)
    w_rls = w_rls + (k * e)
    # Update P
    p = (p / l) - ((k @ x.T @ p) / l)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
fig.suptitle('Linear Regression using Recursive Least Square')
ax[0].set_title('Accuracy')
ax[0].scatter(y_true, (X @ w_rls))
ax[0].grid(True)
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Prediction")

ax[1].set_title('Model')
ax[1].scatter(w_true, w_rls)
ax[1].grid(True)
ax[1].set_xlabel("True Weights")
ax[1].set_ylabel("Estimated Weights")

ax[2].set_title('Learning Curve')
ax[2].plot(range(num_iteration), errors)
ax[2].grid(True)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Error")
# plt.savefig("RLS.png")

fig, ax = plt.subplots(figsize=(4,4))
ax.set_title(f'num_dimension = {num_dim}')
ax.plot(range(num_iteration), errors)
ax.grid(True)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
# plt.savefig(f'RLS-{num_dim}D.png')

# %% Load UCI Dataset
data = pd.read_csv('auto-mpg.data', sep='\s+',
                    names=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model', 'origin', 'name'])
data = data[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']]

X = data[['displacement', 'horsepower', 'weight', 'acceleration']].to_numpy()
X = (X - X.mean(axis = 0)) / X.std(axis=0)
Y = data['mpg'].to_numpy()
Y = (Y - Y.mean()) / Y.std()

num_data = X.shape[0]
num_features = X.shape[1]

# %% Train with SGD
w_sgd = np.random.randn(num_features, 1)
num_iteration = 500
learning_rate = 0.01
errors = np.zeros((num_iteration, 1))
decay = 0.0001

for i in range(num_iteration):
    errors[i] = np.linalg.norm(Y - (X @ w_sgd))
    index = np.floor(np.random.rand() * num_data).astype(int)
    x = np.reshape(X[index,:], (1, num_features))
    y = Y[index]
    predict = x @ w_sgd
    gradient = (predict - y) * x.T
    w_sgd -= learning_rate * gradient
    learning_rate *= (1/(1 + decay * i))

fig, ax = plt.subplots(figsize=(4,4))
# fig.suptitle('AutoMPG using Stochastic Gradient Descent')
# ax[0].set_title('Accuracy')
# ax[0].scatter(Y, (X @ w_sgd))
# ax[0].grid(True)
# ax[0].set_xlabel("Target")
# ax[0].set_ylabel("Prediction")

ax.set_title('Stochastic GD Learning Curve')
ax.plot(range(num_iteration), errors)
ax.grid(True)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
# plt.savefig("MPG-SGD.png")

# %% Train with RLS
w_rls = np.random.randn(num_features, 1)
l = 1.0
num_iteration = 500
errors = np.zeros((num_iteration, 1))
# x0 = np.reshape(X[0,:],(1, num_features))
# r = x0.T @ x0
p = np.eye(num_features)

for i in range(num_iteration):
    errors[i] = np.linalg.norm(Y - (X @ w_rls))
    index = np.floor(np.random.rand() * num_data).astype(int)
    x = np.reshape(X[index,:], (num_features, 1))
    y = Y[index]
    k = (p @ x / l) / (1 + (x.T @ p @ x / l))
    e = y - (w_rls.T @ x)
    w_rls = w_rls + (k * e)
    # Update P
    p = (p / l) - ((k @ x.T @ p) / l)

fig, ax = plt.subplots(figsize=(4,4))
# fig.suptitle('AutoMPG using Recursive Least Square')
# ax[0].set_title('Accuracy')
# ax[0].scatter(Y, (X @ w_rls))
# ax[0].grid(True)
# ax[0].set_xlabel("Target")
# ax[0].set_ylabel("Prediction")

ax.set_title('Recursive Least Square Learning Curve')
ax.plot(range(num_iteration), errors)
ax.grid(True)
ax.set_xlabel("Iteration")
ax.set_ylabel("Error")
# plt.savefig("MPG-RLS.png")

# %% Generalization
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# %%
w_sgd = np.random.randn(num_features, 1)
num_iteration = 500
learning_rate = 0.01
errors = np.zeros((num_iteration, 1))
decay = 0.0001

for i in range(num_iteration):
    errors[i] = np.linalg.norm(Y_train - (X_train @ w_sgd))
    index = np.floor(np.random.rand() * X_train.shape[0]).astype(int)
    x = np.reshape(X_train[index,:], (1, num_features))
    y = Y_train[index]
    predict = x @ w_sgd
    gradient = (predict - y) * x.T
    w_sgd -= learning_rate * gradient
    learning_rate *= (1/(1 + decay * i))

fig, ax = plt.subplots(figsize=(4,4))
ax.set_title('SGD Accuracy')
ax.scatter(Y_test, (X_test @ w_sgd))
ax.grid(True)
ax.set_xlabel("Target")
ax.set_ylabel("Prediction")

# %%
w_rls = np.random.randn(num_features, 1)
l = 1.0
num_iteration = 500
errors = np.zeros((num_iteration, 1))
# x0 = np.reshape(X[0,:],(1, num_features))
# r = x0.T @ x0
p = np.eye(num_features)

for i in range(num_iteration):
    errors[i] = np.linalg.norm(Y_train - (X_train @ w_rls))
    index = np.floor(np.random.rand() * X_train.shape[0]).astype(int)
    x = np.reshape(X_train[index,:], (num_features, 1))
    y = Y_train[index]
    k = (p @ x / l) / (1 + (x.T @ p @ x / l))
    e = y - (w_rls.T @ x)
    w_rls = w_rls + (k * e)
    # Update P
    p = (p / l) - ((k @ x.T @ p) / l)

fig, ax = plt.subplots(figsize=(4,4))
ax.set_title('RLS Accuracy')
ax.scatter(Y_test, (X_test @ w_rls))
ax.grid(True)
ax.set_xlabel("Target")
ax.set_ylabel("Prediction")

# %%
