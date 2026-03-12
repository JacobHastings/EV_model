import cvxpy as cp
import numpy as np

x = cp.Variable()
y = cp.Variable()

m = cp.Parameter(nonneg=True)
m.value = 3

constraints = [(m*x) + y == 1, x - y >= 1]

objective = cp.Minimize((x-y)**2)

prob = cp.Problem(objective,constraints)

prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)

A = cp.Parameter((24,24),nonneg=True)
A.value = np.ones((24,24))


