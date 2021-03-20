from mip import *
import numpy as np

p = np.ones(6)
w = np.ones(6)
c = 100
I = range(len(w))

 

m = Model(sense=MAXIMIZE, solver_name=CBC)


x = [m.add_var(var_type=INTEGER, lb=0) for i in I]

m.objective = maximize(xsum(p[i] * x[i] for i in I))

 

m += xsum(w[i] * x[i] for i in I) <= c

for i in range(len(w)):
    m+= x[i] <= 1 + i


m.optimize()

 
for i in range(len(w)):
    print(x[i].x)

print(m.objective_value)
