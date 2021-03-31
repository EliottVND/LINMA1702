import numpy as np
import matplotlib.pyplot as plt
from mip import *


# -----------------------------------
#       Donnees du probleme
# -----------------------------------

T = 350               # duree de la campagne
m = 5                 # nombre de classes d'age : [Young, Adult, Senior, Old, Centenarian]
c_tr  = 0             # Prix de livraison d'un vaccin
c_v   = 15            # prix d'administration d'un vaccin
c_tot = 100e6         # budget total autorise
b_l   = float("inf")  # nombre maximal de vaccins livres par jour
b_v   = 14646         # nombre maximal de vaccins administres par jour

# Fraction de malades
lambda_t = lambda t: np.array([0.000298 * (1/5 + np.sin(t/50-1/5)**2),
                               0.000301 * (1/5 + np.sin(t/50)**2),
                               0.000204 * (1/5 + np.sin(t/50-1/5)**2),
                               0.000209 * (1/5 + np.sin(t/50-2/5)**2),
                               0.000329 * (1/5 + np.sin(t/50-2/5)**2)])

# Fraction de morts
epsilon_t = lambda t: np.array([0.000100 * (6/5-t/1000),
                                0.000400 * (6/5-t/1000), 
                                0.005790 * (6/5-t/1000), 
                                0.027179 * (6/5-t/1000),
                                0.150000 * (6/5-t/1000)])

mu  = np.array([0.3, 0.6, 0.7, 0.9, 0.9])     # proportion de la population disposee a se faire vacciner.

# Quantite livree au hangar central
b_c = np.zeros(T);
days = np.arange(1, T+1)
b_c_eff =  [80000, 80000,60000, 60000, 40000, 40000, 40000, 40000, 60000,60000,60000,60000,80000,80000,80000,
            80000,100000,100000,100000,100000,100000,100000,100000,100000,120000,120000,120000,120000,120000,
            120000,120000,120000,120000,120000,120000,120000,150000,150000,150000,150000,150000,150000,150000,
            150000,150000,150000,150000,150000,150000,150000]

for t in days-1:
    if(t != 0 and t % 7 == 0):
        b_c[t-1] = b_c_eff[t//7]

b_c[-1] = b_c_eff[-1] # /!\: a modifier

_lambda  = lambda_t(days)
_epsilon = epsilon_t(days)


# -----------------------------------
#       Resolution du probleme
# -----------------------------------

model = Model('centre unique', sense=MINIMIZE, solver_name=CBC)

# Variables
x = np.array( [model.add_var() for t in days-1])
y = np.array([[model.add_var() for t in days-1] for i in range(m)])


# nombre de personnes susceptibles dans chaque classe d'age
n_s = [[3778123, 2846993, 2790883, 1390502, 111533]]  
for t in range(1, len(days)):
    n_s.append([(1 - _lambda[i,t-1]) * n_s[t-1][i] - y[i, t-1] for i in range(m)])
    
# Objectif
model.objective = minimize(xsum(_epsilon[i,t] * _lambda[i,t] * n_s[t][i] for i in range(m) for t in days[1:-1]))

# Contraintes
model += xsum(c_tr * x) + xsum(c_v * y[i,t] for i in range(m) for t in days-1) <= c_tot
for t in days-1:
    if t > 1: 
        model += x[t-1] - xsum(y[:,t]) >= 0
    model += x[t] <= b_c[t]
    model += xsum(y[:,t]) <= b_v
    for i in range(m):
        model += xsum(y[i,k] for k in range(t)) - mu[i] * n_s[0][i]  <= 0

model.optimize()

# Objectif
print(f"f(x,y) = {model.objective_value}")


# -----------------------------------
#        Plot des resultats
# -----------------------------------
population  = ["Young 0-29 ans", "Adult 30-49 ans", "Senior 50-69 ans", "Old 70-89 ans", "Centenarian 90-   ans"]
livraisons  = [x[t].x for t in days-1]
vaccination = [[y[i,t].x for t in days-1] for i in range(m)]
susceptible = [[n_s[t][i].x for t in days[1:-1]] for i in range(m)]
pop_s       = [sum(n_s[t][i].x for i in range(m)) for t in days[1:-1]]
pop_malade  = [sum( _lambda[i,t] * n_s[t][i].x for i in range(m)) for t in days[1:-1]]
pop_morte   = [sum(_epsilon[i,t] * _lambda[i,t] * n_s[t][i].x for i in range(m)) for t in days[1:-1]]

# Plot des livraisons
plt.figure()
plt.title("Livraisons")
markerline, stemlines, baseline = plt.stem(days, livraisons)
plt.xlabel("$t$ [jour]", fontsize=10)
plt.ylabel("vaccins", fontsize=10)
baseline.set_color('k')
baseline.set_linewidth(1)
markerline.set_markersize(1)
plt.show()

# Plot des populations au global
pop = [pop_s, pop_malade, pop_morte]
titres_pop = ["Population susceptible au temps t", "Population tombee malade au jour t", "Population malade au jour t qui va mourir"]
plt.figure()
for i in range(3):
    plt.title(titres_pop[i])
    markerline, stemlines, baseline = plt.stem(days[1:-1], pop[i])
    plt.xlabel("$t$ [jour]", fontsize=10)
    plt.ylabel("population", fontsize=10)
    baseline.set_color('k')
    baseline.set_linewidth(1)
    markerline.set_markersize(1)
    plt.show()

# Plot des populations par classe d'age
titres = [f"{population[i]}" for i in range(m)]

plt.figure("Question 1.1")
for i in range(5):
        plt.subplot(1, 1, 1)
        plt.title(titres[i])
        markerline, stemlines, baseline = plt.stem(days, vaccination[i])       
        plt.xlabel("$t$ [jour]", fontsize=10)
        plt.ylabel("personnes vaccinees", fontsize=10)

        baseline.set_color('k')
        baseline.set_linewidth(1)
        markerline.set_markersize(1)
        plt.grid()
        plt.show()
        
        plt.subplot(2, 1, 2)
        plt.plot(days[1:-1], susceptible[i])
        plt.xlabel("$t$ [jour]", fontsize=10)
        plt.ylabel("population", fontsize=10)
        plt.ylim((0, max(susceptible[i])))
        plt.grid()
        plt.show()

