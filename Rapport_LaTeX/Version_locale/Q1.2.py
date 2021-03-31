import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mip import *


def init_variables(model, m, n, t):
    """
    Initialise les variables x,y,z au format: x[t][j], y[t][j][i], z[t][j].
    
    @args: 
        m: int: nombre de classes d'age
        n: int: nombre de provinces
    
    @returns:
        x: List[List[Var]]: variables de livraison
        y: List[List[List[Var]]]: variables d'administration des vaccins
        z: List[List[Var]]: variables de stockage
    """    
    x = model.add_var_tensor((t, n), 'x')
    y = model.add_var_tensor((t, n, m), 'y')
    z = model.add_var_tensor((t, n), 'z')
    
    return x,y,z

def init_population(y, _lambda, population_Province):
    """
    Initialise la population de susceptibles au format: n_s[t][j][i].
    
    @args: 
        y: List[List[List[Var]]]: variables d'administration des vaccins
        _lambda: pandas DataFrame: donnees sur la fraction des malades
        population_Province: pandas DataFrame: donnees sur la population initiale dans chaque province.
    
    @returns: 
        n_s: List[List[List[Var]]]: population de susceptibles dans chaque province pour tout t.
        province: List[String]: noms des provinces.
        tranche: List[String]: noms des tranches
    """
    province = _lambda.columns.to_numpy()
    tranche =  _lambda.index.to_numpy()
    pop = population_Province["Population"].to_numpy()        
    n_s = [[[pop[i+m*j] for i in range(m)] for j in range(n)]]
    for t in range(1, len(days)):
        array = [[(1 - _lambda[province[j]][tranche[i]]) * n_s[t-1][j][i] - y[t-1][j][i] for i in range(m)] for j in range(n)]
        n_s.append(array)
    
    return n_s, province, tranche


# -----------------------------------
#       Donnees du probleme
# -----------------------------------

T = 350               # duree de la campagne
m = 5                 # nombre de classes d'age : [Young, Adult, Senior, Old, Centenarian]
n = 10                # Nombre de provinces
c_tr  = 0             # Prix de livraison d'un vaccin
c_v   = 15            # prix d'administration d'un vaccin
c_s   = 0             # Cout de stockage
c_tot = 100e6         # budget total autorise
b_l   = float("inf")  # nombre maximal de vaccins livres par jour
b_v   = 14646         # nombre maximal de vaccins administres par jour

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

_epsilon = epsilon_t(days)

# Extraction des donnees
_lambda = pd.read_csv("Donnees-v1.1/Fraction_malade_province.csv")
_lambda = _lambda.set_index('Tranche')

population_Province = pd.read_csv("Donnees-v1.1/Population_province.csv")


# -----------------------------------
#       Resolution du probleme
# -----------------------------------
model = Model('multi-centre', sense=MINIMIZE, solver_name=CBC)
model.clear() # On clear pour etre sur que le modele soit reinitialise comme il faut

# Initialisation des variables
x,y,z = init_variables(model, m, n, len(days))
n_s, province, tranche = init_population(y, _lambda, population_Province)


# Objectif
model.objective = minimize(xsum(_epsilon[i,t] * _lambda[province[j]][tranche[i]] * n_s[t-1][j][i] 
                                for i in range(m) for j in range(n) for t in days[1:-1]))

# Contrainte sur le cout total
model +=  xsum(c_tr * x[t][j] for j in range(n) for t in days-1) \
        + xsum(c_v * y[t][j][i] for i in range(m) for j in range(n) for t in days-1) \
        + xsum(c_s * z[t][j] for j in range(n) for t in days-1) <= c_tot

# On ne peut stocker ou administrer aucun vaccin le premier jour.
for j in range(n):   
    model += z[0][j] == 0
    for i in range(m):
        model += y[0][j][i] == 0
    
for t in days-1:   
    # Contrainte sur la quantite de vaccins livrables au temps t
    model += xsum(x[t][j] for j in range(n)) <= b_c[t]
    for j in range(n):   
        if t >= 1: 
            # Contrainte sur la quantite de vaccins disponibles et stockables au temps t
            model += x[t-1][j] + z[t-1][j] - xsum(y[t][j][i] for i in range(m)) - z[t][j] >= 0
            
        # Contrainte sur le nombre de vaccins administrables au temps t
        model += xsum(y[t][j][i] for i in range(m)) <= b_v # b_v est commun a toutes les provinces
        for i in range(m):
            # Contrainte sur le nombre de personnes vaccinables au temps t
            model += xsum(y[k][j][i] for k in range(t)) - mu[i] * n_s[0][j][i]  <= 0

model.optimize()

print(f"f(x,y,z) = {model.objective_value}")


# -----------------------------------
#        Plot des resultats
# -----------------------------------
population  = ["Young 0-29 ans", "Adult 30-49 ans", "Senior 50-69 ans", "Old 70-89 ans", "Centenarian 90-   ans"]
livraisons  = [[x[t][j].x for t in days-1] for j in range(n)]
vaccination = [sum(y[t][j][i].x for j in range(n) for i in range(m)) for t in days-1]
stockage = [[z[t][j].x for t in days-1] for j in range(n)]

population = [sum(n_s[t-1][j][i].x for j in range(n) for i in range(m)) for t in days[1:-1]]
pop_morte  = [sum(_epsilon[i,t] * _lambda[province[j]][tranche[i]] * n_s[t-1][j][i].x for j in range(n) for i in range(m)) for t in days[1:-1]]


# Plot des livraisons
for j in range(n):
    plt.figure(j)
    plt.title(f"Livraisons region : {province[j]}")
    markerline, stemlines, baseline = plt.stem(days, livraisons[:][j])
    plt.xlabel("$t$ [jour]", fontsize=10)
    plt.ylabel("vaccins", fontsize=10)
    baseline.set_color('k')
    baseline.set_linewidth(1)
    markerline.set_markersize(1)
    plt.show()

# Plot du stockage
for j in range(n):
    plt.figure(j)
    plt.title(f"Stockage region : {province[j]}")
    markerline, stemlines, baseline = plt.stem(days, stockage[:][j])
    plt.xlabel("$t$ [jour]", fontsize=10)
    plt.ylabel("vaccins", fontsize=10)
    baseline.set_color('k')
    baseline.set_linewidth(1)
    markerline.set_markersize(1)
    plt.show()
    
# Plot des vaccinations
plt.figure()
plt.title(f"Vaccination")
markerline, stemlines, baseline = plt.stem(days, vaccination)
plt.xlabel("$t$ [jour]", fontsize=10)
plt.ylabel("vaccins", fontsize=10)
baseline.set_color('k')
baseline.set_linewidth(1)
markerline.set_markersize(1)
plt.show()  

# Plot de la population
plt.figure()
plt.title(f"Population")
markerline, stemlines, baseline = plt.stem(days[1:-1], population)
plt.xlabel("$t$ [jour]", fontsize=10)
plt.ylabel("population", fontsize=10)
baseline.set_color('k')
baseline.set_linewidth(1)
markerline.set_markersize(1)
plt.show()  


# Plot de la population morte
plt.figure()
plt.title(f"Population morte")
markerline, stemlines, baseline = plt.stem(days[1:-1], pop_morte)
plt.xlabel("$t$ [jour]", fontsize=10)
plt.ylabel("population", fontsize=10)
baseline.set_color('k')
baseline.set_linewidth(1)
markerline.set_markersize(1)
plt.show()  
