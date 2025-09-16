# # Definizione di una variabile
# variabile = {'chiave': 3.3,
#              'prova':2}
# #variabile=(0,1,2,3)
# print(type(variabile))
# # Verifica se la variabile è un dizionario
# if isinstance(variabile, dict):
#     print("La variabile è un dizionario.")
# else:
#     print("La variabile non è un dizionario.")

import sys
sys.path.append(r'C:\Users\Luca\AMR23-FP7-MPSWMR\AMR23-FP7-MPSWMR\Model')
from decimal import Decimal
import numpy as np

dec=True
if dec:
    i=Decimal('0.0')
else:
    i=0
n=10
tau_step=0.1
tau_step_s=str(tau_step)
freq=10
camp=1/freq
camp_s=str(camp)
decimal_array = [i]
while i<n:
    if dec:
        i+=Decimal(tau_step_s)
    else:
        i+=tau_step
t = np.arange(Decimal('0.0'), i, Decimal(camp_s))
#t = np.arange(0, i, 1/10)
#np_array = np.array(decimal_array)
#print(t)
if i not in t:
    t = np.concatenate((t, np.array([i])))
for el in t:
    print(el)

it = len(t)
print("n",it)
for j in range(it - 1): 
    h = t[j+1] - t[j] #il problema è qui perchè essendo diventati di nuovo float, la loro differenza si rompe
    print(h)
    print(j)

i=Decimal('0.00001')
j=Decimal('1.3456')
print(type(i+j))
