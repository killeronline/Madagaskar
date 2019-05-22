import numpy as np

traits = []
op = 0 # Baseline
for hp in range(-2,3):
    for lp in range(-2,3):
        for cp in range(-2,3):
            hpC = max(op,hp,lp,cp)
            lpC = min(op,hp,lp,cp)
            traits.append([op,hpC,lpC,cp])

traits = np.array(traits).astype(float)
traits = np.unique(traits, axis=0)
traits += 3







traitsT = np.transpose(traits)




           