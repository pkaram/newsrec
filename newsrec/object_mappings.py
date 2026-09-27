from newsrec.recommenders import popularity, implicit, svd, bpr, itemknn, ease

model_mappings = {
    'Popular': popularity.Popular(),
    'iALS': implicit.iALS(),
    'SVD': svd.SVD(),
    'BPR': bpr.BPR(),
    'ItemKNN': itemknn.ItemKNN(),
    'EASE': ease.EASE()
}

