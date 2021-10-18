from .recommenders import popularity, implicit, svd

model_mappings = {
    'Popular': popularity.Popular(),
    'iALS': implicit.iALS(),
    'SVD': svd.SVD()
}

