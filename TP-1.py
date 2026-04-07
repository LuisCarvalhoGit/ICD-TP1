# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:06:03 2026

@author: nogue
"""

import pandas as pd

# Exercicio - 1
df = pd.read_csv("online_learning_engagement_dataset.csv")

paises_alvo = ["Canada", "India", "USA"]
df_filtered = df[df['country'].isin(paises_alvo)]

df_filtered.to_csv("filtered_df.csv")