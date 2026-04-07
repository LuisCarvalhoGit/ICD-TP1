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

# Exercicio 2
import matplotlib.pyplot as plt

medias = df_filtered.groupby('country')[['video_watch_time_min', 'assignments_submitted']].mean()

medias.plot(kind='bar', figsize=(10, 6))
plt.title('Média de Engajamento por País')
plt.ylabel('Valor Médio')
plt.show()

# Exercicio 3

df_usa = df_filtered[df_filtered['country'] == "USA"]
gender = df_usa['gender'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(gender, labels=gender.index, autopct='%1.1f%%', colors=['pink', 'skyblue'])
plt.title('Distribuição de Género nos USA')
plt.legend()
plt.show()