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


# Exercicio 4
def weekly_study_hours(pais, dispositivo):
    new_df = df[(df['country'] == pais) & (df['device_type'] == dispositivo)]
    
    if new_df.empty:
        return "Sem dados para comparação"
    
    melhor_aluno = new_df.loc[new_df['final_grade'].idxmax()]
    
    return melhor_aluno['final_grade'], melhor_aluno['study_hours_weekly']
    

grade, study_hours = weekly_study_hours("USA", "Tablet")
print(f"highest grade: {grade}  study hours (weekly): {study_hours}")


























