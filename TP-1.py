# -*- coding: utf-8 -*-


# Exercicio - 1
import pandas as pd

df = pd.read_csv("online_learning_engagement_dataset.csv")

paises_alvo = ["Canada", "India", "USA"]
df_filtered = df[df['country'].isin(paises_alvo)]

df_filtered.to_csv("filtered_df.csv")

# Exercicio 2
import matplotlib.pyplot as plt

# Agrupar os dados
medias = df_filtered.groupby('country')[['video_watch_time_min', 'assignments_submitted']].mean()

# Criar o gráfico com subplots=True para separar as escalas
medias.plot(kind='bar', 
            subplots=True, 
            figsize=(10, 8), 
            color=['#1f77b4', '#ff7f0e'], 
            title=['Tempo Médio de Visualização (min)', 'Média de Tarefas Submetidas'],
            legend=False) # Removemos a legenda pois o título já explica

plt.xlabel('País')
plt.tight_layout() # Garante que os gráficos não ficam sobrepostos
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

# Exercicio 5
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

X_reg = df[['video_watch_time_min']]
y_reg = df['final_grade']

modelo_regressao = LinearRegression()
modelo_regressao.fit(X_reg, y_reg)


y_pred_reg = modelo_regressao.predict(X_reg)

mse = mean_squared_error(y_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_reg, y_pred_reg)

print("\n--- Análise de Erros (Exercício 5) ---")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Criar o Gráfico de Dispersão (Scatter Plot) com a reta de regressão 
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, alpha=0.5, color='teal', label='Dados Observados')
plt.plot(X_reg, y_pred_reg, color='red', linewidth=2, label=f'Reta de Regressão ($R^2$ = {r2:.2f})')

plt.title('Relação entre Tempo de Visualização de Vídeos e Nota Final')
plt.xlabel('Minutos de Observação de Vídeos (video_watch_time_min)')
plt.ylabel('Nota Final (final_grade)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()


# Exercicio 6
from sklearn.model_selection import train_test_split

features = ['device_type', 'study_hours_weekly', 'forum_posts']
X = df[features]
y = df['final_grade']

# device_type é uma variavel de texto
# É preciso numeros para o ML, por isso usar One-Hot Encoding
# O parâmetro drop_first=True evita o problema de multicolinearidade (Dummy Variable Trap).
X_encoded = pd.get_dummies(X, columns=['device_type'], drop_first=True)

# Dividir dataset em 80% Treino e 20% Teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Treinar modelo
# Using Multiple Linear Regression (bom para prever valores contínuos de forma interpretável)
modelo_ml = LinearRegression()
modelo_ml.fit(X_train, y_train)

# Fazer previsoes
y_pred = modelo_ml.predict(X_test)

# Avaliar performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados Machine Learning (Exercício 6) ---")
print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")

# Visualização de Resultados 
# Comparar graficamente os valores reais vs previsões
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Previsão Perfeita')
plt.xlabel('Nota Final Real')
plt.ylabel('Nota Final Prevista')
plt.title('Regressão Linear Múltipla: Valores Reais vs Previstos')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# Teste Extra: Random Forest
from sklearn.ensemble import RandomForestRegressor

# Usamos os mesmos dados de treino e teste do Exercício 6
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

y_pred_rf = modelo_rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"R² com Random Forest: {r2_rf:.4f}")
