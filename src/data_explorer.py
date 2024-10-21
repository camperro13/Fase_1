import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats
import statsmodels.api as sm

def explore_data(data):
    print('Explorando dataset:')
    print(data.head().T)
    print(data.describe())
    print(data.info())
    na = data.isna().mean()*100
    flag = 0
    for i in range(0,len(na)):
        if na.iloc[i] > 0:
            flag = 1
            print('Valores faltantes en: ')
            print(na.iloc[[i]])
    if flag == 0:
        print('No hay valores faltantes.')
    
def plot_depen_indepen(data):
    physical_columns = ['V-1']
    V9 = 'V-9'  # Actual sales prices (target)
    V10 = 'V-10'  # Actual construction costs (target)
    # Crear un boxplot para V-9 (Actual sales prices)
    data.boxplot(column=V9, by=physical_columns, grid=False)
    plt.title('Boxplot - Actual Sales Prices vs Independent Variables')
    plt.suptitle('')
    plt.xticks(rotation=90)
    plt.show()
    # Crear un boxplot para V-10 (Actual construction costs)
    data.boxplot(column=V10, by=physical_columns, grid=False)
    plt.title('Boxplot - Actual Construction Costs vs Independent Variables')
    plt.suptitle('')
    plt.xticks(rotation=90)
    plt.show()   

def plot_histograms(data):
    numeric = data.columns
    fig, axes = plt.subplots(int(len(numeric)/5)+1, 5, figsize=(25, 25))
    i_limit = int(len(numeric)/5) + 1
    j_limit = 4
    i = 0
    j = 0
    for variable in numeric:
        if j <= j_limit:
            sns.histplot(x=variable, data=data, ax=axes[i,j], kde=True)
        else:
            j = 0
            i = i + 1
            sns.histplot(x=variable, data=data, ax=axes[i,j], kde=True)
        j = j + 1

def plot_correlation_matrix(data):
    plt.figure(figsize=(50,50))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.show()

def q_qplot(data):
    numeric = data.columns
    fig, axes = plt.subplots(int(len(numeric)/5)+1, 5, figsize=(25, 25))
    i_limit = int(len(numeric)/5) + 1
    j_limit = 4
    i = 0
    j = 0
    for variable in numeric:
        if j <= j_limit:
            sm.qqplot(data[variable], line='s', ax=axes[i,j])
            plt.title(f"Q-Q plot of {variable}")
        else:
            j = 0
            i = i + 1
            sm.qqplot(data[variable], line='s', ax=axes[i,j])
            plt.title(f"Q-Q plot of {variable}")
        j = j + 1
    plt.show()

def pca_get(data):
    X = data.iloc[:, :-2]  # Todas las columnas excepto las dos últimas (variables dependientes)
    # Escalado de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Aplicamos PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    # Analizamos la varianza explicada por los componentes principales
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    # Seleccionamos el número óptimo de componentes (por ejemplo, los que expliquen el 99% de la varianza)
    n_components = next(i for i, cum_var in enumerate(cumulative_variance) if cum_var >= 0.99)
    # Graficamos la varianza acumulada explicada
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance * 100, marker='o', linestyle='-', color='b')
    # Añadimos etiquetas y títulos
    plt.title('% de Varianza Acumulada Explicada por los Componentes Principales')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('% de Varianza Acumulada')
    plt.grid(True)
    # Mostrar el gráfico
    plt.show()
    print(f'El número de componentes para el dataset utilizado, que explican el 99% de la varianza es de: ' + str(n_components))
    return n_components