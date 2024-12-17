import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# Calculo de coordinalidad 
def cardi(df_in,umbral_categoria, umbral_continua):
    df_cardi= df_cardi = pd.DataFrame({
    "Cardi": df_in.nunique(),
    "% Cardi": df_in.nunique() / len(df_in) * 100
})
    clasificacion= []
    for index, valor in df_cardi["Cardi"].items():
        if valor==2:
            clasificacion.append("Binaria")
        elif valor < umbral_categoria:
            clasificacion.append("Categorica")
        elif valor >= umbral_categoria:
            if df_cardi.loc[index, "% Cardi"] >= umbral_continua:
                clasificacion.append("Numerica Continua")
            else:
                clasificacion.append("Numerica Discreta")
    df_cardi["Clasificacion"]=clasificacion
    return df_cardi

# Calculo distribuciones categóricas
def pinta_distribucion_categoricas(df, columnas_categoricas, relativa=False, mostrar_valores=False):
    num_columnas = len(columnas_categoricas)
    num_filas = (num_columnas // 2) + (num_columnas % 2)

    # Aumentar el tamaño de la figura
    fig, axes = plt.subplots(num_filas, 2, figsize=(20, 8 * num_filas))  # Ajuste de tamaño
    axes = axes.flatten()

    for i, col in enumerate(columnas_categoricas):
        ax = axes[i]
        if relativa:
            total = df[col].value_counts().sum()
            # Calcular la frecuencia relativa en porcentaje
            serie = df[col].value_counts().apply(lambda x: x / total * 100)  # Convertir a porcentaje
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='Pastel2', hue=serie.index, legend=False)
            ax.set_ylabel('Frecuencia Relativa (%)')  # Añadir el símbolo de porcentaje
        else:
            # Calcular la frecuencia absoluta
            serie = df[col].value_counts()
            sns.barplot(x=serie.index, y=serie, ax=ax, palette='Pastel2', hue=serie.index, legend=False)
            ax.set_ylabel('Frecuencia Absoluta')

        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)


        if mostrar_valores:
            for p in ax.patches:
                height = p.get_height()
                if relativa:
                    # Mostrar los valores como porcentaje sin decimales
                    ax.annotate(f'{height:.0f}%', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                else:
                    # Mostrar los valores absolutos como números enteros
                    ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Ocultar los subgráficos no utilizados
    for j in range(i + 1, num_filas * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
# Calculo histograma variables numericas
def plot_numerical_histograms(df, numerical_columns, bins_list, kde=False):
    num_cols = 2
    num_rows = (len(numerical_columns) + 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(numerical_columns):
        bins = bins_list[i] if i < len(bins_list) else 10
        sns.histplot(data=df, x=col, bins=bins, kde=kde, ax=axes[i])
        axes[i].set_title(f'Histograma de {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()    
# Boxplots de variables numericas
def plot_multiple_boxplots(df, columns, dim_matriz_visual = 2):
    num_cols = len(columns)
    num_rows = num_cols // dim_matriz_visual + num_cols % dim_matriz_visual
    fig, axes = plt.subplots(num_rows, dim_matriz_visual, figsize=(12, 6 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        if df[column].dtype in ['int64', 'float64']:
            sns.boxplot(data=df, x=column, ax=axes[i])
            axes[i].set_title(column)

    # Ocultar ejes vacíos
    for j in range(i+1, num_rows * 2):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

#Calculo de variabilidad
def variabilidad(df_perfil_turistico):
    # Seleccionar solo las columnas numéricas
    df_numericas = df_perfil_turistico.select_dtypes(include=["number"])
    # Calcular el resumen estadístico y seleccionar las filas de std y mean
    df_var = df_numericas.describe().loc[["std", "mean"]].T
    # Calcular el Coeficiente de Variación (CV)
    df_var["CV"] = df_var["std"] / df_var["mean"]
    return df_var

# Crear un boxplot de la calificación de la experiencia por tipo de alojamiento

def plot_calificacion_por_alojamiento(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tipo_alojamiento', y='Calificacion', data=df)
    plt.title('Distribución de la calificación de la experiencia por tipo de alojamiento')
    plt.xticks(rotation=45)
    plt.show()


#Función para graficar las 10 nacionalidades - noches 2019

def plot_duracion_promedio_por_nacionalidad_2019(duracion_promedio_df_2019):
    # Colores pastel personalizados para las barras
    colores = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', 
               '#E7BAFF', '#FFC8E2', '#C2F0FC', '#FFD6A5', '#F3D6FF']  
    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(duracion_promedio_df_2019['Nacionalidad'][:10], 
            duracion_promedio_df_2019['Duracion_Promedio_2019'][:10], 
            color=colores) 
    # Agregar etiquetas y título
    plt.xlabel('Nacionalidad')
    plt.ylabel('Duración Promedio (Noches)')
    plt.title('Duración Promedio de Estancia por Nacionalidad 2019')    
    # Rotar las etiquetas del eje X para mejor legibilidad
    plt.xticks(rotation=45)   
    # Mostrar gráfico
    plt.show()

#Función para graficar las 10 nacionalidades - noches 2022

def plot_duracion_promedio_por_nacionalidad_2022(duracion_promedio_df_2022):
    # Colores pastel personalizados para las barras
    colores = ['#FFB3BA', '#FFDFBA', '#FFFFBA', '#BAFFC9', '#BAE1FF', 
               '#E7BAFF', '#FFC8E2', '#C2F0FC', '#FFD6A5', '#F3D6FF']  
    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(duracion_promedio_df_2022['Nacionalidad'][:10], 
            duracion_promedio_df_2022['Duracion_Promedio_2022'][:10], 
            color=colores) 
    # Agregar etiquetas y título
    plt.xlabel('Nacionalidad')
    plt.ylabel('Duración Promedio (Noches)')
    plt.title('Duración Promedio de Estancia por Nacionalidad 2022')    
    # Rotar las etiquetas del eje X para mejor legibilidad
    plt.xticks(rotation=45)   
    # Mostrar gráfico
    plt.show()

#Turistas por nacionalidad y cuatrimestres
def graficar_turistas_por_nacionalidad_y_cuatrimestre(viaje_2019, viaje_2022):
    # Crear un gráfico de barras apiladas para 2019 y 2022
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    # Gráfico para 2019
    viaje_2019.plot(kind='bar', stacked=True, ax=ax[0], colormap='Set3')
    ax[0].set_title('Distribución de turistas por Nacionalidad y Cuatrimestre (2019)')
    ax[0].set_xlabel('Nacionalidad')
    ax[0].set_ylabel('Número de Turistas')
    ax[0].legend(title='Cuatrimestre', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Gráfico para 2022
    viaje_2022.plot(kind='bar', stacked=True, ax=ax[1], colormap='Set3')
    ax[1].set_title('Distribución de turistas por Nacionalidad y Cuatrimestre (2022)')
    ax[1].set_xlabel('Nacionalidad')
    ax[1].set_ylabel('Número de Turistas')
    ax[1].legend(title='Cuatrimestre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Gráfico de barras comparativo

def plot_comparacion_turistas_year(data_comparacion):
    # Crear el gráfico de barras
    plt.figure(figsize=(16, 12))
    barras = plt.bar(data_comparacion['Año'], data_comparacion['Total_Turistas'], color=['#FFDFBA', '#E7BAFF'])
    # Etiquetas y título del gráfico
    plt.xlabel('Año')
    plt.ylabel('Número de Turistas')
    plt.title('Comparación del Número de Turistas: 2019 vs 2022')
    # Añadir etiquetas con los valores encima de las barras
    for barra in barras:
        altura = barra.get_height()  # Obtener la altura de cada barra
        plt.text(barra.get_x() + barra.get_width() / 2, altura + 200,  # Ajuste de posición
                 f'{int(altura)}', ha='center', va='bottom', fontsize=10, color='black')

    # Mostrar el gráfico
    plt.show()

import matplotlib.pyplot as plt

#Función para graficar el número de turistas y la duración promedio de la estancia
#por cuatrimestre y año en gráficos de barras.
def plot_turistas_y_duracion_por_cuatrimestre(num_turistas_cuatrimestre_anio, duracion_promedio_cuatrimestre_anio):
    # Crear los subgráficos
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # Gráfico para número de turistas
    num_turistas_cuatrimestre_anio.plot(kind='bar', ax=ax[0], colormap='Pastel1')
    ax[0].set_xlabel('Cuatrimestre')
    ax[0].set_ylabel('Número de Turistas')
    ax[0].set_title('Número de Turistas por Cuatrimestre y Año')
    ax[0].legend(title="Año", bbox_to_anchor=(1, 1))  # Coloca la leyenda al lado derecho
    # Gráfico para duración promedio
    duracion_promedio_cuatrimestre_anio.plot(kind='bar', ax=ax[1], colormap='Pastel2')
    ax[1].set_xlabel('Cuatrimestre')
    ax[1].set_ylabel('Duración Promedio (Noches)')
    ax[1].set_title('Duración Promedio de la Estancia por Cuatrimestre y Año')
    ax[1].legend(title="Año", bbox_to_anchor=(1, 1))  # Coloca la leyenda al lado derecho
    # Ajustar el espacio entre los gráficos
    plt.tight_layout()
    # Mostrar los gráficos
    plt.show()

#Función para graficar el tipo de alojamiento en 2019 y 2022 por cuatrimestre
#usando gráficos de barras apiladas para cada año.

def plot_tipo_alojamiento_por_cuatrimestre(preferencia_alojamiento_2019, preferencia_alojamiento_2022):
    # Crear los subgráficos
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # Gráfico de barras apiladas para 2019
    preferencia_alojamiento_2019.plot(kind='bar', stacked=True, ax=ax[0], colormap='Pastel1')
    ax[0].set_xlabel('Cuatrimestre')
    ax[0].set_ylabel('Número de Turistas')
    ax[0].set_title('Tipo de Alojamiento en 2019 por Cuatrimestre')
    ax[0].legend(title="Tipo de Alojamiento", bbox_to_anchor=(1, 1))  # Coloca la leyenda a la derecha
    # Gráfico de barras apiladas para 2022
    preferencia_alojamiento_2022.plot(kind='bar', stacked=True, ax=ax[1], colormap='Pastel2')
    ax[1].set_xlabel('Cuatrimestre')
    ax[1].set_ylabel('Número de Turistas')
    ax[1].set_title('Tipo de Alojamiento en 2022 por Cuatrimestre')
    ax[1].legend(title="Tipo de Alojamiento", bbox_to_anchor=(1, 1))  # Coloca la leyenda a la derecha

    # Ajustar el espacio entre los gráficos
    plt.tight_layout()
    # Mostrar el gráfico
    plt.show()

"""    Función para graficar la recomendación del destino por nacionalidad en 2022
    usando un gráfico de barras apiladas."""

def plot_recomendacion_por_nacionalidad(recomendacion_nacionalidad_2022):
    # Crear el gráfico de barras apiladas
    plt.figure(figsize=(12, 6))
    recomendacion_nacionalidad_2022.plot(kind='bar', stacked=True, colormap='coolwarm')
    # Ajustes para mejorar la visualización
    plt.xlabel('Nacionalidad')
    plt.ylabel('Número de Turistas')
    plt.title('Recomendación del destino por Nacionalidad en 2022')
    plt.xticks(rotation=90)  # Rotar etiquetas del eje X para mayor legibilidad
    plt.legend(title='Recomendación', bbox_to_anchor=(1, 1))  # Colocar la leyenda a la derecha
    plt.tight_layout()  # Ajuste de los elementos para evitar solapamientos
    # Mostrar el gráfico
    plt.show()

#Graficar el perfil sociodemográfico 
import matplotlib.pyplot as plt

def perfil_sociodemografico(edad_promedio_nacionalidad_ano, 
                                     educacion_nacionalidad_ano, laboral_nacionalidad_ano, 
                                     personas_hogar_nacionalidad_ano):

    # Gráfico de Edad Promedio por Nacionalidad y Año
    plt.figure(figsize=(16, 8))
    edad_promedio_nacionalidad_ano.plot(kind='bar', color=['#FFDFBA', '#E7BAFF'], width=0.8)
    plt.xlabel('Nacionalidad')
    plt.ylabel('Edad Promedio')
    plt.title('Edad Promedio por Nacionalidad y Año')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Gráfico de Distribución de Nivel Educativo por Nacionalidad y Año
    educacion_nacionalidad_ano.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Pastel2')
    plt.xlabel('Nacionalidad')
    plt.ylabel('Número de Turistas')
    plt.title('Distribución de Nivel Educativo por Nacionalidad y Año (2019 vs 2022)')
    plt.xticks(rotation=90)
    plt.legend(title="Nivel Educativo", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    # Gráfico de Distribución de Situación Laboral por Nacionalidad y Año
    laboral_nacionalidad_ano.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='Pastel2')
    plt.xlabel('Nacionalidad')
    plt.ylabel('Número de Turistas')
    plt.title('Distribución de Situación Laboral por Nacionalidad y Año (2019 vs 2022)')
    plt.xticks(rotation=90)
    plt.legend(title="Situación Laboral", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    # Gráfico de Número Promedio de Personas en el Hogar por Nacionalidad y Año
    plt.figure(figsize=(12, 6))
    personas_hogar_nacionalidad_ano.plot(kind='bar', color=['#FF6347', '#87CEFA'], width=0.8)
    plt.xlabel('Nacionalidad')
    plt.ylabel('Número Promedio de Personas en el Hogar')
    plt.title('Número Promedio de Personas en el Hogar por Nacionalidad y Año (2019 vs 2022)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
#Nacionalidad por sexo 2022
def grafico_sexo_por_nacionalidad_2022(df, año=2022):
    # Filtrar los datos solo para el año 2022
    df_2022 = df[df['Año'] == año]
    
    # Crear tabla de contingencia para Nacionalidad y Sexo
    sexo_nacionalidad_2022 = pd.crosstab(df_2022['Nacionalidad'], df_2022['Sexo'])
    
    # Crear gráfico de barras apiladas
    sexo_nacionalidad_2022.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='Pastel2')

    # Ajustes de etiquetas y título
    plt.xlabel('Nacionalidad')
    plt.ylabel('Número de Turistas')
    plt.title(f'Distribución de Sexo por Nacionalidad ({año})')
    plt.xticks(rotation=90)
    plt.legend(title="Sexo", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Ajuste del layout para evitar superposiciones
    plt.tight_layout()
    plt.show()

#Matriz de correlaciones
def matriz_correlacion(correlation_matrix):  
    # Configuración del gráfico
    plt.figure(figsize=(10, 8))  
    # Generar el mapa de calor
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1) 
    # Título del gráfico
    plt.title('Matriz de correlación')   
    # Mostrar el gráfico
    plt.show()

# gráfico de dispersión (scatter plot) de dos variables
def graficar_relacion_visitas(df, x_col, y_col, titulo, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
#Gráfica de la relación del Alojamiento y las personas del hogar
def boxplot_alojamiento_hogar(df, x_col, y_col, titulo, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

#gráfico de dispersión (pairplot) entre todas las variables numéricas de un DataFrame.
def graficar_dispersión_variables(df, variables_numericas):
    plt.figure(figsize=(12, 8))
    sns.pairplot(df[variables_numericas])
    plt.show()


#PCA
def graficar_PCA(pca_df, pca):
    # Graficar los componentes principales
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, color='purple')
    plt.title('Análisis de Componentes Principales (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

#Clusters
def graficar_clusters(df, variable_x, variable_y, cluster_column):
    # Graficar los clusters usando un gráfico de dispersión
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[variable_x], y=df[variable_y], hue=df[cluster_column], palette='Set2', s=80) 
    # Añadir título y etiquetas
    plt.title(f'Segmentación de Turistas por Clustering (K-means) - {variable_x} vs {variable_y}')
    plt.xlabel(variable_x)
    plt.ylabel(variable_y)  
    # Ajustar la leyenda
    plt.legend(title="Cluster", bbox_to_anchor=(1, 1))
    # Ajustar layout y mostrar gráfico
    plt.tight_layout()
    plt.show()

# Gráfico clusters y PCA
def graficar_clusters_pca(pca_df, df, cluster_column):
    # Graficar los clusters en función de los componentes principales
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=df[cluster_column], palette='Set2', s=80)
    # Añadir título y etiquetas
    plt.title('Segmentación de Turistas por Clustering (K-means)')
    plt.xlabel('Componente Principal 1 (PC1)')
    plt.ylabel('Componente Principal 2 (PC2)')
    # Ajustar la leyenda
    plt.legend(title="Cluster", bbox_to_anchor=(1, 1))
    # Ajustar layout y mostrar gráfico
    plt.tight_layout()
    plt.show()
