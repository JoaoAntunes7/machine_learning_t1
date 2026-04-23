# Funções pra visualização dos dados
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def report_outliers_iqr(df, columns=None, group_col=None, multiplier=1.5):
    """
    Detecta outliers usando a regra do IQR.

    - columns: lista de colunas numéricas (se None, detecta automaticamente)
    - group_col: coluna opcional para calcular outliers por grupo (ex: burnout_level)
    - multiplier: fator da regra do boxplot (default 1.5)

    Retorna um DataFrame resumo com contagem e percentual de outliers.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        # Evita analisar identificadores como variáveis contínuas.
        columns = [c for c in columns if c.lower() not in {'id', 'student_id'} and not c.lower().endswith('_id')]

    if group_col and group_col in columns:
        columns.remove(group_col)

    summary_rows = []

    def _calc_outliers(series):
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        n = len(numeric_series)

        if n == 0:
            return np.nan, np.nan, np.nan, np.nan, 0, 0, 0.0

        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outlier_mask = (numeric_series < lower) | (numeric_series > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_pct = (outlier_count / n) * 100

        return q1, q3, iqr, lower, upper, outlier_count, outlier_pct

    if group_col and group_col in df.columns:
        for group_name, group_df in df.groupby(group_col):
            for col in columns:
                q1, q3, iqr, lower, upper, outlier_count, outlier_pct = _calc_outliers(group_df[col])
                summary_rows.append({
                    'group': group_name,
                    'column': col,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'lower_bound': lower,
                    'upper_bound': upper,
                    'outlier_count': outlier_count,
                    'outlier_pct': outlier_pct,
                })
    else:
        for col in columns:
            q1, q3, iqr, lower, upper, outlier_count, outlier_pct = _calc_outliers(df[col])
            summary_rows.append({
                'group': 'all_data',
                'column': col,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower,
                'upper_bound': upper,
                'outlier_count': outlier_count,
                'outlier_pct': outlier_pct,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['group', 'outlier_pct', 'outlier_count'], ascending=[True, False, False])

    print(f"\n{'='*80}")
    print('RESUMO DE OUTLIERS (REGRA IQR)')
    print(f"{'='*80}")
    print(summary_df[['group', 'column', 'outlier_count', 'outlier_pct']].to_string(index=False))

    return summary_df


def plot_burnout_distribution(df):
    """
    Visualiza a distribuição das classes de burnout.
    Mostra contagem e porcentagem de cada classe.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contagem absoluta
    burnout_counts = df['burnout_level'].value_counts()
    ax1.bar(burnout_counts.index, burnout_counts.values, color=['#ff6b6b', '#ffd93d', '#6bcf7f'])
    ax1.set_title('Distribuição de Burnout (Contagem)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequência')
    ax1.grid(axis='y', alpha=0.3)
    
    # Porcentagem
    burnout_pct = df['burnout_level'].value_counts(normalize=True) * 100
    ax2.pie(burnout_pct.values, labels=burnout_pct.index, autopct='%1.1f%%',
            colors=['#ff6b6b', '#ffd93d', '#6bcf7f'])
    ax2.set_title('Distribuição de Burnout (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_numerical_by_burnout(df, figsize=(16, 12)):
    """
    Box plots de todos os atributos numéricos por classe de burnout.
    Ideal para ver dispersão (IQR, mediana, outliers) e comparar entre classes.
    """
    # Atributos numéricos principais
    numerical_cols = ['age', 'daily_study_hours', 'daily_sleep_hours', 'screen_time_hours',
                      'anxiety_score', 'depression_score', 
                      'academic_pressure_score', 'financial_stress_score', 
                      'social_support_score', 'physical_activity_hours',
                      'attendance_percentage', 'cgpa']
    
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(numerical_cols):
        sns.boxplot(data=df, x='burnout_level', y=col, ax=axes[idx],
                   palette=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        axes[idx].set_title(f'{col.replace("_", " ").title()} vs Burnout', 
                           fontsize=10, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Remove eixos vazios
    for idx in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


def plot_violin_comparison(df, attribute, figsize=(10, 6)):
    """
    Violin plot para um atributo específico.
    Mostra distribuição completa + box plot + pontos de dados.
    Melhor para entender bem a dispersão.
    """
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x='burnout_level', y=attribute,
                  palette=['#ff6b6b', '#ffd93d', '#6bcf7f'], inner='box')
    plt.title(f'Distribuição de {attribute.replace("_", " ").title()} por Classe de Burnout',
             fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_dispersion_statistics(df):
    """
    Calcula e exibe medidas de dispersão por classe de burnout.
    Mostra: Min, Q1, Mediana, Q3, Max, IQR, Desvio Padrão
    """
    numerical_cols = ['age', 'daily_study_hours', 'daily_sleep_hours', 'screen_time_hours',
                      'anxiety_score', 'depression_score', 
                      'academic_pressure_score', 'financial_stress_score', 
                      'social_support_score', 'physical_activity_hours',
                      'attendance_percentage', 'cgpa']
    
    burnout_classes = df['burnout_level'].unique()
    
    for col in numerical_cols:
        print(f"\n{'='*80}")
        print(f"MEDIDAS DE DISPERSÃO: {col.upper()}")
        print(f"{'='*80}")
        
        for burnout in sorted(burnout_classes):
            subset = df[df['burnout_level'] == burnout][col]
            subset = pd.to_numeric(subset, errors='coerce').dropna()

            if subset.empty:
                print(f"\n{burnout}:")
                print("  Sem dados numéricos válidos para cálculo.")
                continue
            
            q1 = subset.quantile(0.25)
            q2 = subset.quantile(0.50)  # mediana
            q3 = subset.quantile(0.75)
            iqr = q3 - q1
            
            print(f"\n{burnout}:")
            print(f"  Min:          {subset.min():.2f}")
            print(f"  Q1 (25%):     {q1:.2f}")
            print(f"  Mediana:      {q2:.2f}")
            print(f"  Q3 (75%):     {q3:.2f}")
            print(f"  Max:          {subset.max():.2f}")
            print(f"  IQR:          {iqr:.2f}  (Q3 - Q1)")
            print(f"  Desvio Padr.: {subset.std():.2f}")
            print(f"  Variância:    {subset.var():.2f}")
            print(f"  Amplitude:    {subset.max() - subset.min():.2f}")


def plot_scatter_by_burnout(df, x_attr, y_attr, figsize=(10, 7)):
    """
    Scatter plot de dois atributos colorido por classe de burnout.
    Bom para ver padrões e separabilidade das classes.
    """
    plt.figure(figsize=figsize)
    
    for burnout in sorted(df['burnout_level'].unique()):
        subset = df[df['burnout_level'] == burnout]
        colors = {'High': '#ff6b6b', 'Low': '#6bcf7f', 'Medium': '#ffd93d'}
        plt.scatter(subset[x_attr], subset[y_attr], 
                   label=f'Burnout: {burnout}', 
                   alpha=0.6, s=100, color=colors.get(burnout, 'gray'))
    
    plt.xlabel(x_attr.replace('_', ' ').title(), fontsize=11)
    plt.ylabel(y_attr.replace('_', ' ').title(), fontsize=11)
    plt.title(f'{x_attr.replace("_", " ").title()} vs {y_attr.replace("_", " ").title()} by Burnout',
             fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Heatmap de correlação de atributos numéricos.
    Mostra quais variáveis estão correlacionadas (útil para feature selection).
    """
    numerical_cols = ['age', 'daily_study_hours', 'daily_sleep_hours', 'screen_time_hours',
                      'anxiety_score', 'depression_score', 
                      'academic_pressure_score', 'financial_stress_score', 
                      'social_support_score', 'physical_activity_hours',
                      'attendance_percentage', 'cgpa']
    
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, cbar_kws={'label': 'Correlação'})
    plt.title('Matriz de Correlação - Atributos Numéricos', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_class_comparison_heatmap(df, figsize=(12, 8)):
    """
    Heatmap mostrando média dos atributos por classe de burnout.
    Bom para ver padrões que diferenciam as classes.
    """
    numerical_cols = ['age', 'daily_study_hours', 'daily_sleep_hours', 'screen_time_hours',
                      'anxiety_score', 'depression_score', 
                      'academic_pressure_score', 'financial_stress_score', 
                      'social_support_score', 'physical_activity_hours',
                      'attendance_percentage', 'cgpa']
    
    # Calcula média por classe, depois normaliza para visualizar melhor
    class_means = df.groupby('burnout_level')[numerical_cols].mean()
    
    # Normaliza para escala 0-1 por coluna
    class_means_normalized = (class_means - class_means.min()) / (class_means.max() - class_means.min())
    
    plt.figure(figsize=figsize)
    sns.heatmap(class_means_normalized.T, annot=class_means.T, fmt='.2f', 
               cmap='RdYlGn', cbar_kws={'label': 'Intensidade (normalizada)'})
    plt.title('Padrões de Atributos por Classe de Burnout\n(Valores mostrados, cor indica intensidade normalizada)',
             fontsize=12, fontweight='bold')
    plt.xlabel('Classe de Burnout')
    plt.ylabel('Atributos')
    plt.tight_layout()
    plt.show()


def plot_histogram_by_class(df, attribute, bins=30, figsize=(12, 5)):
    """
    Histogramas sobrepostos de um atributo para cada classe.
    Mostra como as distribuições diferem entre classes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'High': '#ff6b6b', 'Low': '#6bcf7f', 'Medium': '#ffd93d'}
    
    for burnout in sorted(df['burnout_level'].unique()):
        subset = df[df['burnout_level'] == burnout][attribute]
        ax.hist(subset, bins=bins, alpha=0.5, label=f'Burnout: {burnout}',
               color=colors.get(burnout, 'gray'))
    
    ax.set_xlabel(attribute.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Frequência')
    ax.set_title(f'Distribuição de {attribute.replace("_", " ").title()} por Classe de Burnout',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()