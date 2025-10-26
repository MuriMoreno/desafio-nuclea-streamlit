import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuração ---
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# Função para carregar e preparar os dados
def load_and_prepare_data():
    # 1. Carregar os dados
    try:
        # ATENÇÃO: Ajuste os caminhos abaixo se os arquivos CSV não estiverem neste local
        df_boletos = pd.read_csv('base_boletos_fiap(in).csv', sep=',')
        df_auxiliar = pd.read_csv('base_auxiliar_fiap(in).csv', sep=',')
    except Exception as e:
        print(f"Erro ao carregar os arquivos. Verifique se os caminhos estão corretos: {e}")
        return None

    # Renomear coluna para merge
    df_auxiliar.rename(columns={'id_cnpj': 'id_pagador'}, inplace=True)

    # Conversão de colunas de data
    date_cols = ['dt_emissao', 'dt_vencimento', 'dt_pagamento']
    for col in date_cols:
        df_boletos[col] = pd.to_datetime(df_boletos[col], errors='coerce')

    # Preenchimento de valores faltantes e merge
    df_boletos['tipo_baixa'] = df_boletos['tipo_baixa'].fillna('Em Aberto')
    df_boletos['vlr_baixa'] = df_boletos['vlr_baixa'].fillna(0)
    df_merged = pd.merge(df_boletos, df_auxiliar, on='id_pagador', how='left')

    # Criação de Variáveis Chave
    def get_payment_status(row):
        if row['tipo_baixa'] == 'Em Aberto':
            return 'Em Aberto'
        if row['dt_pagamento'] > row['dt_vencimento']:
            return 'Pago Atrasado'
        return 'Pago em Dia'

    df_merged['status_pagamento'] = df_merged.apply(get_payment_status, axis=1)
    df_merged['dias_atraso'] = (df_merged['dt_pagamento'] - df_merged['dt_vencimento']).dt.days
    df_merged['dias_atraso'] = df_merged['dias_atraso'].apply(lambda x: x if x > 0 else 0)
    df_merged['inadimplente'] = df_merged['status_pagamento'].apply(lambda x: 1 if x == 'Em Aberto' else 0)
    
    return df_merged

df = load_and_prepare_data()

if df is None:
    exit()

# Agregação por Pagador para obter a Taxa de Inadimplência e Indicadores
df_pagador_agg = df.groupby('id_pagador').agg(
    taxa_inadimplencia=('inadimplente', 'mean'),
    score_materialidade=('score_materialidade_v2', 'mean'),
    score_quantidade=('score_quantidade_v2', 'mean'),
    media_atraso_dias_aux=('media_atraso_dias', 'mean'),
    share_vl_inad_pag_bol_6_a_15d=('share_vl_inad_pag_bol_6_a_15d', 'mean'),
    indicador_liquidez_quantitativo_3m=('indicador_liquidez_quantitativo_3m', 'mean'),
    score_materialidade_evolucao=('score_materialidade_evolucao', 'mean'),
    sacado_indice_liquidez_1m=('sacado_indice_liquidez_1m', 'mean'),
    cedente_indice_liquidez_1m=('cedente_indice_liquidez_1m', 'mean')
).reset_index()

# Criar indicador de risco para Boxplots
df_pagador_agg['alto_risco'] = df_pagador_agg['taxa_inadimplencia'].apply(lambda x: 'Alto Risco' if x > 0 else 'Baixo Risco')


# --- ANÁLISE EXPLORATÓRIA INICIAL (EDA) ---
print("\n--- ANÁLISE EXPLORATÓRIA INICIAL (EDA) ---")

# Análise 1: Distribuição do Status de Pagamento
status_counts = df['status_pagamento'].value_counts(normalize=True) * 100
print("\nDistribuição do Status de Pagamento (%):")
print(status_counts)

# Análise 5: Relação entre Scores e Inadimplência (para correlação)
correlation_eda = df_pagador_agg[['taxa_inadimplencia', 'score_materialidade', 'score_quantidade', 'media_atraso_dias_aux']].corr()
print("\nCorrelação entre Scores do Pagador e Taxa de Inadimplência (EDA Inicial):")
print(correlation_eda)


# --- ANÁLISES DE APROFUNDAMENTO (Outliers, Temporal, CNAE) ---
print("\n--- ANÁLISES DE APROFUNDAMENTO (Outliers, Temporal, CNAE) ---")

# 1. Análise de Outliers (vlr_nominal)
p99_nominal = df['vlr_nominal'].quantile(0.99)
df_outliers = df[df['vlr_nominal'] > p99_nominal]
outlier_inadimplencia = df_outliers['inadimplente'].mean() * 100
taxa_geral_inadimplencia = df['inadimplente'].mean() * 100
print(f"\n1. Análise de Outliers (Vlr Nominal > R$ {p99_nominal:,.2f}):")
print(f"   Taxa de Inadimplência nos Outliers: {outlier_inadimplencia:.2f}% (Geral: {taxa_geral_inadimplencia:.2f}%)")

# 2. Análise Temporal (Evolução da Inadimplência)
df['mes_emissao'] = df['dt_emissao'].dt.to_period('M')
inadimplencia_mensal = df.groupby('mes_emissao')['inadimplente'].mean() * 100
print("\n2. Análise Temporal (Inadimplência Mensal - Top 5):")
print(inadimplencia_mensal.sort_values(ascending=False).head())

# 3. Impacto do CNAE
df['cnae_4digitos'] = df['cd_cnae_prin'].astype(str).str[:4]
cnae_counts = df['cnae_4digitos'].value_counts()
cnae_validos = cnae_counts[cnae_counts >= 50].index
df_cnae = df[df['cnae_4digitos'].isin(cnae_validos)]
inadimplencia_por_cnae = df_cnae.groupby('cnae_4digitos')['inadimplente'].mean() * 100
print("\n3. Impacto do CNAE (Top 5 Inadimplência):")
print(inadimplencia_por_cnae.sort_values(ascending=False).head())


# --- ANÁLISES DE APROFUNDAMENTO (Indicadores de Risco e Liquidez) ---
print("\n--- ANÁLISES DE APROFUNDAMENTO (Indicadores de Risco e Liquidez) ---")

# 4. Análise de Indicadores de Risco Adicionais
cols_risco = ['taxa_inadimplencia', 'share_vl_inad_pag_bol_6_a_15d', 'indicador_liquidez_quantitativo_3m', 'score_materialidade_evolucao']
correlation_risco = df_pagador_agg[cols_risco].corr()
print("\n4. Correlação com Indicadores de Risco Adicionais:")
print(correlation_risco['taxa_inadimplencia'].sort_values(ascending=False))

# 5. Análise de Indicadores de Liquidez de 1 Mês
cols_liquidez = ['taxa_inadimplencia', 'sacado_indice_liquidez_1m', 'cedente_indice_liquidez_1m']
correlation_liquidez = df_pagador_agg[cols_liquidez].corr()
print("\n5. Correlação com Indicadores de Liquidez de 1 Mês:")
print(correlation_liquidez['taxa_inadimplencia'].sort_values(ascending=False))

# Análise de Casos Extremos
df_pagador_agg['sacado_liquidez_baixa'] = df_pagador_agg['sacado_indice_liquidez_1m'] < 0.5
df_extremos = df_pagador_agg[(df_pagador_agg['sacado_liquidez_baixa'] == True) & (df_pagador_agg['alto_risco'] == 'Baixo Risco')]
print(f"\nNúmero de Pagadores com Liquidez Baixa (< 50%) mas Baixo Risco (0% Inadimplência): {len(df_extremos)}")


# --- GERAÇÃO DE VISUALIZAÇÕES (TODOS OS GRÁFICOS) ---
print("\n--- GERAÇÃO DE VISUALIZAÇÕES ---")

# 1. Distribuição do Status de Pagamento
plt.figure(figsize=(8, 6))
sns.barplot(x=status_counts.index, y=status_counts.values, palette="viridis")
plt.title('Distribuição do Status de Pagamento dos Boletos')
plt.ylabel('Percentual (%)')
plt.xlabel('Status de Pagamento')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/ubuntu/status_pagamento_distribuicao.png')
plt.close()

# 2. Top 10 Tipos de Espécie com Maior Potencial de Inadimplência
inadimplencia_por_especie = df.groupby('tipo_especie')['inadimplente'].mean().sort_values(ascending=False) * 100
plt.figure(figsize=(10, 6))
sns.barplot(x=inadimplencia_por_especie.head(10).index, y=inadimplencia_por_especie.head(10).values, palette="rocket")
plt.title('Top 10 Tipos de Espécie com Maior Potencial de Inadimplência')
plt.ylabel('Percentual de Boletos "Em Aberto" (%)')
plt.xlabel('Tipo de Espécie')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/ubuntu/inadimplencia_por_especie.png')
plt.close()

# 3. Scatter Plot: Score Materialidade vs Taxa de Inadimplência
plt.figure(figsize=(10, 6))
sns.scatterplot(x='score_materialidade', y='taxa_inadimplencia', data=df_pagador_agg)
plt.title('Score Materialidade vs. Taxa de Inadimplência (por Pagador)')
plt.xlabel('Score Materialidade V2 (Média por Pagador)')
plt.ylabel('Taxa de Inadimplência (Boletos "Em Aberto")')
plt.tight_layout()
plt.savefig('/home/ubuntu/score_materialidade_vs_inadimplencia.png')
plt.close()

# 4. Boxplot para vlr_nominal (com zoom no corpo principal)
plt.figure(figsize=(8, 6))
sns.boxplot(y=df['vlr_nominal'])
plt.ylim(0, df['vlr_nominal'].quantile(0.75) * 5)
plt.title('Boxplot do Valor Nominal (Zoom)')
plt.ylabel('Valor Nominal (R$)')
plt.tight_layout()
plt.savefig('/home/ubuntu/boxplot_vlr_nominal_zoom.png')
plt.close()

# 5. Evolução da Taxa de Inadimplência
inadimplencia_mensal_plot = inadimplencia_mensal.reset_index()
inadimplencia_mensal_plot['mes_emissao'] = inadimplencia_mensal_plot['mes_emissao'].astype(str)
plt.figure(figsize=(12, 6))
sns.lineplot(x='mes_emissao', y='inadimplente', data=inadimplencia_mensal_plot, marker='o')
plt.title('Evolução Mensal da Taxa de Boletos "Em Aberto" (Potencial Inadimplência)')
plt.ylabel('Taxa de Inadimplência (%)')
plt.xlabel('Mês de Emissão')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/ubuntu/inadimplencia_temporal.png')
plt.close()

# 6. Inadimplência por CNAE
inadimplencia_por_cnae_plot = inadimplencia_por_cnae.sort_values(ascending=False).head(10).reset_index()
plt.figure(figsize=(12, 6))
sns.barplot(x='cnae_4digitos', y='inadimplente', data=inadimplencia_por_cnae_plot, palette="cubehelix")
plt.title('Top 10 CNAEs (4 Dígitos) com Maior Taxa de Potencial Inadimplência')
plt.ylabel('Taxa de Inadimplência (%)')
plt.xlabel('CNAE (4 Primeiros Dígitos)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/ubuntu/inadimplencia_por_cnae.png')
plt.close()

# 7. Boxplots Indicadores de Risco Adicionais
analysis_cols_risco = ['share_vl_inad_pag_bol_6_a_15d', 'indicador_liquidez_quantitativo_3m', 'score_materialidade_evolucao']
for col in analysis_cols_risco:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='alto_risco', y=col, data=df_pagador_agg.dropna(subset=[col]))
    plt.title(f'Distribuição de {col} por Grupo de Risco de Inadimplência')
    plt.savefig(f'/home/ubuntu/boxplot_{col}_by_risk.png')
    plt.close()

# 8. Boxplots Indicadores de Liquidez de 1 Mês
analysis_cols_liquidez = ['sacado_indice_liquidez_1m', 'cedente_indice_liquidez_1m']
for col in analysis_cols_liquidez:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='alto_risco', y=col, data=df_pagador_agg.dropna(subset=[col]))
    plt.title(f'Distribuição de {col} por Grupo de Risco de Inadimplência')
    plt.savefig(f'/home/ubuntu/boxplot_{col}_by_risk.png')
    plt.close()

print("\nScript de análise completa final gerado com sucesso. Todos os gráficos foram salvos.")
