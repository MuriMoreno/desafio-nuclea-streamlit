import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# --- Configuração da página ---
st.set_page_config(
    page_title="Relatório EDA - Desafio Fiap/Nuclea",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Função para carregar e preparar os dados ---
@st.cache_data
def load_and_prepare_data():
    try:
        df_boletos = pd.read_csv('base_boletos_fiap(in).csv', sep=',')
        df_auxiliar = pd.read_csv('base_auxiliar_fiap(in).csv', sep=',')
    except Exception as e:
        st.error(f"Erro ao carregar os arquivos: {e}")
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

# Carregar dados
df = load_and_prepare_data()

if df is None:
    st.stop()

# --- Sidebar ---
st.sidebar.title("📋 Navegação")
page = st.sidebar.radio(
    "Selecione uma seção:",
    ["🏠 Home", "📊 Análise Exploratória", "🔍 Análises de Aprofundamento", 
     "⚠️ Indicadores de Risco", "💧 Indicadores de Liquidez", "📈 Dados Detalhados"]
)

# --- HOME ---
if page == "🏠 Home":
    st.title("📊 Relatório de Análise Exploratória de Dados (EDA)")
    st.markdown("### Desafio Fiap/Nuclea - Análise de Boletos e Risco de Inadimplência")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Boletos", f"{len(df):,}")
    with col2:
        st.metric("Taxa de Inadimplência", f"{(df['inadimplente'].mean() * 100):.2f}%")
    with col3:
        st.metric("Valor Total Nominal", f"R$ {df['vlr_nominal'].sum():,.2f}")
    
    st.markdown("---")
    
    st.markdown("""
    ## 📌 Sobre este Relatório
    
    Este relatório apresenta uma **Análise Exploratória de Dados (EDA)** completa sobre as bases de dados de boletos fornecidas pelo Desafio Fiap/Nuclea.
    
    ### 🎯 Objetivos da Análise
    - Inspecionar a qualidade dos dados
    - Identificar padrões de pagamento e inadimplência
    - Avaliar indicadores de risco e liquidez
    - Fornecer insights para gestão de risco do FIDC
    
    ### 📂 Seções Disponíveis
    1. **Análise Exploratória**: Distribuição de pagamentos, valores e tipos de boletos
    2. **Análises de Aprofundamento**: Outliers, análise temporal e impacto do CNAE
    3. **Indicadores de Risco**: Análise de scores e indicadores de risco adicionais
    4. **Indicadores de Liquidez**: Análise de liquidez de 1 mês (sacado e cedente)
    5. **Dados Detalhados**: Visualização e download dos dados processados
    
    ### 📊 Principais Descobertas
    - **69.70%** dos boletos foram pagos em dia
    - **29.32%** dos boletos foram pagos com atraso
    - **0.98%** dos boletos estão em aberto (potencial inadimplência)
    - Boletos de **alto valor** (> R$ 314 mil) têm risco **3x maior**
    - O setor de **Restaurantes** e **Fabricação de Plástico** apresentam maior risco
    """)

# --- ANÁLISE EXPLORATÓRIA ---
elif page == "📊 Análise Exploratória":
    st.title("📊 Análise Exploratória de Dados (EDA)")
    
    st.markdown("---")
    
    # Seção 1: Distribuição do Status de Pagamento
    st.subheader("1️⃣ Distribuição do Status de Pagamento")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        status_counts = df['status_pagamento'].value_counts(normalize=True) * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=status_counts.index, y=status_counts.values, palette="viridis", ax=ax)
        ax.set_title('Distribuição do Status de Pagamento dos Boletos')
        ax.set_ylabel('Percentual (%)')
        ax.set_xlabel('Status de Pagamento')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    with col2:
        st.metric("Pago em Dia", f"{status_counts['Pago em Dia']:.2f}%")
        st.metric("Pago Atrasado", f"{status_counts['Pago Atrasado']:.2f}%")
        st.metric("Em Aberto", f"{status_counts['Em Aberto']:.2f}%")
    
    st.markdown("---")
    
    # Seção 2: Análise de Valores
    st.subheader("2️⃣ Análise de Valores Nominais e de Baixa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Estatísticas Descritivas:**")
        stats_df = pd.DataFrame({
            'Métrica': ['Média', 'Desvio Padrão', 'Mínimo', 'Mediana', 'Máximo'],
            'Valor Nominal (R$)': [
                f"{df['vlr_nominal'].mean():,.2f}",
                f"{df['vlr_nominal'].std():,.2f}",
                f"{df['vlr_nominal'].min():,.2f}",
                f"{df['vlr_nominal'].median():,.2f}",
                f"{df['vlr_nominal'].max():,.2f}"
            ],
            'Valor de Baixa (R$)': [
                f"{df['vlr_baixa'].mean():,.2f}",
                f"{df['vlr_baixa'].std():,.2f}",
                f"{df['vlr_baixa'].min():,.2f}",
                f"{df['vlr_baixa'].median():,.2f}",
                f"{df['vlr_baixa'].max():,.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.write("**Distribuição de Valores:**")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['vlr_nominal'], bins=50, alpha=0.7, label='Valor Nominal', edgecolor='black')
        ax.set_xlabel('Valor (R$)')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Valores Nominais')
        ax.legend()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Seção 3: Inadimplência por Tipo de Espécie
    st.subheader("3️⃣ Inadimplência por Tipo de Espécie")
    
    inadimplencia_por_especie = df.groupby('tipo_especie')['inadimplente'].mean().sort_values(ascending=False) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=inadimplencia_por_especie.head(10).index, y=inadimplencia_por_especie.head(10).values, palette="rocket", ax=ax)
    ax.set_title('Top 10 Tipos de Espécie com Maior Potencial de Inadimplência')
    ax.set_ylabel('Percentual de Boletos "Em Aberto" (%)')
    ax.set_xlabel('Tipo de Espécie')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# --- ANÁLISES DE APROFUNDAMENTO ---
elif page == "🔍 Análises de Aprofundamento":
    st.title("🔍 Análises de Aprofundamento")
    
    st.markdown("---")
    
    # Seção 1: Outliers
    st.subheader("1️⃣ Análise de Outliers (Alto Valor)")
    
    p99_nominal = df['vlr_nominal'].quantile(0.99)
    df_outliers = df[df['vlr_nominal'] > p99_nominal]
    outlier_inadimplencia = df_outliers['inadimplente'].mean() * 100
    taxa_geral = df['inadimplente'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("99º Percentil (Vlr Nominal)", f"R$ {p99_nominal:,.2f}")
    with col2:
        st.metric("Taxa de Inadimplência (Outliers)", f"{outlier_inadimplencia:.2f}%")
    with col3:
        st.metric("Taxa Geral", f"{taxa_geral:.2f}%")
    
    st.info(f"⚠️ Boletos de alto valor têm risco **{outlier_inadimplencia/taxa_geral:.1f}x maior** que a média!")
    
    st.markdown("---")
    
    # Seção 2: Análise Temporal
    st.subheader("2️⃣ Análise Temporal da Inadimplência")
    
    df['mes_emissao'] = df['dt_emissao'].dt.to_period('M')
    inadimplencia_mensal = df.groupby('mes_emissao')['inadimplente'].mean() * 100
    inadimplencia_mensal_plot = inadimplencia_mensal.reset_index()
    inadimplencia_mensal_plot['mes_emissao'] = inadimplencia_mensal_plot['mes_emissao'].astype(str)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(inadimplencia_mensal_plot['mes_emissao'], inadimplencia_mensal_plot['inadimplente'], marker='o', linewidth=2)
    ax.set_title('Evolução Mensal da Taxa de Boletos "Em Aberto" (Potencial Inadimplência)')
    ax.set_ylabel('Taxa de Inadimplência (%)')
    ax.set_xlabel('Mês de Emissão')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    
    st.warning("⚠️ Pico de inadimplência detectado em **Maio/2024** (4.61%)")
    
    st.markdown("---")
    
    # Seção 3: Impacto do CNAE
    st.subheader("3️⃣ Impacto do CNAE (Classificação Nacional de Atividades Econômicas)")
    
    df['cnae_4digitos'] = df['cd_cnae_prin'].astype(str).str[:4]
    cnae_counts = df['cnae_4digitos'].value_counts()
    cnae_validos = cnae_counts[cnae_counts >= 50].index
    df_cnae = df[df['cnae_4digitos'].isin(cnae_validos)]
    inadimplencia_por_cnae = df_cnae.groupby('cnae_4digitos')['inadimplente'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    inadimplencia_por_cnae_top = inadimplencia_por_cnae.sort_values(ascending=False).head(10)
    sns.barplot(x=inadimplencia_por_cnae_top.index, y=inadimplencia_por_cnae_top.values, palette="cubehelix", ax=ax)
    ax.set_title('Top 10 CNAEs (4 Dígitos) com Maior Taxa de Potencial Inadimplência')
    ax.set_ylabel('Taxa de Inadimplência (%)')
    ax.set_xlabel('CNAE (4 Primeiros Dígitos)')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

# --- INDICADORES DE RISCO ---
elif page == "⚠️ Indicadores de Risco":
    st.title("⚠️ Indicadores de Risco")
    
    st.markdown("---")
    
    # Agregação por Pagador
    df_pagador_agg = df.groupby('id_pagador').agg(
        taxa_inadimplencia=('inadimplente', 'mean'),
        share_vl_inad_pag_bol_6_a_15d=('share_vl_inad_pag_bol_6_a_15d', 'mean'),
        indicador_liquidez_quantitativo_3m=('indicador_liquidez_quantitativo_3m', 'mean'),
        score_materialidade_evolucao=('score_materialidade_evolucao', 'mean'),
        score_materialidade=('score_materialidade_v2', 'mean'),
        score_quantidade=('score_quantidade_v2', 'mean')
    ).reset_index()
    
    df_pagador_agg['alto_risco'] = df_pagador_agg['taxa_inadimplencia'].apply(lambda x: 'Alto Risco' if x > 0 else 'Baixo Risco')
    
    st.subheader("1️⃣ Correlação de Indicadores com Inadimplência")
    
    cols_analise = ['taxa_inadimplencia', 'share_vl_inad_pag_bol_6_a_15d', 
                    'indicador_liquidez_quantitativo_3m', 'score_materialidade_evolucao']
    correlation = df_pagador_agg[cols_analise].corr()
    
    st.write("**Matriz de Correlação:**")
    st.dataframe(correlation, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("2️⃣ Boxplots por Grupo de Risco")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Share de Atraso 6-15 dias**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='alto_risco', y='share_vl_inad_pag_bol_6_a_15d', data=df_pagador_agg.dropna(), ax=ax)
        ax.set_title('Por Grupo de Risco')
        st.pyplot(fig)
    
    with col2:
        st.write("**Liquidez Quantitativa 3M**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='alto_risco', y='indicador_liquidez_quantitativo_3m', data=df_pagador_agg.dropna(), ax=ax)
        ax.set_title('Por Grupo de Risco')
        st.pyplot(fig)
    
    with col3:
        st.write("**Score Materialidade Evolução**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='alto_risco', y='score_materialidade_evolucao', data=df_pagador_agg.dropna(), ax=ax)
        ax.set_title('Por Grupo de Risco')
        st.pyplot(fig)

# --- INDICADORES DE LIQUIDEZ ---
elif page == "💧 Indicadores de Liquidez":
    st.title("💧 Indicadores de Liquidez de 1 Mês")
    
    st.markdown("---")
    
    # Agregação por Pagador
    df_pagador_agg = df.groupby('id_pagador').agg(
        taxa_inadimplencia=('inadimplente', 'mean'),
        sacado_indice_liquidez_1m=('sacado_indice_liquidez_1m', 'mean'),
        cedente_indice_liquidez_1m=('cedente_indice_liquidez_1m', 'mean')
    ).reset_index()
    
    df_pagador_agg['alto_risco'] = df_pagador_agg['taxa_inadimplencia'].apply(lambda x: 'Alto Risco' if x > 0 else 'Baixo Risco')
    
    st.subheader("1️⃣ Correlação com Inadimplência")
    
    cols_liquidez = ['taxa_inadimplencia', 'sacado_indice_liquidez_1m', 'cedente_indice_liquidez_1m']
    correlation_liquidez = df_pagador_agg[cols_liquidez].corr()
    
    st.write("**Matriz de Correlação:**")
    st.dataframe(correlation_liquidez, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("2️⃣ Boxplots por Grupo de Risco")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Liquidez Sacado (Pagador) - 1 Mês**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='alto_risco', y='sacado_indice_liquidez_1m', data=df_pagador_agg.dropna(), ax=ax)
        ax.set_title('Por Grupo de Risco')
        st.pyplot(fig)
    
    with col2:
        st.write("**Liquidez Cedente - 1 Mês**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='alto_risco', y='cedente_indice_liquidez_1m', data=df_pagador_agg.dropna(), ax=ax)
        ax.set_title('Por Grupo de Risco')
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("3️⃣ Análise de Casos Extremos")
    
    df_pagador_agg['sacado_liquidez_baixa'] = df_pagador_agg['sacado_indice_liquidez_1m'] < 0.5
    df_extremos = df_pagador_agg[(df_pagador_agg['sacado_liquidez_baixa'] == True) & (df_pagador_agg['alto_risco'] == 'Baixo Risco')]
    
    st.success(f"✅ Identificados **{len(df_extremos)}** pagadores com liquidez baixa (< 50%) mas sem inadimplência atual.")
    st.info("💡 Estes representam um risco iminente ou uma oportunidade de renegociação.")

# --- DADOS DETALHADOS ---
elif page == "📈 Dados Detalhados":
    st.title("📈 Dados Detalhados")
    
    st.markdown("---")
    
    st.subheader("1️⃣ Visualizar Dados Brutos")
    
    num_rows = st.slider("Número de linhas a exibir:", 10, 100, 20)
    st.dataframe(df.head(num_rows), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("2️⃣ Filtrar Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.multiselect(
            "Filtrar por Status de Pagamento:",
            df['status_pagamento'].unique(),
            default=df['status_pagamento'].unique()
        )
    
    with col2:
        tipo_especie_filter = st.multiselect(
            "Filtrar por Tipo de Espécie:",
            df['tipo_especie'].unique(),
            default=df['tipo_especie'].unique()[:5]
        )
    
    df_filtered = df[(df['status_pagamento'].isin(status_filter)) & (df['tipo_especie'].isin(tipo_especie_filter))]
    
    st.write(f"**Total de registros filtrados:** {len(df_filtered)}")
    st.dataframe(df_filtered, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("3️⃣ Download de Dados")
    
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados filtrados como CSV",
        data=csv,
        file_name="boletos_filtrados.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>Relatório EDA - Desafio Fiap/Nuclea | Desenvolvido com Streamlit</p>
</div>
""", unsafe_allow_html=True)

