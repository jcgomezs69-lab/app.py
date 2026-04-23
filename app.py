# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página
st.set_page_config(
    page_title="Dashboard Adopción de Tecnología",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar modelo y datos
@st.cache_resource
def load_model():
    model = joblib.load('models/rf_model.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, feature_names

@st.cache_data
def load_data():
    return pd.read_csv('data/adopcion_tecnologia.csv')

model, feature_names = load_model()
df = load_data()

# Título principal
st.title("📊 Dashboard de Resultados - Random Forest")
st.markdown("### Estudio de Adopción de Tecnología")

# Sidebar - Navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio("Selecciona una sección:", [
    "🏠 Inicio", 
    "📈 Análisis Exploratorio", 
    "🤖 Predicción Individual",
    "🔍 Interpretación del Modelo",
    "📉 Rendimiento del Modelo"
])

# ==========================================
# PÁGINA 1: INICIO
# ==========================================
if page == "🏠 Inicio":
    st.header("Resumen del Estudio")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Participantes", len(df))
    with col2:
        tasa_adopcion = (df['adopcion'].sum() / len(df)) * 100
        st.metric("Tasa de Adopción", f"{tasa_adopcion:.1f}%")
    with col3:
        st.metric("Variables Predictoras", len(feature_names))
    with col4:
        st.metric("Árboles en el Bosque", model.n_estimators)
    
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Distribución de Adopción")
    fig_pie = px.pie(
        df, names='adopcion', 
        title="Proporción de Adopción de Tecnología",
        color='adopcion',
        color_discrete_map={0: '#FF6B6B', 1: '#4ECDC4'}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# PÁGINA 2: ANÁLISIS EXPLORATORIO
# ==========================================
elif page == "📈 Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    # Selector de variable
    var_seleccionada = st.selectbox(
        "Selecciona una variable para visualizar:", 
        feature_names
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma
        fig_hist = px.histogram(
            df, x=var_seleccionada, color='adopcion',
            barmode='group',
            title=f"Distribución de {var_seleccionada}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Boxplot
        fig_box = px.box(
            df, y=var_seleccionada, x='adopcion',
            title=f"Boxplot de {var_seleccionada} por Adopción"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Matriz de correlación
    st.subheader("Matriz de Correlación")
    corr_matrix = df[feature_names + ['adopcion']].corr()
    fig_corr = px.imshow(
        corr_matrix, text_auto=True, aspect="auto",
        title="Correlación entre Variables"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# PÁGINA 3: PREDICCIÓN INDIVIDUAL
# ==========================================
elif page == "🤖 Predicción Individual":
    st.header("Predicción de Adopción Individual")
    st.markdown("Ingresa los valores del participante para predecir si adoptará la tecnología.")
    
    # Inputs del usuario en sidebar
    st.sidebar.header("Características del Participante")
    
    inputs = {}
    for feature in feature_names:
        if df[feature].dtype in ['int64', 'float64']:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            inputs[feature] = st.sidebar.slider(
                f"{feature}", min_val, max_val, mean_val
            )
        else:
            options = df[feature].unique().tolist()
            inputs[feature] = st.sidebar.selectbox(f"{feature}", options)
    
    # Botón de predicción
    if st.sidebar.button("🔮 Predecir Adopción"):
        # Preparar datos
        input_df = pd.DataFrame([inputs])
        
        # Predicción
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Mostrar resultados
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Resultado")
            if prediction == 1:
                st.success("✅ **ADOPTARÁ** la tecnología")
            else:
                st.error("❌ **NO ADOPTARÁ** la tecnología")
            
            st.metric(
                "Confianza de Predicción", 
                f"{max(prediction_proba)*100:.1f}%"
            )
        
        with col2:
            # Gráfico de probabilidades
            fig_proba = go.Figure(data=[
                go.Bar(
                    x=['No Adopta', 'Adopta'],
                    y=prediction_proba,
                    marker_color=['#FF6B6B', '#4ECDC4']
                )
            ])
            fig_proba.update_layout(
                title="Probabilidades de Predicción",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_proba, use_container_width=True)

# ==========================================
# PÁGINA 4: INTERPRETACIÓN DEL MODELO
# ==========================================
elif page == "🔍 Interpretación del Modelo":
    st.header("Interpretación del Modelo Random Forest")
    
    # Importancia de características
    st.subheader("Importancia de Variables")
    
    importances = pd.DataFrame({
        'variable': feature_names,
        'importancia': model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_imp = px.bar(
            importances, x='importancia', y='variable',
            orientation='h', title="Importancia de Características"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col2:
        st.dataframe(importances, use_container_width=True)
    
    # Análisis de un árbol individual
    st.subheader("Visualización de un Árbol Individual")
    from sklearn.tree import plot_tree
    
    tree_index = st.slider("Selecciona árbol a visualizar", 0, model.n_estimators-1, 0)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model.estimators_[tree_index], 
        feature_names=feature_names,
        class_names=['No Adopta', 'Adopta'],
        filled=True, ax=ax, max_depth=3
    )
    st.pyplot(fig)

# ==========================================
# PÁGINA 5: RENDIMIENTO DEL MODELO
# ==========================================
elif page == "📉 Rendimiento del Modelo":
    st.header("Métricas de Rendimiento del Modelo")
    
    # Recalcular predicciones para todo el dataset
    X_full = df[feature_names]
    y_true = df['adopcion']
    y_pred_full = model.predict(X_full)
    y_proba_full = model.predict_proba(X_full)[:, 1]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        from sklearn.metrics import accuracy_score
        st.metric("Accuracy", f"{accuracy_score(y_true, y_pred_full):.3f}")
    with col2:
        from sklearn.metrics import precision_score
        st.metric("Precision", f"{precision_score(y_true, y_pred_full):.3f}")
    with col3:
        from sklearn.metrics import recall_score
        st.metric("Recall", f"{recall_score(y_true, y_pred_full):.3f}")
    
    # Matriz de confusión
    st.subheader("Matriz de Confusión")
    cm = confusion_matrix(y_true, y_pred_full)
    fig_cm = px.imshow(
        cm, text_auto=True, aspect="auto",
        labels=dict(x="Predicción", y="Real"),
        x=['No Adopta', 'Adopta'],
        y=['No Adopta', 'Adopta'],
        title="Matriz de Confusión"
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Curva ROC
    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_true, y_proba_full)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, 
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        fill='tozeroy'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], 
        mode='lines', name='Random',
        line=dict(dash='dash')
    ))
    fig_roc.update_layout(
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        title='Curva ROC'
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Reporte de clasificación
    st.subheader("Reporte de Clasificación")
    report = classification_report(y_true, y_pred_full, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)
