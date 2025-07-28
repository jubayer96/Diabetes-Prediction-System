import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import plotly.express as px

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 18px !important;
    }
    .block-container {
        padding: 1rem 2rem;
    }
    .element-container {
        padding: 1rem;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .pill {
        display: inline-block;
        margin: 0.25rem;
        padding: 0.4rem 0.8rem;
        background-color: #e0f7fa;
        border-radius: 999px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data
df = pd.read_csv("top_feature_df_ohe.csv")
X = df.drop(columns='Diabetes')
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Column Types
binary_columns = [
    'High BP', 'Cholesterol', 'Heart Attack', 'Heart Disease', 'Stroke', 'Asthma', 'Depressive Disorder',
    'Kidney Disease', 'Arthritis', 'Deaf', 'Blind', 'Difficulty Concentrating', 'Difficulty Walking',
    'Cancer History', 'Health Insurance Status', 'Smoker', 'Diabetes', 'Covid', 'Obesity',
    'Heavy Alcohol Consumption', 'Sex', 'SNAP Benefits', 'Consumed Marijuana Edibles']

categorical_columns = [
    'General Health', 'Physical Activity', 'Marital Status', 'Age Group', 'Race Group',
    'Routine Checkup Last Visit', 'Education Level', 'Employment Status', 'Household Income',
    'Blood Sugar Test History']

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to:", ["Project Overview", "Data Exploration", "Model Insights", "Predictions"])

# Project Overview
if option == "Project Overview":
    st.title("Diabetes Risk Factors and Relevant Visualization Techniques Dashboard")
    st.markdown("""<div class="element-container">This app uses a Random Forest model trained on top features from a health dataset to predict the likelihood of Diabetes.</div>""", unsafe_allow_html=True)

    with st.container():
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", df.shape[0])
        col2.metric("Total Features", len(X.columns))
        col3.metric("Diabetes %", f"{df['Diabetes'].mean() * 100:.2f}%")

    with st.container():
        st.subheader("Feature Types Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Binary Features", len(binary_columns))
        col2.metric("Categorical Features", len(categorical_columns))
        col3.metric("Other Features", len(X.columns) - len(binary_columns) - len(categorical_columns))

        st.markdown("#### Binary Features")
        st.markdown(' '.join([f'<span class="pill">{col}</span>' for col in binary_columns]), unsafe_allow_html=True)

        st.markdown("#### Categorical Features")
        st.markdown(' '.join([f'<span class="pill">{col}</span>' for col in categorical_columns]), unsafe_allow_html=True)

    st.subheader("Binary Feature Class Distribution")
    binary_features = []
    for col in X.columns:
        counts = df[col].value_counts(normalize=True) * 100
        binary_features.append((col, counts.get(0, 0), counts.get(1, 0)))

    for i in range(0, len(binary_features), 3):
        cols = st.columns(3)
        for j in range(3):
            if i + j < len(binary_features):
                feat, perc_0, perc_1 = binary_features[i + j]
                with cols[j]:
                    st.markdown(f"**{feat}**")
                    st.progress(int(perc_1), text=f"1: {perc_1:.1f}% | 0: {perc_0:.1f}%")

# The rest of the sections remain unchanged...


elif option == "Data Exploration":
    st.title("Data Exploration")
    class_counts = df['Diabetes'].value_counts()

    with st.container():
        st.subheader("Diabetes Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.barplot(x=class_counts.index, y=class_counts.values, palette='Set2', ax=ax)
            ax.set_xticklabels(['No Diabetes', 'Diabetes'])
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(class_counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize", X.columns)
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=selected_feature, hue="Diabetes", multiple="stack", palette="pastel", bins=10, ax=ax)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Diabetes', y=selected_feature, palette="Set3", ax=ax)
        ax.set_xticklabels(['No Diabetes', 'Diabetes'])
        st.pyplot(fig)

    st.subheader("Feature Importance")
    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig = go.Figure(go.Bar(
        x=feat_df['Importance'],
        y=feat_df['Feature'],
        orientation='h',
        marker=dict(color='teal')
    ))
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig)

    

    st.subheader("Feature Connectivity Network")

    # Include Diabetes in the correlation matrix (do NOT drop it)
    corr_matrix = df.corr()

    threshold = 0.2
    G = nx.Graph()

    # Add Diabetes node explicitly (it will be added automatically anyway but just for clarity)
    G.add_node('Diabetes')

    # Add edges from Diabetes to other features with abs(corr) >= threshold
    for feature in corr_matrix.columns:
        if feature != 'Diabetes':
            corr_value = corr_matrix.loc['Diabetes', feature]
            if abs(corr_value) >= threshold:
                G.add_edge('Diabetes', feature, weight=corr_value)

    # Optionally add edges between other features too (comment out if not needed)
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and i != 'Diabetes' and j != 'Diabetes':
                if abs(corr_matrix.loc[i, j]) >= threshold:
                    G.add_edge(i, j, weight=corr_matrix.loc[i, j])

    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Color Diabetes node distinctly (red), others by avg correlation with neighbors
    node_colors = [
        'red' if node == 'Diabetes' else
        np.mean([abs(corr_matrix.loc[node, other]) for other in G.neighbors(node)])
        for node in G.nodes()
    ]

    node_sizes = [30 if node == 'Diabetes' else 20 for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(thickness=15, title='Avg. Correlation', xanchor='left', titleside='right'),
            line_width=2
        )
    )

    fig_network = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='<b>Feature Connectivity Network (Diabetes-centered)</b>',
            titlefont_size=20,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )

    st.plotly_chart(fig_network, use_container_width=True)

# Model Insights
elif option == "Model Insights":
    st.title("Model Performance")
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    st.pyplot(fig)

# Predictions
elif option == "Predictions":
    st.title("Make a Prediction")
    st.markdown("### Individual Prediction")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.selectbox(f"{col}", [0, 1], key=col)

    input_df = pd.DataFrame([user_input])
    prediction = rf_model.predict(input_df)[0]
    probability = rf_model.predict_proba(input_df)[:, 1][0]

    st.subheader("Prediction Result")
    st.write("✅ Diabetes" if prediction == 1 else "❌ No Diabetes")
    st.write(f"Probability: {probability:.2f}")

    st.markdown("---")
    st.markdown("### Batch Prediction from CSV")
    st.info("Upload a CSV file with the same feature columns used in training to get predictions.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_preds = rf_model.predict(batch_df)
            batch_probs = rf_model.predict_proba(batch_df)[:, 1]

            batch_results = batch_df.copy()
            batch_results['Prediction'] = batch_preds
            batch_results['Probability'] = batch_probs

            st.write("Sample Predictions:")
            st.dataframe(batch_results.head())

            csv = batch_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, file_name="diabetes_batch_predictions.csv", mime='text/csv')

        except Exception as e:
            st.error(f"Error processing file: {e}")
