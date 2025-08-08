import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openai
import os
from io import BytesIO
import base64
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# Load OpenAI API key with validation
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.sidebar.warning("⚠ OpenAI API key not found. AI features limited to template mode.")

# Cached data loading
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Function to call GPT for code generation
def call_openai_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that returns python matplotlib/seaborn code."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return response['choices'][0]['message']['content']

# Enhanced validation function
def validate_columns(df, chart_type, x_axis, y_axis=None, group_by=None):
    errors = []
    valid_cols = df.columns.tolist()
    
    if chart_type in ["scatter", "line", "box", "violin", "area"]:
        if x_axis not in valid_cols:
            errors.append(f"X-axis column '{x_axis}' not found")
        if y_axis and y_axis not in valid_cols:
            errors.append(f"Y-axis column '{y_axis}' not found")
    elif chart_type == "bar":
        if x_axis not in valid_cols:
            errors.append(f"X-axis column '{x_axis}' not found")
        if y_axis and y_axis not in valid_cols:
            errors.append(f"Y-axis column '{y_axis}' not found")
    elif chart_type in ["hist", "kde"]:
        if x_axis not in valid_cols:
            errors.append(f"Column '{x_axis}' not found")
    elif chart_type == "pie":
        if x_axis not in valid_cols:
            errors.append(f"Label column '{x_axis}' not found")
        if y_axis and y_axis not in valid_cols:
            errors.append(f"Value column '{y_axis}' not found")
    elif chart_type == "count":
        if x_axis not in valid_cols:
            errors.append(f"Column '{x_axis}' not found")
    elif chart_type == "heatmap":
        if len(df.select_dtypes(include="number").columns) < 2:
            errors.append("Need at least 2 numeric columns for heatmap")
    
    if errors:
        st.error("\n".join([f"❌ {e}" for e in errors]))
        return False
    return True

# ML-powered data analysis
def analyze_data_patterns(df):
    """Analyze data characteristics for visualization recommendations"""
    analysis = {}
    
    # Check data shape
    analysis['num_rows'] = len(df)
    analysis['num_cols'] = len(df.columns)
    
    # Check column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    analysis['numeric_cols'] = numeric_cols
    analysis['categorical_cols'] = categorical_cols
    analysis['datetime_cols'] = datetime_cols
    
    # Check data distribution
    if len(numeric_cols) > 0:
        analysis['skewness'] = df[numeric_cols].skew().to_dict()
    
    return analysis

# Manual plot builder with enhancements
def manual_plot_builder(df):
    st.subheader("📉 Smart Manual Plot Builder")

    chart_types = [
        "scatter", "line", "bar", "hist", "box", "violin", "kde",
        "area", "count", "heatmap", "pairplot", "pie", "interactive"
    ]

    chart_type = st.selectbox("Choose Chart Type", chart_types, key="chart_type")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    # Chart-specific axis selection
    x_axis = y_axis = group_by = None

    if chart_type in ["scatter", "line", "box", "violin", "area"]:
        x_axis = st.selectbox("Select X-axis", all_cols, key="x_axis")
        valid_numeric = [col for col in numeric_cols if col in df.columns]
        y_axis = st.selectbox("Select Y-axis (numeric)", valid_numeric, key="y_axis") if valid_numeric else None
    elif chart_type == "bar":
        valid_categorical = [col for col in categorical_cols if col in df.columns]
        x_axis = st.selectbox("Select X-axis (categorical)", valid_categorical, key="x_axis_bar") if valid_categorical else None
        valid_numeric = [col for col in numeric_cols if col in df.columns]
        y_axis = st.selectbox("Select Y-axis (numeric)", valid_numeric, key="y_axis_bar") if valid_numeric else None
    elif chart_type == "hist":
        valid_numeric = [col for col in numeric_cols if col in df.columns]
        x_axis = st.selectbox("Select Numeric Column", valid_numeric, key="x_hist") if valid_numeric else None
    elif chart_type == "kde":
        valid_numeric = [col for col in numeric_cols if col in df.columns]
        x_axis = st.selectbox("Select Numeric Column", valid_numeric, key="x_kde") if valid_numeric else None
    elif chart_type == "count":
        valid_categorical = [col for col in categorical_cols if col in df.columns]
        x_axis = st.selectbox("Select Categorical Column", valid_categorical, key="x_count") if valid_categorical else None
    elif chart_type == "heatmap":
        st.info("✅ Correlation matrix will be generated using numeric columns.")
    elif chart_type == "pairplot":
        st.info("✅ Pairplot will show relationships between numeric features.")
    elif chart_type == "pie":
        valid_categorical = [col for col in categorical_cols if col in df.columns]
        x_axis = st.selectbox("Select Categorical Column for Pie Labels", valid_categorical, key="x_pie") if valid_categorical else None
        valid_numeric = [col for col in numeric_cols if col in df.columns]
        y_axis = st.selectbox("Select Numeric Column for Pie Values", valid_numeric, key="y_pie") if valid_numeric else None
    elif chart_type == "interactive":
        st.info("ℹ Select columns for interactive visualization")

    group_by = st.selectbox("Optional - Group by", [None] + all_cols, key="group_by")
    show_legend = st.checkbox("Show legend", value=True)

    # Plot Customization Section
    st.subheader("🎨 Plot Customization")
    title = st.text_input("Chart Title", value="Data Visualization")
    x_label = st.text_input("X-axis Label", value=x_axis if x_axis else "")
    y_label = st.text_input("Y-axis Label", value=y_axis if y_axis else "")
    palette = st.selectbox("Color Palette", ["deep", "pastel", "dark", "colorblind", "coolwarm"])

    # ML Recommendations
    st.subheader("🤖 Smart Recommendations")
    if st.button("Get Chart Suggestions"):
        with st.spinner("Analyzing data..."):
            patterns = analyze_data_patterns(df)
        
        # Generate recommendations based on data characteristics
        recommendations = []
        
        # Check for time series potential
        if len(patterns['datetime_cols']) > 0:
            recommendations.append("📈 Time series plot using datetime column")
        
        # Check for categorical vs numeric relationships
        if len(patterns['categorical_cols']) > 0 and len(patterns['numeric_cols']) > 0:
            recommendations.append("📊 Bar chart comparing categories and values")
            recommendations.append("📊 Box plot showing value distributions by category")
        
        # Check for correlation potential
        if len(patterns['numeric_cols']) >= 3:
            recommendations.append("🔥 Correlation heatmap of numeric columns")
        
        # Check for distribution analysis
        if len(patterns['numeric_cols']) > 0:
            recommendations.append("📊 Histogram of numeric column distributions")
        
        # Display recommendations
        if recommendations:
            st.write("Suggested charts based on your data:")
            for rec in recommendations[:3]:  # Show top 3
                st.markdown(f"- {rec}")
        else:
            st.info("Not enough data features for specific recommendations")

    if st.button("Generate Plot", key="plot_button"):
        if not validate_columns(df, chart_type, x_axis, y_axis, group_by):
            return
            
        plt.figure(figsize=(10, 6))
        try:
            # Plotting logic
            if chart_type == "interactive":
                plot_type = st.selectbox("Plotly Type", ["scatter", "line", "bar", "3d-scatter"])
                
                if plot_type == "scatter":
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=group_by)
                elif plot_type == "line":
                    fig = px.line(df, x=x_axis, y=y_axis, color=group_by)
                elif plot_type == "bar":
                    fig = px.bar(df, x=x_axis, y=y_axis, color=group_by)
                elif plot_type == "3d-scatter":
                    z_axis = st.selectbox("Z-axis", numeric_cols)
                    fig = px.scatter_3d(df, x=x_axis, y=y_axis, z=z_axis, color=group_by)
                
                fig.update_layout(title=title)
                st.plotly_chart(fig, use_container_width=True)
                return
            elif chart_type in ["scatter", "line", "box", "violin", "area", "bar"]:
                if not x_axis or not y_axis:
                    st.error("Missing axis selection")
                    return
                if chart_type == "scatter":
                    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=group_by)
                elif chart_type == "line":
                    sns.lineplot(data=df, x=x_axis, y=y_axis, hue=group_by)
                elif chart_type == "bar":
                    sns.barplot(data=df, x=x_axis, y=y_axis, hue=group_by)
                elif chart_type == "box":
                    sns.boxplot(data=df, x=x_axis, y=y_axis, hue=group_by)
                elif chart_type == "violin":
                    sns.violinplot(data=df, x=x_axis, y=y_axis, hue=group_by)
                elif chart_type == "area":
                    clean_df = df[[x_axis, y_axis]].dropna()
                    clean_df.set_index(x_axis)[y_axis].plot(kind='area', label=y_axis)
            elif chart_type == "hist":
                if not x_axis:
                    st.error("Missing column selection")
                    return
                sns.histplot(data=df, x=x_axis, bins=30, hue=group_by)
            elif chart_type == "kde":
                if not x_axis:
                    st.error("Missing column selection")
                    return
                sns.kdeplot(data=df[x_axis].dropna(), fill=True, label=x_axis)
            elif chart_type == "count":
                if not x_axis:
                    st.error("Missing column selection")
                    return
                sns.countplot(data=df, x=x_axis, hue=group_by)
            elif chart_type == "heatmap":
                corr = df.select_dtypes(include="number").corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            elif chart_type == "pairplot":
                fig = sns.pairplot(df.select_dtypes(include="number"), hue=group_by)
                st.pyplot(fig)
                return
            elif chart_type == "pie":
                if not x_axis or not y_axis:
                    st.error("Missing axis selection")
                    return
                pie_data = df[[x_axis, y_axis]].dropna()
                pie_data = pie_data.groupby(x_axis)[y_axis].sum()
                plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
                plt.axis('equal')

            # Add customization
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            sns.set_palette(palette)

            # Show/hide legend
            if show_legend:
                plt.legend(loc="best")
            else:
                plt.legend([], [], frameon=False)

            # Show plot with progress
            with st.spinner("Generating plot..."):
                st.pyplot(plt.gcf())
                
            # Add download button
            buffer = BytesIO()
            plt.savefig(buffer, format="png", bbox_inches='tight')
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            # Advanced Export Options
            st.subheader("💾 Export Options")
            export_format = st.selectbox("Choose Format", ["PNG", "PDF", "SVG", "JPEG"])
            if st.button("Export Plot"):
                if export_format == "PNG":
                    st.download_button(
                        label="Download PNG",
                        data=buffer.getvalue(),
                        file_name=f"{chart_type}_plot.png",
                        mime="image/png"
                    )
                elif export_format == "PDF":
                    buffer_pdf = BytesIO()
                    plt.savefig(buffer_pdf, format="pdf", bbox_inches='tight')
                    st.download_button(
                        label="Download PDF",
                        data=buffer_pdf.getvalue(),
                        file_name=f"{chart_type}_plot.pdf",
                        mime="application/pdf"
                    )
                elif export_format == "SVG":
                    buffer_svg = BytesIO()
                    plt.savefig(buffer_svg, format="svg", bbox_inches='tight')
                    st.download_button(
                        label="Download SVG",
                        data=buffer_svg.getvalue(),
                        file_name=f"{chart_type}_plot.svg",
                        mime="image/svg+xml"
                    )
                elif export_format == "JPEG":
                    buffer_jpeg = BytesIO()
                    plt.savefig(buffer_jpeg, format="jpeg", bbox_inches='tight')
                    st.download_button(
                        label="Download JPEG",
                        data=buffer_jpeg.getvalue(),
                        file_name=f"{chart_type}_plot.jpeg",
                        mime="image/jpeg"
                    )

            plt.clf()

        except Exception as e:
            st.error(f"❌ Error generating plot: {e}")

# Main app
def main():
    st.title("🤖📊 ChatGPT-Powered Data Visualizer")
    st.write("Upload a CSV file and describe the chart you want in plain English!")

    uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.write("🔍 Preview of Uploaded Data")
            st.dataframe(df.head())

            # Data Preprocessing
            st.sidebar.subheader("🧹 Data Preprocessing")
            # Use session state to remember button state
            if 'show_cleaning' not in st.session_state:
                st.session_state.show_cleaning = False

            if st.sidebar.button("Show Data Cleaning Options"):
                st.session_state.show_cleaning = not st.session_state.show_cleaning

            if st.session_state.show_cleaning:
                st.subheader("Data Cleaning")
                
                # Missing values
                if st.checkbox("Handle Missing Values"):
                    strategy = st.selectbox("Strategy", ["drop", "mean", "median", "ffill"])
                    if st.button("Apply Missing Value Handling"):
                        if strategy == "drop":
                            df = df.dropna()
                        else:
                            for col in df.select_dtypes(include=np.number).columns:
                                if strategy == "mean":
                                    df[col] = df[col].fillna(df[col].mean())
                                elif strategy == "median":
                                    df[col] = df[col].fillna(df[col].median())
                                elif strategy == "ffill":
                                    df[col] = df[col].fillna(method='ffill')
                        st.success("Missing values handled")
                        st.dataframe(df.head())
                
                # Outlier handling
                if st.checkbox("Detect Outliers"):
                    outlier_cols = st.multiselect("Select columns for outlier detection", 
                                                 df.select_dtypes(include=np.number).columns.tolist())
                    if st.button("Detect Outliers"):
                        for col in outlier_cols:
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            outliers = df[(df[col] < (q1 - 1.5*iqr)) | (df[col] > (q3 + 1.5*iqr))]
                            st.write(f"Outliers in {col}: {len(outliers)}")

            prompt = st.text_input("🧠 Describe your plot (e.g. 'scatter plot of age vs income colored by gender')")

            if st.button("Generate & Run Visualization"):
                if prompt.strip():
                    try:
                        code_output = call_openai_api(prompt)
                        st.code(code_output, language="python")
                        st.info("✅ GPT returned Python code. Copy it to your notebook to run.")
                        st.warning("⚠ This app does not auto-run code for safety. Use the manual plot builder below.")
                    except Exception as api_error:
                        st.warning(f"⚠ OpenAI API error: {api_error}")
                        st.info("Switching to manual fallback plotting mode.")
                        manual_plot_builder(df)
                else:
                    st.warning("⚠ Please enter a plot description or use the manual builder below.")
                    manual_plot_builder(df)
            else:
                st.info("Use the manual plot builder below.")
                manual_plot_builder(df)

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")

if __name__ == "__main__":
    main()
