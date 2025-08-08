import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv
import io
import contextlib
import re
from datetime import datetime
from sklearn.ensemble import IsolationForest

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.sidebar.error("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=api_key)

# Session state initialization
if 'show_insights' not in st.session_state:
    st.session_state.show_insights = False
if 'show_cleaning' not in st.session_state:
    st.session_state.show_cleaning = False

def sanitize_code(code):
    # Remove markdown code blocks
    code = re.sub(r"```python", "", code)
    code = re.sub(r"```", "", code)
    # Remove plt.show() but keep everything else
    code = re.sub(r"plt\.show\(\)", "", code)
    return code.strip()

@st.cache_data(ttl=300)
def call_gemini_api(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        enhanced_prompt = f"""You are a helpful assistant that returns ONLY Python matplotlib/seaborn code.
The DataFrame is named df. Generate code that directly produces a plot.
Do not include explanations, imports, print statements, or plt.show().
Only return valid Python code.

IMPORTANT: If the user asks for a pie chart, you MUST use plt.pie() function, not bar charts or other plot types.

Examples:
- For pie chart: plt.pie(df['column'].value_counts(), labels=df['column'].unique(), autopct='%1.1f%%')
- For bar chart: plt.bar(df['column'].value_counts().index, df['column'].value_counts().values)

User request: {prompt}
"""
        
        response = model.generate_content(enhanced_prompt)
        code = sanitize_code(response.text)
        
        st.write("Generated Code:")
        st.code(code)
        
        return code
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return ""

def analyze_data_patterns(df):
    """ML-powered data insights"""
    insights = {}
    
    insights['shape'] = df.shape
    insights['numeric_cols'] = df.select_dtypes(include=np.number).columns.tolist()
    insights['categorical_cols'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    insights['datetime_cols'] = df.select_dtypes(include='datetime64').columns.tolist()
    
    numeric_df = df[insights['numeric_cols']].dropna()
    if len(numeric_df.columns) > 0:
        try:
            iso = IsolationForest(contamination=0.1)
            outliers = iso.fit_predict(numeric_df)
            insights['outlier_cols'] = [col for col, vals in 
                                      zip(numeric_df.columns, outliers) 
                                      if sum(vals == -1) > 0]
        except:
            pass
    
    return insights

def create_download_buttons(fig):
    """Create download buttons for all formats"""
    st.subheader("Export Options")
    
    downloads = {}
    for fmt_name, (fmt_code, mime_type) in {
        "PNG": ("png", "image/png"),
        "PDF": ("pdf", "application/pdf"), 
        "SVG": ("svg", "image/svg+xml"),
        "JPEG": ("jpeg", "image/jpeg")
    }.items():
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt_code, bbox_inches="tight")
            buf.seek(0)
            downloads[fmt_name] = (buf.getvalue(), mime_type, fmt_code)
        except Exception as e:
            st.warning(f"Could not generate {fmt_name}: {str(e)}")
    
    if downloads:
        cols = st.columns(len(downloads))
        for i, (fmt_name, (data, mime_type, fmt_code)) in enumerate(downloads.items()):
            with cols[i]:
                st.download_button(
                    label=f"{fmt_name}",
                    data=data,
                    file_name=f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt_code}",
                    mime=mime_type
                )

def main():
    st.title("Data Visualizer")
    st.write("Upload CSV and generate plots from plain English prompts.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())

            if st.sidebar.button("Show Data Insights"):
                st.session_state.show_insights = not st.session_state.show_insights
            
            if st.session_state.show_insights:
                with st.spinner("Analyzing data..."):
                    insights = analyze_data_patterns(df)
                st.subheader("Data Insights")
                st.write(f"Rows: {insights['shape'][0]}, Columns: {insights['shape'][1]}")
                st.write(f"Numeric columns: {len(insights['numeric_cols'])}")
                st.write(f"Categorical columns: {len(insights['categorical_cols'])}")
                if insights.get('outlier_cols'):
                    st.write(f"Potential outliers in: {', '.join(insights['outlier_cols'])}")

            if st.sidebar.button("Show Cleaning Tools"):
                st.session_state.show_cleaning = not st.session_state.show_cleaning
            
            if st.session_state.show_cleaning:
                st.subheader("Data Cleaning Tools")
                
                non_numeric = df.select_dtypes(exclude=np.number).columns.tolist()
                if non_numeric:
                    convert_col = st.selectbox("Select column to convert", non_numeric)
                    conversion_method = st.radio("Conversion method", 
                                              ["Auto-detect numbers", "Map categories to numbers", "Drop column"])
                    
                    if st.button("Apply Conversion"):
                        try:
                            if conversion_method == "Auto-detect numbers":
                                df[convert_col] = pd.to_numeric(df[convert_col], errors='coerce')
                                st.success(f"Converted {convert_col} to numeric")
                                
                            elif conversion_method == "Map categories to numbers":
                                unique_vals = df[convert_col].unique()
                                mapping = {val: i+1 for i, val in enumerate(unique_vals)}
                                df[convert_col] = df[convert_col].map(mapping)
                                st.success(f"Mapped {convert_col} to numeric codes")
                                
                            elif conversion_method == "Drop column":
                                df = df.drop(columns=[convert_col])
                                st.success(f"Dropped column {convert_col}")
                            
                            st.dataframe(df.head())
                            
                        except Exception as e:
                            st.error(f"Conversion failed: {str(e)}")
                else:
                    st.info("All columns are numeric.")

            prompt = st.text_input("Describe your plot (e.g., 'scatter plot of age vs income by gender')")

            if st.button("Generate Plot") and prompt.strip():
                with st.spinner("Generating plot..."):
                    code_output = call_gemini_api(prompt)
                
                if code_output:
                    user_wants_pie = any(word in prompt.lower() for word in ['pie', 'donut', 'circle'])
                    code_has_pie = 'plt.pie' in code_output or '.plot.pie' in code_output
                    
                    if user_wants_pie and not code_has_pie:
                        st.warning("AI didn't generate pie chart code. Trying manual pie chart generation...")
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        if categorical_cols:
                            first_cat_col = categorical_cols[0]
                            code_output = f"""plt.figure(figsize=(8,8))
plt.pie(df['{first_cat_col}'].value_counts(), labels=df['{first_cat_col}'].value_counts().index, autopct='%1.1f%%', startangle=90)
plt.title('{first_cat_col} Distribution')
plt.axis('equal')"""
                            st.write("Using fallback pie chart code:")
                            st.code(code_output)
                        else:
                            st.error("No categorical columns found for pie chart")
                            return
                    
                    st.text("Final Code to Execute:\n" + code_output)
                    
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            exec(code_output, {"df": df, "plt": plt, "sns": sns, "pd": pd, "np": np})
                        
                        fig = plt.gcf()
                        
                        if fig.get_axes():
                            st.pyplot(fig)
                            create_download_buttons(fig)
                        else:
                            st.warning("Code ran but produced no visible plot")
                            
                    except Exception as e:
                        st.error(f"Execution Error: {str(e)}")
                        st.text("Code:\n" + code_output)
                else:
                    st.warning("No valid plot code generated")

            else:
                st.info("Please enter a plot description and click 'Generate Plot'.")

        except Exception as e:
            st.error(f"Could not read CSV file: {e}")

if __name__ == "__main__":
    main()
