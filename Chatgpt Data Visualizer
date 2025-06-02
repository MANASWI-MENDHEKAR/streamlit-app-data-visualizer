import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import openai

# Load OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT call function
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

# Manual plotting fallback
def manual_plot_builder(df):
    st.subheader("📉 Fallback Mode: Manual Plot Builder")

    chart_types = [
        "scatter", "line", "bar", "hist", "box", "violin", "kde",
        "area", "count", "heatmap", "pairplot"
    ]

    chart_type = st.selectbox("Choose Chart Type", chart_types)

    # X and Y selections depending on chart type
    x_axis = st.selectbox("Select X-axis", df.columns) if chart_type not in ["heatmap", "pairplot", "count"] else None
    y_axis = st.selectbox("Select Y-axis", df.columns) if chart_type in ["scatter", "line", "bar", "box", "violin", "area"] else None
    group_by = st.selectbox("Optional - Group by", [None] + list(df.columns))

    if st.button("Generate Plot"):
        plt.figure(figsize=(10, 6))
        try:
            if chart_type == "scatter":
                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "line":
                sns.lineplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "bar":
                sns.barplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "hist":
                sns.histplot(data=df, x=x_axis, bins=30, hue=group_by)
            elif chart_type == "box":
                sns.boxplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "violin":
                sns.violinplot(data=df, x=x_axis, y=y_axis, hue=group_by)
            elif chart_type == "kde":
                sns.kdeplot(data=df[x_axis], fill=True)
            elif chart_type == "area":
                df.set_index(x_axis)[y_axis].plot(kind='area')
            elif chart_type == "count":
                sns.countplot(data=df, x=x_axis, hue=group_by)
            elif chart_type == "heatmap":
                corr = df.select_dtypes(include="number").corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            elif chart_type == "pairplot":
                fig = sns.pairplot(df.select_dtypes(include="number"), hue=group_by)
                st.pyplot(fig)
                return

            st.pyplot(plt.gcf())
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
            df = pd.read_csv(uploaded_file)
            st.write("🔍 Preview of Uploaded Data")
            st.dataframe(df.head())

            prompt = st.text_input("🧠 Describe your plot (e.g. 'scatter plot of age vs income colored by gender')")

            if st.button("Generate & Run Visualization") and prompt.strip() != "":
                try:
                    code_output = call_openai_api(prompt)
                    st.code(code_output, language="python")
                    st.info("✅ GPT returned Python code. You can copy and run it in a notebook or local script.")
                except Exception as api_error:
                    st.warning(f"⚠ OpenAI API error: {api_error}")
                    st.info("Switching to fallback manual plotting mode.")
                    manual_plot_builder(df)
            else:
                st.info("Use the manual plot builder below if you prefer not to use GPT.")
                manual_plot_builder(df)

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")

if __name__ == "__main__":
    main()
