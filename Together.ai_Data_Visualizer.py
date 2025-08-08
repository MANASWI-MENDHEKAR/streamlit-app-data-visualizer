import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from together import Together
import re
from io import BytesIO

# Load Together AI API Key
api_key = "tgp_v1_DAOBax7cmlPZpTutDGPvbU8qLNxNioI8d6rygKAGESk"
client = Together(api_key=api_key)

def extract_code_from_response(response_text):
    """Extract Python code block from Together AI's response."""
    code_blocks = re.findall(r"(?:python)?\n(.*?)", response_text, re.DOTALL)
    return code_blocks[0] if code_blocks else response_text.strip()

def get_code_from_together(prompt):
    """Call Together AI for Python plot code."""
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful data visualization assistant.\n"
                    "Always use matplotlib and seaborn.\n"
                    "Always create plots using fig, ax = plt.subplots().\n"
                    "Do NOT call plt.show().\n"
                    "Do NOT call plt.savefig().\n"
                    "Always assign the figure to fig.\n"
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content

def main():
    st.title("Together AI Data Visualizer")

    f = st.file_uploader("Upload your CSV file", type=["csv"])
    if not f:
        st.info("Please upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(f)

        st.subheader("Data Preview")
        st.dataframe(df.head())

        st.sidebar.subheader("Data Cleaning")

        if st.sidebar.checkbox("Handle Missing Values"):
            strategy = st.sidebar.selectbox("Strategy", ["drop", "mean", "median", "ffill"])
            if st.sidebar.button("Apply Missing Value Handling"):
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

        if st.sidebar.checkbox("Detect Outliers"):
            outlier_cols = st.sidebar.multiselect(
                "Columns to check for outliers", df.select_dtypes(include=np.number).columns.tolist()
            )
            if st.sidebar.button("Detect Outliers"):
                for col in outlier_cols:
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = df[(df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))]
                    st.write(f"Outliers in {col}: {len(outliers)} rows")

        user_prompt = st.text_input("Describe the plot you want:")

        if st.button("Generate Plot"):
            if not user_prompt.strip():
                cols = df.columns.tolist()
                if len(cols) >= 2:
                    user_prompt = (
                        f"Create a scatter plot with seaborn: "
                        f"fig, ax = plt.subplots(); sns.scatterplot(data=df, x='{cols[0]}', y='{cols[1]}', ax=ax)"
                    )
                else:
                    user_prompt = (
                        f"Create a histogram of '{cols[0]}': "
                        f"fig, ax = plt.subplots(); sns.histplot(df['{cols[0]}'], ax=ax)"
                    )

            try:
                raw_code = get_code_from_together(user_prompt)
                code = extract_code_from_response(raw_code)

                # Remove any accidental plt.show()
                code = re.sub(r'plt\.show\(\)', '', code)

                exec_globals = {"df": df, "sns": sns, "plt": plt, "pd": pd}
                exec_locals = {}
                exec(code, exec_globals, exec_locals)

                fig = exec_locals.get("fig", None)
                if fig is None:
                    fig = plt.gcf()

                buffer = BytesIO()
                fig.savefig(buffer, format="png", bbox_inches='tight')
                buffer.seek(0)

                st.session_state.plot_image = buffer.getvalue()

                plt.close(fig)

            except Exception as e:
                st.error(f"Error generating plot: {e}")

        if "plot_image" in st.session_state:
            st.subheader("Plot")
            st.image(st.session_state.plot_image)

            st.subheader("Export Plot")
            export_format = st.selectbox("Choose Format", ["PNG", "PDF", "SVG", "JPEG"])

            export_buffer = BytesIO()

            fig_bytes = BytesIO(st.session_state.plot_image)
            img = plt.imread(fig_bytes)

            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.axis("off")

            fig.savefig(export_buffer, format=export_format.lower(), bbox_inches='tight')
            plt.close(fig)

            export_buffer.seek(0)

            mime = {
                "PNG": "image/png",
                "PDF": "application/pdf",
                "SVG": "image/svg+xml",
                "JPEG": "image/jpeg"
            }[export_format]

            filename = f"plot.{export_format.lower()}"

            st.download_button(
                label=f"Download {export_format}",
                data=export_buffer,
                file_name=filename,
                mime=mime
            )

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

if __name__ == "__main__":
    main()
