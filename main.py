import json
import fireworks.client

import streamlit as st
import pandas as pd
import time
import io
from dotenv import load_dotenv
import os

load_dotenv()

fireworks.client.api_key = os.getenv('FIREWORKS_API')

prompt = """
Bạn là một chuyên gia phân tích ngôn ngữ tự nhiên, nhiệm vụ của bạn là phân tích cảm xúc cho từng comment dựa trên nội dung của bài post. Mỗi comment sẽ có một ID duy nhất được cung cấp từ input. Bạn cần phân loại cảm xúc, xác định các từ khóa chính ảnh hưởng đến cảm xúc đó, trích xuất thông tin về thương hiệu, xác định xem comment có phải quảng cáo hoặc spam không, dự đoán nhãn ngành nghề (label), giải thích ý định (intent) và góc nhìn (angle) của comment.

**Sentiment labels:**
- POSITIVE: Cảm xúc tích cực.
- NEGATIVE: Cảm xúc tiêu cực.
- NEUTRAL: Cảm xúc trung lập, không rõ ràng.
- MIXED: Nếu một comment chứa cả hai loại cảm xúc, tích cực và tiêu cực.

**Additional Tasks:**
1. **Từ khóa (keyword)**: Phải là một tính từ mô tả cảm xúc trong comment, không chứa tên riêng hoặc brand attribute.
2. **Brand attribute**: Nếu trong comment có nhắc đến thương hiệu (brand), hãy trích xuất và thêm trường "brand attribute" chứa danh sách các thương hiệu được đề cập. Nếu không có thương hiệu nào được nhắc đến, để giá trị của trường này là một chuỗi rỗng.
3. **Advertisement**: Xác định liệu comment có phải là quảng cáo không. Trả về giá trị `advertisement: true/false`.
4. **Spam**: Xác định liệu comment có phải là spam không. Trả về giá trị `spam: true/false`.
5. **Label**: Dự đoán nhãn ngành nghề của nội dung đang đề cập trong comment. Trả về trường `label`.
6. **Intent**: Giải thích ngắn gọn ý định của comment, trả về trường `intent`.
7. **Angle**: Giải thích ngắn gọn góc nhìn (perspective) của comment, trả về trường `angle`.

**Input format:**
{
  "post_content": "Nội dung bài post cần phân tích, là một đoạn văn bản.",
  "comments": [
    {
      "id": "Mã định danh duy nhất của comment.",
      "comment": "Nội dung comment cần phân tích, là một đoạn văn bản phản hồi về bài post."
    }
  ]
}

**Task:**
1. Đọc và phân tích nội dung bài post và từng comment.
2. Phân loại cảm xúc của mỗi comment vào một trong bốn nhóm: POSITIVE, NEGATIVE, NEUTRAL, hoặc MIXED.
3. Xác định các từ khóa chính trong mỗi comment đã ảnh hưởng đến cảm xúc. Từ khóa phải là tính từ và không chứa tên riêng hoặc brand attribute.
4. Trích xuất thông tin thương hiệu nếu có, nếu không có thì đặt brand attribute là một chuỗi rỗng.
5. Xác định xem comment có phải là quảng cáo (advertisement) hay không và trả về `true/false`.
6. Xác định xem comment có phải là spam hay không và trả về `true/false`.
7. Dự đoán nhãn ngành nghề (label) của nội dung trong comment và trả về trường `label` sử dụng tiếng Việt.
8. Giải thích ngắn gọn ý định của comment và trả về trường `intent` sử dụng tiếng Việt.
9. Giải thích ngắn gọn góc nhìn của comment và trả về trường `angle` sử dụng tiếng Việt.
10. Trả về kết quả dưới dạng danh sách JSON, trong đó mỗi object chứa id, comment, phân loại cảm xúc, danh sách từ khóa, brand attribute, advertisement, spam, label, intent, và angle.

**Output format:**
[
  {
    "id": "Mã định danh duy nhất của comment.",
    "comment": "Nội dung comment.",
    "sentiment": "POSITIVE, NEGATIVE, NEUTRAL, hoặc MIXED",
    "keyword": [
      "Danh sách các từ khóa ảnh hưởng đến cảm xúc, là tính từ mô tả cảm xúc, không chứa tên riêng hoặc brand."
    ],
    "brand_attribute": "Danh sách các thương hiệu được đề cập (nếu có), nếu không thì để chuỗi rỗng.",
    "advertisement": "true/false",
    "spam": "true/false",
    "label": "Nhãn ngành nghề của nội dung comment.",
    "intent": "Giải thích ngắn gọn ý định của comment.",
    "angle": "Giải thích ngắn gọn góc nhìn của comment."
  }
]

Chỉ trả về kết quả theo format ```json không giải thích gì thêm, phân tích dựa trên dữ liệu đã cung cấp, không sáng tạo gì thêm
No yapping
"""

def extract_json_from_string(input_string):
    try:
        start_index = input_string.find("[")
        end_index = input_string.rfind("]") + 1

        json_string = input_string[start_index:end_index]

        json_data = json.loads(json_string)

        return json_data
    except json.JSONDecodeError as e:
        # print(input_string)

        # print(f"Error parsing JSON: {e}")
        return None


def group_comments_by_title(df):
    grouped_data = []
    for title, group in df.groupby('Title'):
        post_data = {
            "post_content": title,
            "comments": []
        }
        for _, row in group.iterrows():
            comment_data = {
                "id": row['Id'],
                "comment": row['Content']
            }
            post_data["comments"].append(comment_data)
        grouped_data.append(post_data)
    return grouped_data


def read_and_group_csv(df):
    """Read CSV file and group comments by title."""
    return group_comments_by_title(df)


def call_inference(post_data, prompt):
    """Call the Llama model for inference on post data."""
    completion = fireworks.client.ChatCompletion.create(
        "accounts/fireworks/models/llama-v3-70b-instruct",
        messages=[
            {"role": "system", "content": "Bạn là một chuyên gia phân tích ngôn ngữ tự nhiên."},
            {"role": "user", "content": f"{prompt}\n\nInput: {str(post_data)}"}
        ],
        n=2,
        max_tokens=16000
    )
    result = completion.choices[0].message.content
    print(result)
    return extract_json_from_string(result)


def inference(grouped_comments, prompt):
    """Perform inference for each post and collect results."""
    results = []
    for post_data in grouped_comments:
        json_data = call_inference(post_data, prompt)
        if json_data:
            results.append(json_data)
    return results


def ensure_columns_exist(df, columns):
    """Ensure the specified columns exist in the DataFrame, creating them if necessary."""
    for field in columns:
        if field not in df.columns:
            df[field] = ""


def update_existing_rows(df, new_df):
    """Update existing rows in the DataFrame with new data from inference."""
    for index, row in new_df.iterrows():
        matching_rows = df[df['Id'] == row['id']]

        if not matching_rows.empty:
            df.loc[matching_rows.index, 'sentiment'] = row.get('sentiment', "")
            df.loc[matching_rows.index, 'keyword'] = ','.join(row.get('keyword', []))
            df.loc[matching_rows.index, 'brand_attribute'] = ','.join(row.get('brand_attribute', []))
            df.loc[matching_rows.index, 'advertisement'] = row.get('advertisement', "")
            df.loc[matching_rows.index, 'spam'] = row.get('spam', "")
            df.loc[matching_rows.index, 'label'] = row.get('label', "")
            df.loc[matching_rows.index, 'intent'] = row.get('intent', "")
            df.loc[matching_rows.index, 'angle'] = row.get('angle', "")
        else:
            new_row = {
                'Id': row['id'],
                'sentiment': row.get('sentiment', ""),
                'keyword': ','.join(row.get('keyword', [])),
                'brand_attribute': ','.join(row.get('brand_attribute', [])),
                'advertisement': row.get('advertisement', ""),
                'spam': row.get('spam', ""),
                'label': row.get('label', ""),
                'intent': row.get('intent', ""),
                'angle': row.get('angle', "")
            }
            df = df.append(new_row, ignore_index=True)
    return df


def merge_results_with_df(results, df_raw):
    """Merge inference results with CSV data and save to a new file."""

    flattened_results = [item for sublist in results for item in sublist]
    new_df = pd.DataFrame(flattened_results)
    expected_fields = ['sentiment', 'keyword', 'brand_attribute', 'advertisement', 'spam', 'label', 'intent', 'angle']
    ensure_columns_exist(df_raw, expected_fields)
    out_df = update_existing_rows(df_raw, new_df)
    out_df['spam'] = out_df['spam'].astype(bool)
    out_df['advertisement'] = out_df['advertisement'].astype(bool)
    out_df.loc[(out_df['spam'] == True) | (out_df['advertisement'] == True), 'sentiment'] = "NEUTRAL"
    return out_df


# Streamlit app
st.set_page_config(page_title="CSV/Excel Inference", page_icon="📄", layout="wide")

# Title section with description
st.title("📄 Upload CSV/Excel and Perform Inference")
st.write("This app allows you to upload a CSV or Excel file, perform inference on the data, and display the results.")

# File upload section with styled file uploader
uploaded_file = st.file_uploader("🔄 Choose a CSV or Excel file", type=["csv", "xlsx"])

# Processing the file if uploaded
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'xlsx':
        df = pd.read_excel(uploaded_file)

    # Check if necessary columns exist
    if not all(col in df.columns for col in ['Id', 'Title', 'Content', 'Type']):
        st.error("The file must contain `Id`, `Title`, `Content`, and `Type` columns.")
    else:
        st.success("File successfully uploaded! Below is a preview of the selected data.")

        # Data preview
        st.subheader("🔍 Data Preview")
        df_selected = df[['Id', 'Title', 'Content', 'Type']]
        st.dataframe(df_selected.style.highlight_max(axis=0))

        # Divider for better structure
        st.markdown("---")

        # Add process button
        st.subheader("🚀 Perform Inference")
        if st.button('Process Data'):
            with st.spinner('🔄 Performing inference...'):
                # Simulate a delay for the processing
                time.sleep(2)
                grouped_comments = read_and_group_csv(df_selected)
                results = inference(grouped_comments, prompt)
                result_df = merge_results_with_df(results, df_selected)

                st.success("Inference completed!")

                # Show the resulting DataFrame
                st.subheader("📊 Inference Results")

                st.dataframe(result_df)

                # Divider for better structure
                st.markdown("---")

                # Provide an option to download the processed file
                st.subheader("📥 Download Processed Results")
                file_format = st.radio("Choose the format for download:", ('Excel', 'CSV'))


                if file_format == 'Excel':
                    processed_file_path = "processed_results.xlsx"
                    towrite = io.BytesIO()
                    result_df.to_excel(towrite, encoding='utf-8', index=False, engine='xlsxwriter')
                    towrite.seek(0)
                    st.download_button('Download Processed Excel', towrite, file_name='processed_results.xlsx',
                                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                elif file_format == 'CSV':
                    processed_file_path = "processed_results.csv"
                    result_df.to_csv(processed_file_path, index=False)
                    with open(processed_file_path, 'rb') as f:
                        st.download_button('Download Processed CSV', f, file_name='processed_results.csv')
# Footer
st.markdown("<br><hr style='border:1px solid #e6e6e6;'>", unsafe_allow_html=True)
# st.write("💡 Developed using [Streamlit](https://streamlit.io/) - Build beautiful apps with less effort.")