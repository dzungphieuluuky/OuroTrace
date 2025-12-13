import pandas as pd
from io import StringIO
import re

def load_and_process_data(data_source, is_filepath=False):
    """
    Loads experiment results data, calculates the number of generated tokens, 
    and prepares the DataFrame for analysis.

    Args:
        data_source (str): The file path to the CSV or the raw CSV string data.
        is_filepath (bool): True if data_source is a file path, False if it's 
                            the raw CSV string content.

    Returns:
        pd.DataFrame: A processed DataFrame ready for analysis, including 
                      the 'generated_tokens' feature.
    """
    if is_filepath:
        # Load from file path
        try:
            df = pd.read_csv(data_source)
        except FileNotFoundError:
            print(f"Lỗi: Không tìm thấy file tại đường dẫn '{data_source}'.")
            return pd.DataFrame()
    else:
        # Load from raw string data
        try:
            df = pd.read_csv(StringIO(data_source))
        except Exception as e:
            print(f"Lỗi khi đọc dữ liệu chuỗi: {e}")
            return pd.DataFrame()

    # --- 1. Tính toán generated_tokens ---
    # Đếm tokens (sử dụng khoảng trắng để ước tính) trong full_response.
    # Thêm 1 cho các chuỗi rỗng để tránh lỗi chia cho 0 trong một số phép tính sau này.
    df['generated_tokens'] = df['full_response'].apply(
        lambda x: len(re.findall(r'\S+', str(x))) if pd.notna(x) else 1
    )

    # --- 2. Chuẩn hóa kiểu dữ liệu ---
    # Chuyển đổi các cột số sang định dạng số chính xác
    
    # Cột 'prediction' có thể chứa số hoặc chuỗi 'Final: X', nên cần làm sạch trước
    def clean_prediction(pred):
        if isinstance(pred, str):
            # Cố gắng trích xuất số từ các định dạng có thể có
            match = re.search(r'[\d\.]+$', pred)
            return float(match.group(0)) if match else None
        return pred

    df['prediction'] = df['prediction'].apply(clean_prediction)
    
    # Ép kiểu các cột số
    numeric_cols = ['prediction', 'generation_time', 'ut_steps', 'expected_answer']
    for col in numeric_cols:
        # Sử dụng 'coerce' để biến các giá trị không hợp lệ thành NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ép kiểu các cột boolean/categorical
    df['is_correct'] = df['is_correct'].astype(bool)
    df['ut_steps'] = df['ut_steps'].astype('Int64') # Sử dụng Int64 để cho phép NaN
    
    # Cắt bớt khoảng trắng ở đầu/cuối của các cột chuỗi
    for col in ['full_response', 'task_type', 'difficulty', 'test_input']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print(f"✅ Dữ liệu đã được tải và xử lý thành công. Tổng số mẫu: {len(df)}")
    print(f"Thêm cột 'generated_tokens' với giá trị trung bình: {df['generated_tokens'].mean():.2f}")
    
    return df

# --- Dữ liệu Mẫu của bạn (Giả lập nội dung file CSV) ---
SAMPLE_CSV_DATA = """full_response,prediction,generation_time,ut_steps,task_type,difficulty,test_input,expected_answer,is_correct,test_id
"Current Sum: 0\nAdd 278: 0 + 278 = 278\nCurrent Sum: 278\nAdd 191: 278 + 191 = 469\nCurrent Sum: 469\nAdd 379: 469 + 379 = 848\nCurrent Sum: 848\nAdd 232: 848 + 232 = 1080\nFinal: 1080",1080,103.30002188682556,4,n_ary,4_ops,"[278, 191, 379, 232]",1080,True,0
"Current Sum: 0\nAdd 764: 0 + 764 = 764\nCurrent Sum: 764\nAdd 267: 764 + 267 = 1031\nCurrent Sum: 1031\nAdd 885: 1031 + 885 = 1916\nCurrent Sum: 1916\nAdd 977: 1916 + 977 = 2893\nFinal: 2893",2893,120.02091717720032,4,n_ary,4_ops,"[764, 267, 885, 977]",2893,True,1
"Current Sum: 0\nAdd 10: 0 + 10 = 10\nFinal: 10",10,20.5,2,n_ary,2_ops,"[10]",10,True,2
"Step 1: Found A->B\nStep 2: Did not find B->C. Must stop.",None,80.1,6,p_hop,6_hops,"A->B, D->E", "E", False, 3
"""
# --- Ví dụ cách sử dụng trong Cell phân tích của bạn ---
# df = load_and_process_data(SAMPLE_CSV_DATA, is_filepath=False)
# print("\nĐầu 5 dòng của DataFrame đã xử lý:")
# print(df[['full_response', 'ut_steps', 'generated_tokens', 'is_correct']].head())