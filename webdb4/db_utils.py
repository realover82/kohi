import pandas as pd
import io

def clean_string_format(value):
    """다양한 형태의 문자열 포맷을 정리하는 함수"""
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    # ="값" 형태 처리
    if value_str.startswith('="') and value_str.endswith('"'):
        return value_str[2:-1]
    
    # ""값"" 형태 처리
    if value_str.startswith('""') and value_str.endswith('""'):
        return value_str[2:-2]
    
    # "값" 형태 처리
    if value_str.startswith('"') and value_str.endswith('"') and len(value_str) > 2:
        return value_str[1:-1]
    
    return value_str

def read_csv_with_dynamic_header(uploaded_file, keywords):
    """
    업로드된 파일에서 동적으로 헤더를 찾아 DataFrame을 로드하는 함수.
    Args:
        uploaded_file: Streamlit의 file_uploader를 통해 업로드된 파일 객체.
        keywords (list): 헤더를 식별하기 위한 키워드 리스트.
    Returns:
        pd.DataFrame: 헤더를 찾아서 로드한 DataFrame. 실패 시 None 반환.
    """
    try:
        file_content = io.BytesIO(uploaded_file.getvalue())
        df_temp = pd.read_csv(file_content, header=None, nrows=100)
        
        header_row = None
        for i, row in df_temp.iterrows():
            row_values = [str(x).strip() for x in row.values if pd.notna(x)]
            if all(keyword in row_values for keyword in keywords):
                header_row = i
                break
        
        if header_row is not None:
            file_content.seek(0)
            df = pd.read_csv(file_content, header=header_row)
            return df
        else:
            return None
    except Exception:
        return None

def process_uploaded_csv(uploaded_file, tab_key):
    """
    업로드된 CSV 파일을 탭별로 처리하여 DataFrame을 반환하는 메인 함수.
    Args:
        uploaded_file: Streamlit의 file_uploader를 통해 업로드된 파일 객체.
        tab_key (str): 현재 탭을 식별하는 키 (pcb, fw, rftx, semi, func).
    Returns:
        pd.DataFrame: 처리된 DataFrame. 실패 시 None 반환.
    """
    if uploaded_file is None:
        return None
    
    # 탭별로 필요한 키워드 및 날짜 컬럼 정보 정의
    configs = {
        'pcb': {
            'keywords': ['SNumber', 'PcbStartTime', 'PcbMaxIrPwr', 'PcbPass'],
            'date_col': 'PcbStartTime'
        },
        'fw': {
            'keywords': ['SNumber', 'FwStamp', 'FwPC', 'FwPass'],
            'date_col': 'FwStamp'
        },
        'rftx': {
            'keywords': ['SNumber', 'RfTxStamp', 'RfTxPC', 'RfTxPass'],
            'date_col': 'RfTxStamp'
        },
        'semi': {
            'keywords': ['SNumber', 'SemiAssyStartTime', 'SemiAssyMaxBatVolt', 'SemiAssyPass'],
            'date_col': 'SemiAssyStartTime'
        },
        'func': {
            'keywords': ['SNumber', 'BatadcStamp', 'BatadcPC', 'BatadcPass'],
            'date_col': 'BatadcStamp'
        }
    }
    
    config = configs.get(tab_key)
    if not config:
        return None

    # 동적 헤더 로딩
    df = read_csv_with_dynamic_header(uploaded_file, config['keywords'])
    
    if df is None:
        return None

    # 데이터 전처리
    df.columns = df.columns.str.strip()
    df = df.applymap(clean_string_format)
    
    # 날짜 컬럼을 datetime으로 변환
    df[config['date_col']] = pd.to_datetime(df[config['date_col']], errors='coerce')
    
    return df
