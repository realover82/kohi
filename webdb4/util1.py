import streamlit as st
import pandas as pd
import io
import warnings

warnings.filterwarnings('ignore')

def clean_string_format(value):
    """
    다양한 형태의 문자열 포맷을 정리하는 함수.
    예: '="O"', '""X""', '"O"' 등을 'O'로 변환
    """
    if pd.isna(value):
        return value
    
    value_str = str(value).strip()
    
    # "값" 형태 처리
    if value_str.startswith('"') and value_str.endswith('"'):
        # =""값"" 또는 ""값"" 형태 처리
        if value_str.startswith('="') or value_str.startswith('""'):
            return value_str.strip('="').strip('"').strip()
        return value_str.strip('"').strip()

    # =값 형태 처리
    if value_str.startswith('='):
        return value_str.strip('=').strip()
    
    return value_str

def find_header_and_read_csv(uploaded_file, keywords):
    """
    업로드된 파일에서 동적으로 헤더를 찾아 DataFrame을 로드하는 함수.
    """
    try:
        file_content = io.BytesIO(uploaded_file.getvalue())
        df_temp = pd.read_csv(file_content, header=None, nrows=100)
        
        header_row = None
        for i, row in df_temp.iterrows():
            row_values_lower = [str(x).lower() for x in row.values if pd.notna(x)]
            
            if all(kw.lower() in row_values_lower for kw in keywords):
                header_row = i
                break
        
        if header_row is not None:
            file_content.seek(0)
            df = pd.read_csv(file_content, header=header_row)
            
            df.columns = df.columns.str.strip()
            df.columns = [col.strip().replace(' ', '') for col in df.columns]

            col_mapping = {}
            for col in df.columns:
                if 'snumber' in col.lower(): col_mapping[col] = 'SNumber'
                elif 'pcbstime' in col.lower(): col_mapping[col] = 'PcbStartTime'
                elif 'pcbpass' in col.lower(): col_mapping[col] = 'PcbPass'
                elif 'pcbmaxirpwr' in col.lower(): col_mapping[col] = 'PcbMaxIrPwr'
                
                elif 'fwstamp' in col.lower(): col_mapping[col] = 'FwStamp'
                elif 'fwpc' in col.lower(): col_mapping[col] = 'FwPC'
                elif 'fwpass' in col.lower(): col_mapping[col] = 'FwPass'
                
                elif 'rftxstamp' in col.lower(): col_mapping[col] = 'RfTxStamp'
                elif 'rftxpc' in col.lower(): col_mapping[col] = 'RfTxPC'
                elif 'rftxpass' in col.lower(): col_mapping[col] = 'RfTxPass'

                elif 'semiassystarttime' in col.lower(): col_mapping[col] = 'SemiAssyStartTime'
                elif 'semiassymaxbatvolt' in col.lower(): col_mapping[col] = 'SemiAssyMaxBatVolt'
                elif 'semiassypass' in col.lower(): col_mapping[col] = 'SemiAssyPass'
                
                elif 'batadcstamp' in col.lower(): col_mapping[col] = 'BatadcStamp'
                elif 'batadcpc' in col.lower(): col_mapping[col] = 'BatadcPC'
                elif 'batadcpass' in col.lower(): col_mapping[col] = 'BatadcPass'
            
            df = df.rename(columns=col_mapping)
            
            return df, None
        else:
            return None, "헤더 키워드를 찾을 수 없습니다. 올바른 형식의 파일인지 확인해주세요."
            
    except Exception as e:
        return None, f"파일 로드 중 오류가 발생했습니다: {e}"

def get_jig_and_date_inputs(uploaded_file, key):
    """업로드된 CSV 파일을 처리하고 DataFrame을 반환하는 메인 함수"""
    if uploaded_file is None:
        return None

    # 탭별로 필요한 키워드 정의
    keywords_map = {
        'pcb': ['SNumber', 'PcbStartTime', 'PcbPass'],
        'fw': ['SNumber', 'FwStamp', 'FwPass'],
        'rftx': ['SNumber', 'RfTxStamp', 'RfTxPass'],
        'semi': ['SNumber', 'SemiAssyStartTime', 'SemiAssyPass'],
        'func': ['SNumber', 'BatadcStamp', 'BatadcPass'],
    }
    
    df, error_message = find_header_and_read_csv(uploaded_file, keywords_map.get(key, []))

    if error_message:
        st.warning(error_message)
        return None

    if df is not None:
        # 데이터 전처리
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].apply(clean_string_format)
            except:
                pass # 오류 발생 시 스킵

        # PassStatus 컬럼에 대해서만 특별히 문자열 정리
        if 'PcbPass' in df.columns:
            df['PcbPass'] = df['PcbPass'].apply(clean_string_format).fillna('').astype(str)
        if 'FwPass' in df.columns:
            df['FwPass'] = df['FwPass'].apply(clean_string_format).fillna('').astype(str)
        if 'RfTxPass' in df.columns:
            df['RfTxPass'] = df['RfTxPass'].apply(clean_string_format).fillna('').astype(str)
        if 'SemiAssyPass' in df.columns:
            df['SemiAssyPass'] = df['SemiAssyPass'].apply(clean_string_format).fillna('').astype(str)
        if 'BatadcPass' in df.columns:
            df['BatadcPass'] = df['BatadcPass'].apply(clean_string_format).fillna('').astype(str)

        return df

    return None

def create_tabs_config():
    return {
        'pcb': {
            'header': "파일 PCB (Pcb_Process)",
            'date_col': 'PcbStartTime',
            'jig_col': 'PcbMaxIrPwr',
            'pass_col': 'PcbPass'
        },
        'fw': {
            'header': "파일 Fw (Fw_Process)",
            'date_col': 'FwStamp',
            'jig_col': 'FwPC',
            'pass_col': 'FwPass'
        },
        'rftx': {
            'header': "파일 RfTx (RfTx_Process)",
            'date_col': 'RfTxStamp',
            'jig_col': 'RfTxPC',
            'pass_col': 'RfTxPass'
        },
        'semi': {
            'header': "파일 Semi (SemiAssy_Process)",
            'date_col': 'SemiAssyStartTime',
            'jig_col': 'SemiAssyMaxBatVolt',
            'pass_col': 'SemiAssyPass'
        },
        'func': {
            'header': "파일 Func (Func_Process)",
            'date_col': 'BatadcStamp',
            'jig_col': 'BatadcPC',
            'pass_col': 'BatadcPass'
        }
    }
