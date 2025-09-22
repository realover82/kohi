#
# csv_Fw.py
# 이 파일은 Streamlit 앱에서 모듈로 사용됩니다.
# 따라서, 콘솔 출력 관련 코드(print, sys, io 등)는 모두 제거했습니다.

import pandas as pd
import numpy as np
import io
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# '="...' 형식의 문자열을 정리하는 함수
def clean_string_format(value):
    if isinstance(value, str) and value.startswith('="') and value.endswith('"'):
        return value[2:-1]
    return value

# read_csv_with_dynamic_header 함수는 Streamlit 앱에서 이미 정의되어 있으므로 필요 없습니다.
# 하지만 파일 로드 키워드만 Fw 데이터에 맞게 변경하여 함수를 하나로 통합합니다.
def read_csv_with_dynamic_header_for_Fw(uploaded_file):
    """Fw 데이터에 맞는 키워드로 헤더를 찾아 DataFrame을 로드하는 함수"""
    try:
        file_content = io.BytesIO(uploaded_file.getvalue())
        # UTF-8 인코딩으로 시도, 실패 시 다른 인코딩으로 대체 가능
        df_temp = pd.read_csv(file_content, header=None, nrows=100, encoding='utf-8')
        
        # 'Fw' 관련 필드명으로 키워드 수정
        keywords = ['SNumber', 'FwStamp', 'FwPC', 'FwPass']
        
        header_row = None
        for i, row in df_temp.iterrows():
            row_values = [str(x).strip() for x in row.values if pd.notna(x)]
            
            if all(keyword in row_values for keyword in keywords):
                header_row = i
                break
        
        if header_row is not None:
            file_content.seek(0)
            df = pd.read_csv(file_content, header=header_row, encoding='utf-8')
            return df
        else:
            return None
    except Exception as e:
        return None

# B파일 분석 로직 함수 (Fw 데이터를 분석하도록 수정)
def analyze_Fw_data(df):
    """Fw 데이터의 분석 로직을 담고 있는 함수"""
    # 데이터 전처리
    for col in df.columns:
        df[col] = df[col].apply(clean_string_format)

    df['FwStamp'] = pd.to_datetime(df['FwStamp'], errors='coerce')
    df['PassStatusNorm'] = df['FwPass'].fillna('').astype(str).str.strip().str.upper()

    summary_data = {}
    
    # FwPC 열이 없는 경우를 대비
    if 'FwPC' not in df.columns:
        df['FwPC'] = 'DefaultJig'

    # 'FwPC'를 기준으로 그룹화
    for jig, group in df.groupby('FwPC'):
        if group['FwStamp'].dt.date.dropna().empty:
            continue
        
        for d, day_group in group.groupby(group['FwStamp'].dt.date):
            if pd.isna(d):
                continue
            
            date_iso = pd.to_datetime(d).strftime("%Y-%m-%d")

            pass_sns_series = day_group.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
            pass_sns = pass_sns_series[pass_sns_series].index.tolist()

            pass_count = (day_group['PassStatusNorm'] == 'O').sum()
            
            false_defect_df = day_group[(day_group['PassStatusNorm'] == 'X') & (day_group['SNumber'].isin(pass_sns))]
            false_defect_count = false_defect_df.shape[0]
            false_defect_sns = false_defect_df['SNumber'].unique().tolist()
            
            true_defect_df = day_group[(day_group['PassStatusNorm'] == 'X') & (~day_group['SNumber'].isin(pass_sns))]
            true_defect_count = true_defect_df.shape[0]
            # 수정: 진성불량 상세 목록 추가
            true_defect_sns = true_defect_df['SNumber'].unique().tolist()

            total_test = len(day_group)
            fail_count = false_defect_count + true_defect_count
            
            # 수정: FAIL 상세 목록 추가
            fail_df = day_group[day_group['PassStatusNorm'] == 'X']
            fail_sns = fail_df['SNumber'].unique().tolist()

            rate = 100 * pass_count / total_test if total_test > 0 else 0

            if jig not in summary_data:
                summary_data[jig] = {}
            
            # 수정: 모든 상세 목록을 summary_data에 포함
            summary_data[jig][date_iso] = {
                'total_test': total_test,
                'pass': pass_count,
                'false_defect': false_defect_count,
                'true_defect': true_defect_count,
                'fail': fail_count,
                'pass_rate': f"{rate:.1f}%",
                'pass_sns': pass_sns,
                'false_defect_sns': false_defect_sns,
                'true_defect_sns': true_defect_sns,
                'fail_sns': fail_sns
            }
    
    all_dates = sorted(list(df['FwStamp'].dt.date.dropna().unique()))
    return summary_data, all_dates