#
# csv_Batadc.py
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

def read_csv_with_dynamic_header_for_Batadc(uploaded_file):
    """ 데이터에 맞는 키워드로 헤더를 찾아 DataFrame을 로드하는 함수"""
    try:
        file_content = io.BytesIO(uploaded_file.getvalue())
        # 다양한 인코딩 시도
        encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1']
        df = None
        for encoding in encodings:
            try:
                file_content.seek(0)
                df_temp = pd.read_csv(file_content, header=None, nrows=100, encoding=encoding)
                keywords = ['SNumber', 'BatadcStamp', 'BatadcPC', 'BatadcPass']
                header_row = None
                for i, row in df_temp.iterrows():
                    row_values = [str(x).strip() for x in row.values if pd.notna(x)]
                    if all(keyword in row_values for keyword in keywords):
                        header_row = i
                        break
                
                if header_row is not None:
                    file_content.seek(0)
                    df = pd.read_csv(file_content, header=header_row, encoding=encoding)
                    return df
            except Exception:
                continue
        return None
    except Exception as e:
        return None

def analyze_Batadc_data(df):
    """ 데이터의 분석 로직을 담고 있는 함수"""
    # 데이터 전처리
    for col in df.columns:
        df[col] = df[col].apply(clean_string_format)

    df['BatadcStamp'] = pd.to_datetime(df['BatadcStamp'], errors='coerce')
    df['PassStatusNorm'] = df['BatadcPass'].fillna('').astype(str).str.strip().str.upper()

    summary_data = {}

    if 'BatadcPC' not in df.columns:
        df['BatadcPC'] = 'DefaultJig'

    # 'PC'를 기준으로 그룹화
    for jig, group in df.groupby('BatadcPC'):
        group = group.dropna(subset=['BatadcStamp'])
        if group.empty:
            continue
        
        for d, day_group in group.groupby(group['BatadcStamp'].dt.date):
            if pd.isna(d):
                continue
            
            date_iso = pd.to_datetime(d).strftime("%Y-%m-%d")

            # 'O'가 한 번이라도 있었던 SNumber 목록
            ever_passed_sns = day_group[day_group['PassStatusNorm'] == 'O']['SNumber'].unique()

            # 각 카테고리별 데이터프레임 필터링
            pass_df = day_group[day_group['PassStatusNorm'] == 'O']
            fail_df = day_group[day_group['PassStatusNorm'] == 'X']
            false_defect_df = fail_df[fail_df['SNumber'].isin(ever_passed_sns)]
            true_defect_df = fail_df[~fail_df['SNumber'].isin(ever_passed_sns)]

            # 각 카테고리별 상세 목록 (고유 SN) 생성
            pass_sns = pass_df['SNumber'].unique().tolist()
            false_defect_sns = false_defect_df['SNumber'].unique().tolist()
            true_defect_sns = true_defect_df['SNumber'].unique().tolist()
            fail_sns = fail_df['SNumber'].unique().tolist()

            # 각 항목별 건수 (테스트 횟수)
            pass_count = len(pass_df)
            false_defect_count = len(false_defect_df)
            true_defect_count = len(true_defect_df)
            fail_count = len(fail_df)
            total_test = len(day_group)
            rate = 100 * pass_count / total_test if total_test > 0 else 0

            if jig not in summary_data:
                summary_data[jig] = {}
            
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
                'fail_sns': fail_sns,

                'pass_unique_count': len(pass_sns),
                'false_defect_unique_count': len(false_defect_sns),
                'true_defect_unique_count': len(true_defect_sns),
                'fail_unique_count': len(fail_sns)
            }
    
    all_dates = sorted(list(df['BatadcStamp'].dt.date.dropna().unique()))
    return summary_data, all_dates
