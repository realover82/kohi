import pandas as pd
import numpy as np
from datetime import datetime

def analyze_data(df, date_col_name, jig_col_name):
    """
    주어진 DataFrame을 날짜와 지그(Jig) 기준으로 분석합니다.
    Args:
        df (pd.DataFrame): 분석할 원본 DataFrame.
        date_col_name (str): 날짜/시간 정보가 있는 컬럼명.
        jig_col_name (str): 지그(PC) 정보가 있는 컬럼명.
    Returns:
        tuple: 분석 결과 요약 데이터, 모든 날짜 목록, 실제로 사용된 지그 컬럼명.
    """
    if df.empty:
        return {}, [], jig_col_name

    df['PassStatusNorm'] = ""
    pass_col = next((col for col in ['PcbPass', 'FwPass', 'RfTxPass', 'SemiAssyPass', 'BatadcPass'] if col in df.columns), None)
    if pass_col:
        df['PassStatusNorm'] = df[pass_col].fillna('').astype(str).str.strip().str.upper()

    summary_data = {}
    
    used_jig_col_name = jig_col_name
    if jig_col_name not in df.columns or df[jig_col_name].isnull().all():
        used_jig_col_name = '__total_group__'
        df[used_jig_col_name] = '전체'

    if used_jig_col_name in df.columns and not df[used_jig_col_name].isnull().all():
        if 'SNumber' in df.columns and date_col_name in df.columns and not df[date_col_name].dt.date.dropna().empty:
            for jig, group in df.groupby(used_jig_col_name):
                for d, day_group in group.groupby(group[date_col_name].dt.date):
                    if pd.isna(d): continue
                    date_iso = pd.to_datetime(d).strftime("%Y-%m-%d")
                    
                    pass_sns_series = day_group.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
                    pass_sns = pass_sns_series[pass_sns_series].index.tolist()

                    false_defect_count = len(day_group[(day_group['PassStatusNorm'] == 'X') & (day_group['SNumber'].isin(pass_sns))]['SNumber'].unique())
                    true_defect_count = len(day_group[(day_group['PassStatusNorm'] == 'X') & (~day_group['SNumber'].isin(pass_sns))]['SNumber'].unique())
                    pass_count = len(pass_sns)
                    total_test = len(day_group['SNumber'].unique())
                    fail_count = total_test - pass_count

                    if jig not in summary_data:
                        summary_data[jig] = {}
                    summary_data[jig][date_iso] = {
                        'total_test': total_test,
                        'pass': pass_count,
                        'false_defect': false_defect_count,
                        'true_defect': true_defect_count,
                        'fail': fail_count,
                    }
    
    all_dates = sorted(list(df[date_col_name].dt.date.dropna().unique()))
    
    return summary_data, all_dates, used_jig_col_name
