# 이 함수만 복사하여 기존 csv_RfTx.py 파일의 analyze_RfTx_data 함수를 덮어쓰세요.

def analyze_RfTx_data(df):
    """Fw 데이터의 분석 로직을 담고 있는 함수"""
    # 데이터 전처리
    for col in df.columns:
        df[col] = df[col].apply(clean_string_format)

    df['RfTxStamp'] = pd.to_datetime(df['RfTxStamp'], errors='coerce')
    df['PassStatusNorm'] = df['RfTxPass'].fillna('').astype(str).str.strip().str.upper()

    summary_data = {}
    
    # RfTxPC 열이 없는 경우를 대비
    if 'RfTxPC' not in df.columns:
        df['RfTxPC'] = 'DefaultJig'

    # 'RfTxPC'를 기준으로 그룹화
    for jig, group in df.groupby('RfTxPC'):
        if group['RfTxStamp'].dt.date.dropna().empty:
            continue
        
        for d, day_group in group.groupby(group['RfTxStamp'].dt.date):
            if pd.isna(d):
                continue
            
            date_iso = pd.to_datetime(d).strftime("%Y-%m-%d")

            # --- 기존 로직 (SNumber 목록 생성) ---
            pass_sns_series = day_group.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
            pass_sns_ever_passed = pass_sns_series[pass_sns_series].index.tolist()

            pass_df = day_group[day_group['PassStatusNorm'] == 'O']
            pass_sns = pass_df['SNumber'].unique().tolist()
            pass_count = len(pass_df)
            
            false_defect_df = day_group[(day_group['PassStatusNorm'] == 'X') & (day_group['SNumber'].isin(pass_sns_ever_passed))]
            false_defect_count = false_defect_df.shape[0]
            false_defect_sns = false_defect_df['SNumber'].unique().tolist()
            
            true_defect_df = day_group[(day_group['PassStatusNorm'] == 'X') & (~day_group['SNumber'].isin(pass_sns_ever_passed))]
            true_defect_count = true_defect_df.shape[0]
            true_defect_sns = true_defect_df['SNumber'].unique().tolist()

            fail_df = day_group[day_group['PassStatusNorm'] == 'X']
            fail_count = len(fail_df)
            fail_sns = fail_df['SNumber'].unique().tolist()

            total_test = len(day_group)
            rate = 100 * pass_count / total_test if total_test > 0 else 0

            if jig not in summary_data:
                summary_data[jig] = {}
            
            # --- 최종 결과 데이터 구성 ---
            summary_data[jig][date_iso] = {
                'total_test': total_test,
                'pass': pass_count,
                'false_defect': false_defect_count,
                'true_defect': true_defect_count,
                'fail': fail_count,
                'pass_rate': f"{rate:.1f}%",
                
                # 상세 목록 (고유 SN)
                'pass_sns': pass_sns,
                'false_defect_sns': false_defect_sns,
                'true_defect_sns': true_defect_sns,
                'fail_sns': fail_sns,

                # ★★★ 수정/추가된 부분: 고유 SN 건수 계산 및 추가 ★★★
                'pass_unique_count': len(pass_sns),
                'false_defect_unique_count': len(false_defect_sns),
                'true_defect_unique_count': len(true_defect_sns),
                'fail_unique_count': len(fail_sns)
            }
    
    all_dates = sorted(list(df['RfTxStamp'].dt.date.dropna().unique()))
    return summary_data, all_dates
