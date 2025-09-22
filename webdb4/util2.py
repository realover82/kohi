import streamlit as st
import pandas as pd
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

    df_copy = df.copy()
    
    pass_col_found = False
    if 'PcbPass' in df_copy.columns:
        df_copy['PassStatusNorm'] = df_copy['PcbPass'].fillna('').astype(str).str.strip().str.upper()
        pass_col_found = True
    elif 'FwPass' in df_copy.columns:
        df_copy['PassStatusNorm'] = df_copy['FwPass'].fillna('').astype(str).str.strip().str.upper()
        pass_col_found = True
    elif 'RfTxPass' in df_copy.columns:
        df_copy['PassStatusNorm'] = df_copy['RfTxPass'].fillna('').astype(str).str.strip().str.upper()
        pass_col_found = True
    elif 'SemiAssyPass' in df_copy.columns:
        df_copy['PassStatusNorm'] = df_copy['SemiAssyPass'].fillna('').astype(str).str.strip().str.upper()
        pass_col_found = True
    elif 'BatadcPass' in df_copy.columns:
        df_copy['PassStatusNorm'] = df_copy['BatadcPass'].fillna('').astype(str).str.strip().str.upper()
        pass_col_found = True
    
    if not pass_col_found:
        st.warning("Pass 상태를 나타내는 컬럼이 없습니다.")
        return {}, [], jig_col_name

    summary_data = {}
    
    used_jig_col_name = jig_col_name
    if jig_col_name not in df_copy.columns or df_copy[jig_col_name].isnull().all() or df_copy[jig_col_name].nunique() < 2:
        used_jig_col_name = '__total_group__'
        df_copy[used_jig_col_name] = '전체'

    if used_jig_col_name in df_copy.columns and not df_copy[used_jig_col_name].isnull().all():
        if 'SNumber' in df_copy.columns and date_col_name in df_copy.columns and not df_copy[date_col_name].dt.date.dropna().empty:
            for jig, group in df_copy.groupby(used_jig_col_name):
                if not pd.api.types.is_datetime64_any_dtype(group[date_col_name]):
                    group.loc[:, date_col_name] = pd.to_datetime(group[date_col_name], errors='coerce')
                
                group = group.dropna(subset=[date_col_name]).copy()

                if group.empty:
                    continue

                for d, day_group in group.groupby(group[date_col_name].dt.date):
                    if pd.isna(d): continue
                    date_iso = pd.to_datetime(d).strftime("%Y-%m-%d")
                    
                    day_group = day_group[day_group['SNumber'].notna()]
                    if day_group.empty:
                        continue

                    if 'PassStatusNorm' not in day_group.columns:
                        continue
                        
                    pass_sns_series = day_group.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
                    pass_sns = pass_sns_series[pass_sns_series].index.tolist()

                    total_test_count = len(day_group['SNumber'].unique())
                    pass_count = len(pass_sns)
                    
                    false_defect_count = len(day_group[(day_group['PassStatusNorm'] == 'X') & (day_group['SNumber'].isin(pass_sns))]['SNumber'].unique())
                    true_defect_count = len(day_group[(day_group['PassStatusNorm'] == 'X') & (~day_group['SNumber'].isin(pass_sns))]['SNumber'].unique())
                    
                    fail_count = total_test_count - pass_count

                    if jig not in summary_data:
                        summary_data[jig] = {}
                    summary_data[jig][date_iso] = {
                        'total_test': total_test_count,
                        'pass': pass_count,
                        'false_defect': false_defect_count,
                        'true_defect': true_defect_count,
                        'fail': fail_count,
                    }
    
    all_dates = sorted(list(df_copy[date_col_name].dt.date.dropna().unique()))
    
    return summary_data, all_dates, used_jig_col_name

def display_analysis_result(analysis_key, table_name, date_col_name, selected_jig=None, used_jig_col=None):
    if st.session_state.analysis_results[analysis_key] is None or st.session_state.analysis_results[analysis_key].empty:
        st.warning("분석할 파일이 업로드되지 않았거나 데이터가 비어 있습니다.")
        return
    if st.session_state.analysis_data[analysis_key] is None:
        st.warning("분석 데이터가 준비되지 않았습니다. '분석 실행' 버튼을 눌러주세요.")
        return

    summary_data, all_dates, used_jig_col_name_from_state = st.session_state.analysis_data[analysis_key]
    
    if used_jig_col is None:
        used_jig_col = used_jig_col_name_from_state
        
    if not summary_data:
        st.warning("선택한 날짜에 해당하는 분석 데이터가 없습니다.")
        return

    st.markdown(f"### '{table_name}' 분석 리포트")
    
    jigs_to_display = [selected_jig] if selected_jig and selected_jig in summary_data else sorted(summary_data.keys())

    if not jigs_to_display:
        st.warning("선택한 PC (Jig)에 대한 데이터가 없습니다.")
        return
        
    kor_date_cols = [f"{d.strftime('%y%m%d')}" for d in all_dates]
    
    st.write(f"**분석 시간**: {st.session_state.analysis_time[analysis_key]}")
    st.markdown("---")

    all_reports_text = ""
    
    for jig in jigs_to_display:
        st.subheader(f"구분: {jig}")
        
        report_data = {
            '지표': ['총 테스트 수', 'PASS', '가성불량', '진성불량', 'FAIL']
        }
        
        for date_iso, date_str in zip([d.strftime('%Y-%m-%d') for d in all_dates], kor_date_cols):
            data_point = summary_data[jig].get(date_iso)
            if data_point:
                report_data[date_str] = [
                    data_point['total_test'],
                    data_point['pass'],
                    data_point['false_defect'],
                    data_point['true_defect'],
                    data_point['fail']
                ]
            else:
                report_data[date_str] = ['N/A'] * 5
        
        report_df = pd.DataFrame(report_data)
        st.table(report_df)
        all_reports_text += report_df.to_csv(index=False) + "\n"

        st.markdown("#### 상세 내역")
        df_filtered = st.session_state.analysis_results[analysis_key]
        
        if used_jig_col == '__total_group__':
            jig_filtered_df = df_filtered.copy()
        elif used_jig_col not in df_filtered.columns:
            st.warning(f"데이터프레임에 '{used_jig_col}' 컬럼이 없어 상세 내역을 표시할 수 없습니다.")
            continue
        else:
            jig_filtered_df = df_filtered[df_filtered[used_jig_col] == jig].copy()
        
        if 'SNumber' not in jig_filtered_df.columns:
            st.warning("'SNumber' 컬럼이 없어 상세 내역을 표시할 수 없습니다.")
            continue
        jig_filtered_df = jig_filtered_df[jig_filtered_df['SNumber'].notna()]
        
        if 'PassStatusNorm' not in jig_filtered_df.columns:
            if 'PcbPass' in jig_filtered_df.columns:
                jig_filtered_df['PassStatusNorm'] = jig_filtered_df['PcbPass'].fillna('').astype(str).str.strip().str.upper()
            elif 'FwPass' in jig_filtered_df.columns:
                jig_filtered_df['PassStatusNorm'] = jig_filtered_df['FwPass'].fillna('').astype(str).str.strip().str.upper()
            elif 'RfTxPass' in jig_filtered_df.columns:
                jig_filtered_df['PassStatusNorm'] = jig_filtered_df['RfTxPass'].fillna('').astype(str).str.strip().str.upper()
            elif 'SemiAssyPass' in jig_filtered_df.columns:
                jig_filtered_df['PassStatusNorm'] = jig_filtered_df['SemiAssyPass'].fillna('').astype(str).str.strip().str.upper()
            elif 'BatadcPass' in jig_filtered_df.columns:
                jig_filtered_df['PassStatusNorm'] = jig_filtered_df['BatadcPass'].fillna('').astype(str).str.strip().str.upper()
            else:
                st.warning("PassStatusNorm 컬럼이 없어 상세 내역을 표시할 수 없습니다.")
                continue

        pass_sns = jig_filtered_df.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
        pass_sns = pass_sns[pass_sns].index.tolist()
        with st.expander(f"PASS ({len(pass_sns)}건)", expanded=False):
            st.text("\n".join(pass_sns))
        
        false_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        with st.expander(f"가성불량 ({len(false_defect_sns)}건)", expanded=False):
            st.text("\n".join(false_defect_sns))
            
        true_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (~jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        with st.expander(f"진성불량 ({len(true_defect_sns)}건)", expanded=False):
            st.text("\n".join(true_defect_sns))

        all_snumbers = jig_filtered_df['SNumber'].unique().tolist()
        all_fail_sns = list(set(all_snumbers) - set(pass_sns))
        with st.expander(f"FAIL ({len(all_fail_sns)}건)", expanded=False):
            st.text("\n".join(all_fail_sns))
        
        st.markdown("---")

    st.success("분석 완료! 결과가 저장되었습니다.")

    st.download_button(
        label="분석 결과 다운로드",
        data=all_reports_text.encode('utf-8-sig'),
        file_name=f"{table_name}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"download_{analysis_key}"
    )

    st.markdown("---")
    st.subheader("그래프")
    
    chart_data_raw = report_df.set_index('지표').T
    chart_data = chart_data_raw[['총 테스트 수', 'PASS', 'FAIL', '가성불량', '진성불량']].copy()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("꺾은선 그래프 보기", key=f"line_chart_btn_{analysis_key}"):
            st.session_state.show_line_chart[analysis_key] = not st.session_state.show_line_chart.get(analysis_key, False)
        if st.session_state.show_line_chart.get(analysis_key, False):
            st.line_chart(chart_data)
    with col2:
        if st.button("막대 그래프 보기", key=f"bar_chart_btn_{analysis_key}"):
            st.session_state.show_bar_chart[analysis_key] = not st.session_state.show_bar_chart.get(analysis_key, False)
        if st.session_state.show_bar_chart.get(analysis_key, False):
            st.bar_chart(chart_data)
