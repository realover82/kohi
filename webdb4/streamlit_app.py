import streamlit as st
import pandas as pd
from datetime import datetime, date
import sqlite3
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# SQLite 연결 함수
@st.cache_resource
def get_connection():
    try:
        db_path = "./db/SJ_TM2360E_v2.sqlite3"
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return conn
    except Exception as e:
        st.error(f"데이터베이스 연결에 실패했습니다: {e}")
        return None

# 데이터베이스에서 테이블을 읽어 DataFrame으로 반환하는 함수
def read_data_from_db(conn, table_name):
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"테이블 '{table_name}'에서 데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

# analyze_data 함수
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
    # DataFrame이 비어 있으면 빈 결과를 반환
    if df.empty:
        return {}, [], jig_col_name

    # PassStatusNorm 컬럼 생성
    df['PassStatusNorm'] = ""
    if 'PcbPass' in df.columns:
        df['PassStatusNorm'] = df['PcbPass'].fillna('').astype(str).str.strip().str.upper()
    elif 'FwPass' in df.columns:
        df['PassStatusNorm'] = df['FwPass'].fillna('').astype(str).str.strip().str.upper()
    elif 'RfTxPass' in df.columns:
        df['PassStatusNorm'] = df['RfTxPass'].fillna('').astype(str).str.strip().str.upper()
    elif 'SemiAssyPass' in df.columns:
        df['PassStatusNorm'] = df['SemiAssyPass'].fillna('').astype(str).str.strip().str.upper()
    elif 'BatadcPass' in df.columns:
        df['PassStatusNorm'] = df['BatadcPass'].fillna('').astype(str).str.strip().str.upper()

    summary_data = {}
    
    # 지그(PC) 컬럼에 데이터가 없는 경우 '전체'를 대체 컬럼으로 사용 (PCB 탭의 경우)
    used_jig_col_name = jig_col_name
    if jig_col_name not in df.columns or df[jig_col_name].isnull().all():
        used_jig_col_name = '__total_group__'
        df[used_jig_col_name] = '전체'

    # 지그(PC) 컬럼이 존재하고 데이터가 있는 경우에만 그룹 분석 실행
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


def display_analysis_result(analysis_key, table_name, date_col_name, selected_jig=None, used_jig_col=None):
    if st.session_state.analysis_results[analysis_key].empty:
        st.warning("선택한 날짜에 해당하는 분석 데이터가 없습니다.")
        return

    summary_data, all_dates, used_jig_col_name_from_state = st.session_state.analysis_data[analysis_key]
    
    # 실제 사용된 지그 컬럼명을 우선적으로 사용
    if used_jig_col is None:
        used_jig_col = used_jig_col_name_from_state
        
    if not summary_data:
        st.warning("선택한 날짜에 해당하는 분석 데이터가 없습니다.")
        return

    st.markdown(f"### '{table_name}' 분석 리포트")
    
    # 선택된 jig에 따라 데이터 필터링
    jigs_to_display = [selected_jig] if selected_jig and selected_jig in summary_data else sorted(summary_data.keys())

    if not jigs_to_display:
        st.warning("선택한 PC (Jig)에 대한 데이터가 없습니다.")
        return
        
    kor_date_cols = [f"{d.strftime('%y%m%d')}" for d in all_dates]
    
    st.write(f"**분석 시간**: {st.session_state.analysis_time[analysis_key]}")
    st.markdown("---")

    all_reports_text = ""
    
    # 보고서 테이블 표시
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

        # 상세 내역 표시
        st.markdown("#### 상세 내역")
        df_filtered = st.session_state.analysis_results[analysis_key]
        
        # 현재 지그에 해당하는 데이터만 필터링
        jig_filtered_df = df_filtered[df_filtered[used_jig_col] == jig].copy()
        
        # PASS 상세 내역
        pass_sns = jig_filtered_df.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
        pass_sns = pass_sns[pass_sns].index.tolist()
        with st.expander(f"PASS ({len(pass_sns)}건)", expanded=False):
            st.text("\n".join(pass_sns))
        
        # 가성불량 (False Defect) 상세 내역
        false_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        with st.expander(f"가성불량 ({len(false_defect_sns)}건)", expanded=False):
            st.text("\n".join(false_defect_sns))
            
        # 진성불량 (True Defect) 상세 내역
        true_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (~jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        with st.expander(f"진성불량 ({len(true_defect_sns)}건)", expanded=False):
            st.text("\n".join(true_defect_sns))

        # FAIL 상세 내역
        fail_sns = jig_filtered_df['SNumber'].unique().tolist()
        all_fail_sns = list(set(fail_sns) - set(pass_sns))
        with st.expander(f"FAIL ({len(all_fail_sns)}건)", expanded=False):
            st.text("\n".join(all_fail_sns))
        
        st.markdown("---") # 각 지그 구분선

    st.success("분석 완료! 결과가 저장되었습니다.")

    st.download_button(
        label="분석 결과 다운로드",
        data=all_reports_text.encode('utf-8-sig'),
        file_name=f"{table_name}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key=f"download_{analysis_key}"
    )

    # 차트 버튼
    st.markdown("---")
    st.subheader("그래프")
    
    chart_data_raw = report_df.set_index('지표').T
    chart_data = chart_data_raw[['총 테스트 수', 'PASS', 'FAIL']].copy()

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


def main():
    st.set_page_config(layout="wide")
    st.title("리모컨 생산 데이터 분석 툴")
    st.markdown("---")

    conn = get_connection()
    if conn is None:
        return

    # 세션 상태 초기화
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {
            'pcb': None, 'fw': None, 'rftx': None, 'semi': None, 'func': None
        }
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {
            'pcb': None, 'fw': None, 'rftx': None, 'semi': None, 'func': None
        }
    if 'analysis_time' not in st.session_state:
        st.session_state.analysis_time = {
            'pcb': None, 'fw': None, 'rftx': None, 'semi': None, 'func': None
        }
    if 'last_analyzed_key' not in st.session_state:
        st.session_state['last_analyzed_key'] = None
    if 'jig_col_mapping' not in st.session_state:
        st.session_state['jig_col_mapping'] = {
            'pcb': 'PcbMaxIrPwr',
            'fw': 'FwPC',
            'rftx': 'RfTxPC',
            'semi': 'SemiAssyMaxBatVolt',
            'func': 'BatadcPC',
        }
    if 'show_line_chart' not in st.session_state:
        st.session_state.show_line_chart = {}
    if 'show_bar_chart' not in st.session_state:
        st.session_state.show_bar_chart = {}
    if 'snumber_search' not in st.session_state:
        st.session_state.snumber_search = {
            'pcb': {'results': pd.DataFrame(), 'show': False},
            'fw': {'results': pd.DataFrame(), 'show': False},
            'rftx': {'results': pd.DataFrame(), 'show': False},
            'semi': {'results': pd.DataFrame(), 'show': False},
            'func': {'results': pd.DataFrame(), 'show': False},
        }
    if 'original_db_view' not in st.session_state:
        st.session_state.original_db_view = {
            'pcb': {'results': pd.DataFrame(), 'show': False},
            'fw': {'results': pd.DataFrame(), 'show': False},
            'rftx': {'results': pd.DataFrame(), 'show': False},
            'semi': {'results': pd.DataFrame(), 'show': False},
            'func': {'results': pd.DataFrame(), 'show': False},
        }
    
    try:
        # 모든 탭에서 공통으로 사용할 원본 데이터를 한 번만 불러옵니다.
        df_all_data = pd.read_sql_query("SELECT * FROM historyinspection;", conn)
    except Exception as e:
        st.error(f"데이터베이스에서 'historyinspection' 테이블을 불러오는 중 오류가 발생했습니다: {e}")
        return

    # 모든 날짜 관련 컬럼을 datetime 객체로 미리 변환
    df_all_data['PcbStartTime_dt'] = pd.to_datetime(df_all_data['PcbStartTime'], errors='coerce')
    df_all_data['FwStamp_dt'] = pd.to_datetime(df_all_data['FwStamp'], errors='coerce')
    df_all_data['RfTxStamp_dt'] = pd.to_datetime(df_all_data['RfTxStamp'], errors='coerce')
    df_all_data['SemiAssyStartTime_dt'] = pd.to_datetime(df_all_data['SemiAssyStartTime'], errors='coerce')
    df_all_data['BatadcStamp_dt'] = pd.to_datetime(df_all_data['BatadcStamp'], errors='coerce')
    
    # --- 탭별 분석 기능 ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["파일 PCB 분석", "파일 Fw 분석", "파일 RfTx 분석", "파일 Semi 분석", "파일 Func 분석"])
    
    try:
        with tab1:
            st.header("파일 PCB (Pcb_Process)")
            
            # PC (Jig) 선택 기능 추가
            pc_col_name = 'PcbMaxIrPwr'
            unique_pc_pcb = df_all_data[pc_col_name].dropna().unique()
            pc_options_pcb = ['모든 PC'] + sorted(list(unique_pc_pcb))
            selected_pc_pcb = st.selectbox("PC (Jig) 선택", pc_options_pcb, key="pc_select_pcb")

            df_dates = df_all_data['PcbStartTime_dt'].dt.date.dropna()
            min_date = df_dates.min() if not df_dates.empty else date.today()
            max_date = df_dates.max() if not df_dates.dropna().empty else date.today()
            selected_dates = st.date_input("날짜 범위 선택", value=(min_date, max_date), key="dates_pcb")
            
            if st.button("분석 실행", key="analyze_pcb"):
                with st.spinner("데이터 분석 및 저장 중..."):
                    if len(selected_dates) == 2:
                        start_date, end_date = selected_dates
                        df_filtered = df_all_data[
                            (df_all_data['PcbStartTime_dt'].dt.date >= start_date) &
                            (df_all_data['PcbStartTime_dt'].dt.date <= end_date)
                        ].copy()
                        if selected_pc_pcb != '모든 PC':
                            df_filtered = df_filtered[df_filtered[pc_col_name] == selected_pc_pcb].copy()
                    else:
                        st.warning("날짜 범위를 올바르게 선택해주세요.")
                        df_filtered = pd.DataFrame()
                    
                    st.session_state.analysis_results['pcb'] = df_filtered
                    st.session_state.analysis_data['pcb'] = analyze_data(df_filtered, 'PcbStartTime_dt', pc_col_name)
                    st.session_state.analysis_time['pcb'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.session_state['last_analyzed_key'] = 'pcb'
                st.success("분석 완료! 결과가 저장되었습니다.")

            # 분석 결과가 존재하면 항상 표시
            if st.session_state.analysis_results['pcb'] is not None:
                display_analysis_result('pcb', 'Pcb_Process', 'PcbStartTime_dt',
                                        selected_jig=selected_pc_pcb if selected_pc_pcb != '모든 PC' else None)
            
            st.markdown("---")
            st.markdown("#### PCB 데이터 조회")
            snumber_query_pcb = st.text_input("SNumber를 입력하세요 (PCB)", key="snumber_search_bar_pcb")
            
            col_search_btn, col_view_btn = st.columns(2)
            with col_search_btn:
                if st.button("SNumber 검색 실행", key="snumber_search_btn_pcb"):
                    st.session_state.snumber_search['pcb']['show'] = True
                    if snumber_query_pcb:
                        with st.spinner("데이터베이스에서 SNumber 검색 중..."):
                            filtered_df = st.session_state.analysis_results['pcb'][
                                st.session_state.analysis_results['pcb']['SNumber'].fillna('').astype(str).str.contains(snumber_query_pcb, case=False, na=False)
                            ]
                        if not filtered_df.empty:
                            st.success(f"'{snumber_query_pcb}'에 대한 {len(filtered_df)}건의 검색 결과를 찾았습니다.")
                            st.session_state.snumber_search['pcb']['results'] = filtered_df
                        else:
                            st.warning(f"'{snumber_query_pcb}'에 대한 검색 결과가 없습니다.")
                            st.session_state.snumber_search['pcb']['results'] = pd.DataFrame()
                    else:
                        st.warning("SNumber를 입력해주세요.")
                        st.session_state.snumber_search['pcb']['results'] = pd.DataFrame()

            with col_view_btn:
                if st.button("원본 DB 조회", key="view_last_db_pcb"):
                    st.session_state.original_db_view['pcb']['show'] = True
                    if st.session_state.analysis_results['pcb'] is not None:
                        st.success(f"PCB 탭의 원본 데이터를 조회합니다.")
                        st.session_state.original_db_view['pcb']['results'] = st.session_state.analysis_results['pcb'].copy()
                    else:
                        st.warning("먼저 PCB 탭에서 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                        st.session_state.original_db_view['pcb']['results'] = pd.DataFrame()

            if st.session_state.snumber_search['pcb']['show'] and not st.session_state.snumber_search['pcb']['results'].empty:
                st.dataframe(st.session_state.snumber_search['pcb']['results'].reset_index(drop=True))

            if st.session_state.original_db_view['pcb']['show'] and not st.session_state.original_db_view['pcb']['results'].empty:
                st.dataframe(st.session_state.original_db_view['pcb']['results'].reset_index(drop=True))


        with tab2:
            st.header("파일 Fw (Fw_Process)")

            pc_col_name = 'FwPC'
            unique_pc_fw = df_all_data[pc_col_name].dropna().unique()
            pc_options_fw = ['모든 PC'] + sorted(list(unique_pc_fw))
            selected_pc_fw = st.selectbox("PC (Jig) 선택", pc_options_fw, key="pc_select_fw")

            df_dates = df_all_data['FwStamp_dt'].dt.date.dropna()
            min_date = df_dates.min() if not df_dates.empty else date.today()
            max_date = df_dates.max() if not df_dates.dropna().empty else date.today()
            selected_dates = st.date_input("날짜 범위 선택", value=(min_date, max_date), key="dates_fw")
            
            if st.button("분석 실행", key="analyze_fw"):
                with st.spinner("데이터 분석 및 저장 중..."):
                    if len(selected_dates) == 2:
                        start_date, end_date = selected_dates
                        df_filtered = df_all_data[
                            (df_all_data['FwStamp_dt'].dt.date >= start_date) &
                            (df_all_data['FwStamp_dt'].dt.date <= end_date)
                        ].copy()
                        if selected_pc_fw != '모든 PC':
                            df_filtered = df_filtered[df_filtered[pc_col_name] == selected_pc_fw].copy()
                    else:
                        st.warning("날짜 범위를 올바르게 선택해주세요.")
                        df_filtered = pd.DataFrame()

                    st.session_state.analysis_results['fw'] = df_filtered
                    st.session_state.analysis_data['fw'] = analyze_data(df_filtered, 'FwStamp_dt', pc_col_name)
                    st.session_state.analysis_time['fw'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.session_state['last_analyzed_key'] = 'fw'
                st.success("분석 완료! 결과가 저장되었습니다.")

            # 분석 결과가 존재하면 항상 표시
            if st.session_state.analysis_results['fw'] is not None:
                display_analysis_result('fw', 'Fw_Process', 'FwStamp_dt',
                                        selected_jig=selected_pc_fw if selected_pc_fw != '모든 PC' else None)

            st.markdown("---")
            st.markdown("#### Fw 데이터 조회")
            snumber_query_fw = st.text_input("SNumber를 입력하세요 (Fw)", key="snumber_search_bar_fw")
            
            col_search_btn, col_view_btn = st.columns(2)
            with col_search_btn:
                if st.button("SNumber 검색 실행", key="snumber_search_btn_fw"):
                    st.session_state.snumber_search['fw']['show'] = True
                    if snumber_query_fw:
                        with st.spinner("데이터베이스에서 SNumber 검색 중..."):
                            filtered_df = st.session_state.analysis_results['fw'][
                                st.session_state.analysis_results['fw']['SNumber'].fillna('').astype(str).str.contains(snumber_query_fw, case=False, na=False)
                            ]
                        if not filtered_df.empty:
                            st.success(f"'{snumber_query_fw}'에 대한 {len(filtered_df)}건의 검색 결과를 찾았습니다.")
                            st.session_state.snumber_search['fw']['results'] = filtered_df
                        else:
                            st.warning(f"'{snumber_query_fw}'에 대한 검색 결과가 없습니다.")
                            st.session_state.snumber_search['fw']['results'] = pd.DataFrame()
                    else:
                        st.warning("SNumber를 입력해주세요.")
                        st.session_state.snumber_search['fw']['results'] = pd.DataFrame()

            with col_view_btn:
                if st.button("원본 DB 조회", key="view_last_db_fw"):
                    st.session_state.original_db_view['fw']['show'] = True
                    if st.session_state.analysis_results['fw'] is not None:
                        st.success(f"Fw 탭의 원본 데이터를 조회합니다.")
                        st.session_state.original_db_view['fw']['results'] = st.session_state.analysis_results['fw'].copy()
                    else:
                        st.warning("먼저 Fw 탭에서 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                        st.session_state.original_db_view['fw']['results'] = pd.DataFrame()

            if st.session_state.snumber_search['fw']['show'] and not st.session_state.snumber_search['fw']['results'].empty:
                st.dataframe(st.session_state.snumber_search['fw']['results'].reset_index(drop=True))

            if st.session_state.original_db_view['fw']['show'] and not st.session_state.original_db_view['fw']['results'].empty:
                st.dataframe(st.session_state.original_db_view['fw']['results'].reset_index(drop=True))

        with tab3:
            st.header("파일 RfTx (RfTx_Process)")

            pc_col_name = 'RfTxPC'
            unique_pc_rftx = df_all_data[pc_col_name].dropna().unique()
            pc_options_rftx = ['모든 PC'] + sorted(list(unique_pc_rftx))
            selected_pc_rftx = st.selectbox("PC (Jig) 선택", pc_options_rftx, key="pc_select_rftx")

            df_dates = df_all_data['RfTxStamp_dt'].dt.date.dropna()
            min_date = df_dates.min() if not df_dates.empty else date.today()
            max_date = df_dates.max() if not df_dates.dropna().empty else date.today()
            selected_dates = st.date_input("날짜 범위 선택", value=(min_date, max_date), key="dates_rftx")
            
            if st.button("분석 실행", key="analyze_rftx"):
                with st.spinner("데이터 분석 및 저장 중..."):
                    if len(selected_dates) == 2:
                        start_date, end_date = selected_dates
                        df_filtered = df_all_data[
                            (df_all_data['RfTxStamp_dt'].dt.date >= start_date) &
                            (df_all_data['RfTxStamp_dt'].dt.date <= end_date)
                        ].copy()
                        if selected_pc_rftx != '모든 PC':
                            df_filtered = df_filtered[df_filtered[pc_col_name] == selected_pc_rftx].copy()
                    else:
                        st.warning("날짜 범위를 올바르게 선택해주세요.")
                        df_filtered = pd.DataFrame()

                    st.session_state.analysis_results['rftx'] = df_filtered
                    st.session_state.analysis_data['rftx'] = analyze_data(df_filtered, 'RfTxStamp_dt', pc_col_name)
                    st.session_state.analysis_time['rftx'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.session_state['last_analyzed_key'] = 'rftx'
                st.success("분석 완료! 결과가 저장되었습니다.")

            # 분석 결과가 존재하면 항상 표시
            if st.session_state.analysis_results['rftx'] is not None:
                display_analysis_result('rftx', 'RfTx_Process', 'RfTxStamp_dt',
                                        selected_jig=selected_pc_rftx if selected_pc_rftx != '모든 PC' else None)

            st.markdown("---")
            st.markdown("#### RfTx 데이터 조회")
            snumber_query_rftx = st.text_input("SNumber를 입력하세요 (RfTx)", key="snumber_search_bar_rftx")
            
            col_search_btn, col_view_btn = st.columns(2)
            with col_search_btn:
                if st.button("SNumber 검색 실행", key="snumber_search_btn_rftx"):
                    st.session_state.snumber_search['rftx']['show'] = True
                    if snumber_query_rftx:
                        with st.spinner("데이터베이스에서 SNumber 검색 중..."):
                            filtered_df = st.session_state.analysis_results['rftx'][
                                st.session_state.analysis_results['rftx']['SNumber'].fillna('').astype(str).str.contains(snumber_query_rftx, case=False, na=False)
                            ]
                        if not filtered_df.empty:
                            st.success(f"'{snumber_query_rftx}'에 대한 {len(filtered_df)}건의 검색 결과를 찾았습니다.")
                            st.session_state.snumber_search['rftx']['results'] = filtered_df
                        else:
                            st.warning(f"'{snumber_query_rftx}'에 대한 검색 결과가 없습니다.")
                            st.session_state.snumber_search['rftx']['results'] = pd.DataFrame()
                    else:
                        st.warning("SNumber를 입력해주세요.")
                        st.session_state.snumber_search['rftx']['results'] = pd.DataFrame()

            with col_view_btn:
                if st.button("원본 DB 조회", key="view_last_db_rftx"):
                    st.session_state.original_db_view['rftx']['show'] = True
                    if st.session_state.analysis_results['rftx'] is not None:
                        st.success(f"RfTx 탭의 원본 데이터를 조회합니다.")
                        st.session_state.original_db_view['rftx']['results'] = st.session_state.analysis_results['rftx'].copy()
                    else:
                        st.warning("먼저 RfTx 탭에서 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                        st.session_state.original_db_view['rftx']['results'] = pd.DataFrame()

            if st.session_state.snumber_search['rftx']['show'] and not st.session_state.snumber_search['rftx']['results'].empty:
                st.dataframe(st.session_state.snumber_search['rftx']['results'].reset_index(drop=True))

            if st.session_state.original_db_view['rftx']['show'] and not st.session_state.original_db_view['rftx']['results'].empty:
                st.dataframe(st.session_state.original_db_view['rftx']['results'].reset_index(drop=True))

        with tab4:
            st.header("파일 Semi (SemiAssy_Process)")

            pc_col_name = 'SemiAssyMaxBatVolt'
            unique_pc_semi = df_all_data[pc_col_name].dropna().unique()
            pc_options_semi = ['모든 PC'] + sorted(list(unique_pc_semi))
            selected_pc_semi = st.selectbox("PC (Jig) 선택", pc_options_semi, key="pc_select_semi")

            df_dates = df_all_data['SemiAssyStartTime_dt'].dt.date.dropna()
            min_date = df_dates.min() if not df_dates.empty else date.today()
            max_date = df_dates.max() if not df_dates.dropna().empty else date.today()
            selected_dates = st.date_input("날짜 범위 선택", value=(min_date, max_date), key="dates_semi")
            
            if st.button("분석 실행", key="analyze_semi"):
                with st.spinner("데이터 분석 및 저장 중..."):
                    if len(selected_dates) == 2:
                        start_date, end_date = selected_dates
                        df_filtered = df_all_data[
                            (df_all_data['SemiAssyStartTime_dt'].dt.date >= start_date) &
                            (df_all_data['SemiAssyStartTime_dt'].dt.date <= end_date)
                        ].copy()
                        if selected_pc_semi != '모든 PC':
                            df_filtered = df_filtered[df_filtered[pc_col_name] == selected_pc_semi].copy()
                    else:
                        st.warning("날짜 범위를 올바르게 선택해주세요.")
                        df_filtered = pd.DataFrame()

                    st.session_state.analysis_results['semi'] = df_filtered
                    st.session_state.analysis_data['semi'] = analyze_data(df_filtered, 'SemiAssyStartTime_dt', pc_col_name)
                    st.session_state.analysis_time['semi'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.session_state['last_analyzed_key'] = 'semi'
                st.success("분석 완료! 결과가 저장되었습니다.")

            # 분석 결과가 존재하면 항상 표시
            if st.session_state.analysis_results['semi'] is not None:
                display_analysis_result('semi', 'SemiAssy_Process', 'SemiAssyStartTime_dt',
                                        selected_jig=selected_pc_semi if selected_pc_semi != '모든 PC' else None)

            st.markdown("---")
            st.markdown("#### Semi 데이터 조회")
            snumber_query_semi = st.text_input("SNumber를 입력하세요 (Semi)", key="snumber_search_bar_semi")
            
            col_search_btn, col_view_btn = st.columns(2)
            with col_search_btn:
                if st.button("SNumber 검색 실행", key="snumber_search_btn_semi"):
                    st.session_state.snumber_search['semi']['show'] = True
                    if snumber_query_semi:
                        with st.spinner("데이터베이스에서 SNumber 검색 중..."):
                            filtered_df = st.session_state.analysis_results['semi'][
                                st.session_state.analysis_results['semi']['SNumber'].fillna('').astype(str).str.contains(snumber_query_semi, case=False, na=False)
                            ]
                        if not filtered_df.empty:
                            st.success(f"'{snumber_query_semi}'에 대한 {len(filtered_df)}건의 검색 결과를 찾았습니다.")
                            st.session_state.snumber_search['semi']['results'] = filtered_df
                        else:
                            st.warning(f"'{snumber_query_semi}'에 대한 검색 결과가 없습니다.")
                            st.session_state.snumber_search['semi']['results'] = pd.DataFrame()
                    else:
                        st.warning("SNumber를 입력해주세요.")
                        st.session_state.snumber_search['semi']['results'] = pd.DataFrame()

            with col_view_btn:
                if st.button("원본 DB 조회", key="view_last_db_semi"):
                    st.session_state.original_db_view['semi']['show'] = True
                    if st.session_state.analysis_results['semi'] is not None:
                        st.success(f"Semi 탭의 원본 데이터를 조회합니다.")
                        st.session_state.original_db_view['semi']['results'] = st.session_state.analysis_results['semi'].copy()
                    else:
                        st.warning("먼저 Semi 탭에서 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                        st.session_state.original_db_view['semi']['results'] = pd.DataFrame()

            if st.session_state.snumber_search['semi']['show'] and not st.session_state.snumber_search['semi']['results'].empty:
                st.dataframe(st.session_state.snumber_search['semi']['results'].reset_index(drop=True))

            if st.session_state.original_db_view['semi']['show'] and not st.session_state.original_db_view['semi']['results'].empty:
                st.dataframe(st.session_state.original_db_view['semi']['results'].reset_index(drop=True))

        with tab5:
            st.header("파일 Func (Func_Process)")

            pc_col_name = 'BatadcPC'
            unique_pc_func = df_all_data[pc_col_name].dropna().unique()
            pc_options_func = ['모든 PC'] + sorted(list(unique_pc_func))
            selected_pc_func = st.selectbox("PC (Jig) 선택", pc_options_func, key="pc_select_func")
            
            df_dates = df_all_data['BatadcStamp_dt'].dt.date.dropna()
            min_date = df_dates.min() if not df_dates.empty else date.today()
            max_date = df_dates.max() if not df_dates.dropna().empty else date.today()
            selected_dates = st.date_input("날짜 범위 선택", value=(min_date, max_date), key="dates_func")
            
            if st.button("분석 실행", key="analyze_func"):
                with st.spinner("데이터 분석 및 저장 중..."):
                    if len(selected_dates) == 2:
                        start_date, end_date = selected_dates
                        df_filtered = df_all_data[
                            (df_all_data['BatadcStamp_dt'].dt.date >= start_date) &
                            (df_all_data['BatadcStamp_dt'].dt.date <= end_date)
                        ].copy()
                        if selected_pc_func != '모든 PC':
                            df_filtered = df_filtered[df_filtered[pc_col_name] == selected_pc_func].copy()
                    else:
                        st.warning("날짜 범위를 올바르게 선택해주세요.")
                        df_filtered = pd.DataFrame()
                    
                    st.session_state.analysis_results['func'] = df_filtered
                    st.session_state.analysis_data['func'] = analyze_data(df_filtered, 'BatadcStamp_dt', pc_col_name)
                    st.session_state.analysis_time['func'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.session_state['last_analyzed_key'] = 'func'
                st.success("분석 완료! 결과가 저장되었습니다.")

            # 분석 결과가 존재하면 항상 표시
            if st.session_state.analysis_results['func'] is not None:
                display_analysis_result('func', 'Func_Process', 'BatadcStamp_dt',
                                        selected_jig=selected_pc_func if selected_pc_func != '모든 PC' else None)
            
            st.markdown("---")
            st.markdown("#### Func 데이터 조회")
            snumber_query_func = st.text_input("SNumber를 입력하세요 (Func)", key="snumber_search_bar_func")
            
            col_search_btn, col_view_btn = st.columns(2)
            with col_search_btn:
                if st.button("SNumber 검색 실행", key="snumber_search_btn_func"):
                    st.session_state.snumber_search['func']['show'] = True
                    if snumber_query_func:
                        with st.spinner("데이터베이스에서 SNumber 검색 중..."):
                            filtered_df = st.session_state.analysis_results['func'][
                                st.session_state.analysis_results['func']['SNumber'].fillna('').astype(str).str.contains(snumber_query_func, case=False, na=False)
                            ]
                        if not filtered_df.empty:
                            st.success(f"'{snumber_query_func}'에 대한 {len(filtered_df)}건의 검색 결과를 찾았습니다.")
                            st.session_state.snumber_search['func']['results'] = filtered_df
                        else:
                            st.warning(f"'{snumber_query_func}'에 대한 검색 결과가 없습니다.")
                            st.session_state.snumber_search['func']['results'] = pd.DataFrame()
                    else:
                        st.warning("SNumber를 입력해주세요.")
                        st.session_state.snumber_search['func']['results'] = pd.DataFrame()

            with col_view_btn:
                if st.button("원본 DB 조회", key="view_last_db_func"):
                    st.session_state.original_db_view['func']['show'] = True
                    if st.session_state.analysis_results['func'] is not None:
                        st.success(f"Func 탭의 원본 데이터를 조회합니다.")
                        st.session_state.original_db_view['func']['results'] = st.session_state.analysis_results['func'].copy()
                    else:
                        st.warning("먼저 Func 탭에서 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                        st.session_state.original_db_view['func']['results'] = pd.DataFrame()
            
            if st.session_state.snumber_search['func']['show'] and not st.session_state.snumber_search['func']['results'].empty:
                st.dataframe(st.session_state.snumber_search['func']['results'].reset_index(drop=True))
            
            if st.session_state.original_db_view['func']['show'] and not st.session_state.original_db_view['func']['results'].empty:
                st.dataframe(st.session_state.original_db_view['func']['results'].reset_index(drop=True))
    
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
