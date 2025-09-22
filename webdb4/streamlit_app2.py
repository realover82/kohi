import streamlit as st
import pandas as pd
from datetime import datetime, date
import warnings
import io
import re

warnings.filterwarnings('ignore')

# csv 업로드 및 데이터 처리 유틸리티 함수를 담은 db_utils 모듈 임포트
from db_utils import process_uploaded_csv

# analyze_data 함수: CSV 파일에서 읽어온 DataFrame을 분석합니다.
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
    df_copy = df.copy()
    
    # 다양한 Pass 컬럼에 대해 PassStatusNorm 생성
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
        # Pass 컬럼이 없는 경우, 분석을 계속할 수 없으므로 빈 결과를 반환
        st.warning("Pass 상태를 나타내는 컬럼이 없습니다. 다음 컬럼 중 하나가 필요합니다: PcbPass, FwPass, RfTxPass, SemiAssyPass, BatadcPass")
        return {}, [], jig_col_name

    summary_data = {}
    
    # 지그(PC) 컬럼에 데이터가 없는 경우 '전체'를 대체 컬럼으로 사용
    used_jig_col_name = jig_col_name
    if jig_col_name not in df_copy.columns or df_copy[jig_col_name].isnull().all() or df_copy[jig_col_name].nunique() < 2:
        used_jig_col_name = '__total_group__'
        df_copy[used_jig_col_name] = '전체'

    # 지그(PC) 컬럼이 존재하고 데이터가 있는 경우에만 그룹 분석 실행
    if used_jig_col_name in df_copy.columns and not df_copy[used_jig_col_name].isnull().all():
        if 'SNumber' in df_copy.columns and date_col_name in df_copy.columns and not df_copy[date_col_name].dt.date.dropna().empty:
            for jig, group in df_copy.groupby(used_jig_col_name):
                # 날짜 열이 datetime 타입인지 확인하고, 아니면 변환
                if not pd.api.types.is_datetime64_any_dtype(group[date_col_name]):
                    group.loc[:, date_col_name] = pd.to_datetime(group[date_col_name], errors='coerce')
                
                # 유효한 날짜 데이터만 필터링
                group = group.dropna(subset=[date_col_name]).copy()

                if group.empty:
                    continue

                for d, day_group in group.groupby(group[date_col_name].dt.date):
                    if pd.isna(d): continue
                    date_iso = pd.to_datetime(d).strftime("%Y-%m-%d")
                    
                    # SNumber가 유효한지 확인하고, 유효한 SNumber만 필터링
                    day_group = day_group[day_group['SNumber'].notna()]
                    if day_group.empty:
                        continue

                    # 'PassStatusNorm'이 존재하는지 확인
                    if 'PassStatusNorm' not in day_group.columns:
                        continue
                        
                    # PassStatusNorm의 O, X 데이터를 활용하여 pass, fail 집계
                    pass_sns_series = day_group.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
                    pass_sns = pass_sns_series[pass_sns_series].index.tolist()

                    total_test_count = len(day_group['SNumber'].unique())
                    pass_count = len(pass_sns)
                    
                    false_defect_count = len(day_group[(day_group['PassStatusNorm'] == 'X') & (day_group['SNumber'].isin(pass_sns))]['SNumber'].unique())
                    true_defect_count = len(day_group[(day_group['PassStatusNorm'] == 'X') & (~day_group['SNumber'].isin(pass_sns))]['SNumber'].unique())
                    
                    # 요청에 따라 'fail' 값 수정
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
    if st.session_state.analysis_results[analysis_key] is None:
        st.warning("분석할 파일이 업로드되지 않았습니다.")
        return
    if st.session_state.analysis_results[analysis_key].empty:
        st.warning("분석 데이터가 비어 있습니다.")
        return

    # Check if analysis_data is available for the key
    if st.session_state.analysis_data[analysis_key] is None:
        st.warning("분석 데이터가 준비되지 않았습니다. '분석 실행' 버튼을 눌러주세요.")
        return

    # Unpack the tuple safely
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
        
        # used_jig_col이 '__total_group__'인 경우 필터링을 건너뜁니다.
        if used_jig_col == '__total_group__':
            jig_filtered_df = df_filtered.copy()
        elif used_jig_col not in df_filtered.columns:
            st.warning(f"데이터프레임에 '{used_jig_col}' 컬럼이 없어 상세 내역을 표시할 수 없습니다.")
            continue
        else:
            # 현재 지그에 해당하는 데이터만 필터링
            jig_filtered_df = df_filtered[df_filtered[used_jig_col] == jig].copy()
        
        # SNumber가 유효한지 확인
        if 'SNumber' not in jig_filtered_df.columns:
            st.warning("'SNumber' 컬럼이 없어 상세 내역을 표시할 수 없습니다.")
            continue
        jig_filtered_df = jig_filtered_df[jig_filtered_df['SNumber'].notna()]
        
        # PassStatusNorm이 존재하는지 확인
        # analyze_data에서 생성된 PassStatusNorm이 원본 analysis_results에는 없으므로, 여기서 다시 생성
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

        # PASS 상세 내역 (SNumber 기준)
        pass_sns = jig_filtered_df.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist())
        pass_sns = pass_sns[pass_sns].index.tolist()
        with st.expander(f"PASS ({len(pass_sns)}건)", expanded=False):
            st.text("\n".join(pass_sns))
        
        # 가성불량 (False Defect) 상세 내역 (SNumber 기준)
        false_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        with st.expander(f"가성불량 ({len(false_defect_sns)}건)", expanded=False):
            st.text("\n".join(false_defect_sns))
            
        # 진성불량 (True Defect) 상세 내역 (SNumber 기준)
        true_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (~jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        with st.expander(f"진성불량 ({len(true_defect_sns)}건)", expanded=False):
            st.text("\n".join(true_defect_sns))

        # FAIL 상세 내역 (SNumber 기준)
        all_snumbers = jig_filtered_df['SNumber'].unique().tolist()
        all_fail_sns = list(set(all_snumbers) - set(pass_sns))
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
    # 가성불량과 진성불량 필드를 추가
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


def main():
    st.set_page_config(layout="wide")
    st.title("리모컨 생산 데이터 분석 툴")
    st.markdown("---")

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
    if 'date_col_mapping' not in st.session_state:
        st.session_state['date_col_mapping'] = {
            'pcb': 'PcbStartTime',
            'fw': 'FwStamp',
            'rftx': 'RfTxStamp',
            'semi': 'SemiAssyStartTime',
            'func': 'BatadcStamp',
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
    if 'selected_cols' not in st.session_state:
        st.session_state.selected_cols = {
            'pcb': [], 'fw': [], 'rftx': [], 'semi': [], 'func': []
        }

    # --- 탭별 분석 기능 ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["파일 PCB 분석", "파일 Fw 분석", "파일 RfTx 분석", "파일 Semi 분석", "파일 Func 분석"])
    
    tabs_config = {
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

    # 탭 순회 및 UI 렌더링
    for key, tab_content in zip(['pcb', 'fw', 'rftx', 'semi', 'func'], [tab1, tab2, tab3, tab4, tab5]):
        with tab_content:
            st.header(tabs_config[key]['header'])
            
            uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"], key=f"uploader_{key}")
            
            if uploaded_file:
                df_all_data = process_uploaded_csv(uploaded_file, key)
                
                if df_all_data is not None and not df_all_data.empty:
                    st.success("파일이 성공적으로 로드되었습니다.")
                    st.session_state.analysis_results[key] = df_all_data.copy()
                    
                    # 모든 컬럼 목록을 세션 상태에 저장
                    st.session_state.selected_cols[key] = df_all_data.columns.tolist()
                else:
                    st.warning("유효한 데이터를 불러오지 못했습니다. 올바른 형식의 파일인지 확인해주세요.")
                    st.session_state.analysis_results[key] = None
            
            if st.session_state.analysis_results[key] is not None:
                df_to_analyze = st.session_state.analysis_results[key].copy()
                
                # PC (Jig) 선택 기능 추가
                jig_col_name = tabs_config[key]['jig_col']
                date_col_name = tabs_config[key]['date_col']
                
                # 날짜 컬럼을 datetime으로 변환 (파일 로드 시 처리될 수 있으나 안전을 위해 다시 확인)
                df_to_analyze[f"{date_col_name}_dt"] = pd.to_datetime(df_to_analyze[date_col_name], errors='coerce')
                
                # jig_col_name이 데이터프레임에 있는지 확인
                if jig_col_name in df_to_analyze.columns:
                    unique_pc = df_to_analyze[jig_col_name].dropna().unique()
                    pc_options = ['모든 PC'] + sorted(list(unique_pc))
                    selected_pc = st.selectbox("PC (Jig) 선택", pc_options, key=f"pc_select_{key}")
                else:
                    st.warning(f"'{jig_col_name}' 컬럼이 없어 PC 선택 기능을 사용할 수 없습니다. '모든 PC'로 설정됩니다.")
                    selected_pc = '모든 PC'

                df_dates = df_to_analyze[f"{date_col_name}_dt"].dt.date.dropna()
                min_date = df_dates.min() if not df_dates.empty else date.today()
                max_date = df_dates.max() if not df_dates.dropna().empty else date.today()
                selected_dates = st.date_input("날짜 범위 선택", value=(min_date, max_date), key=f"dates_{key}")
                
                if st.button("분석 실행", key=f"analyze_{key}"):
                    with st.spinner("데이터 분석 및 저장 중..."):
                        if len(selected_dates) == 2:
                            start_date, end_date = selected_dates
                            df_filtered = df_to_analyze[
                                (df_to_analyze[f"{date_col_name}_dt"].dt.date >= start_date) &
                                (df_to_analyze[f"{date_col_name}_dt"].dt.date <= end_date)
                            ].copy()
                            
                            if selected_pc != '모든 PC':
                                df_filtered = df_filtered[df_filtered[jig_col_name] == selected_pc].copy()
                        else:
                            st.warning("날짜 범위를 올바르게 선택해주세요.")
                            df_filtered = pd.DataFrame()
                        
                        st.session_state.analysis_results[key] = df_filtered
                        st.session_state.analysis_data[key] = analyze_data(df_filtered, f"{date_col_name}_dt", jig_col_name)
                        st.session_state.analysis_time[key] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state['last_analyzed_key'] = key
                    st.success("분석 완료! 결과가 저장되었습니다.")

                # 분석 결과가 존재하면 항상 표시
                if st.session_state.analysis_results[key] is not None:
                    if st.session_state.analysis_results[key].empty:
                        st.warning("선택한 조건에 맞는 데이터가 없습니다.")
                    else:
                        display_analysis_result(key, tabs_config[key]['header'], f"{date_col_name}_dt",
                                                selected_jig=selected_pc if selected_pc != '모든 PC' else None)
                
                st.markdown("---")
                st.markdown(f"#### {tabs_config[key]['header'].split()[1]} 데이터 조회")
                
                # 필드 선택 기능 추가
                all_cols = st.session_state.analysis_results[key].columns.tolist()
                selected_display_cols = st.multiselect(
                    "표시할 필드를 선택하세요",
                    options=all_cols,
                    default=[col for col in ['SNumber'] if col in all_cols],
                    key=f"col_select_{key}"
                )
                
                snumber_query = st.text_input("SNumber를 입력하세요", key=f"snumber_search_bar_{key}")
                
                col_search_btn, col_view_btn = st.columns(2)
                with col_search_btn:
                    if st.button("SNumber 검색 실행", key=f"snumber_search_btn_{key}"):
                        st.session_state.snumber_search[key]['show'] = True
                        if snumber_query:
                            with st.spinner("데이터에서 SNumber 검색 중..."):
                                df_source = st.session_state.analysis_results.get(key)
                                if df_source is not None and not df_source.empty:
                                    filtered_df = df_source[
                                        df_source['SNumber'].fillna('').astype(str).str.contains(snumber_query, case=False, na=False)
                                    ]
                                    if not filtered_df.empty:
                                        st.success(f"'{snumber_query}'에 대한 {len(filtered_df)}건의 검색 결과를 찾았습니다.")
                                        st.session_state.snumber_search[key]['results'] = filtered_df.copy()
                                    else:
                                        st.warning(f"'{snumber_query}'에 대한 검색 결과가 없습니다.")
                                        st.session_state.snumber_search[key]['results'] = pd.DataFrame()
                                else:
                                    st.warning("먼저 CSV 파일을 업로드하고 분석을 실행해주세요.")
                                    st.session_state.snumber_search[key]['results'] = pd.DataFrame()
                        else:
                            st.warning("SNumber를 입력해주세요.")
                            st.session_state.snumber_search[key]['results'] = pd.DataFrame()

                with col_view_btn:
                    if st.button("업로드된 파일 원본 조회", key=f"view_last_db_{key}"):
                        st.session_state.original_db_view[key]['show'] = True
                        if st.session_state.analysis_results[key] is not None:
                            st.success(f"{tabs_config[key]['header'].split()[1]} 탭의 원본 데이터를 조회합니다.")
                            st.session_state.original_db_view[key]['results'] = st.session_state.analysis_results[key].copy()
                        else:
                            st.warning("먼저 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                            st.session_state.original_db_view[key]['results'] = pd.DataFrame()

                if st.session_state.snumber_search[key]['show'] and not st.session_state.snumber_search[key]['results'].empty:
                    st.dataframe(st.session_state.snumber_search[key]['results'][selected_display_cols].reset_index(drop=True))

                if st.session_state.original_db_view[key]['show'] and not st.session_state.original_db_view[key]['results'].empty:
                    st.dataframe(st.session_state.original_db_view[key]['results'][selected_display_cols].reset_index(drop=True))

if __name__ == "__main__":
    main()
