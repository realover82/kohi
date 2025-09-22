import streamlit as st
import pandas as pd
from datetime import datetime

# 각 CSV 분석 모듈 불러오기
from csv2 import read_csv_with_dynamic_header, analyze_data
from csv_Fw import read_csv_with_dynamic_header_for_Fw, analyze_Fw_data
from csv_RfTx import read_csv_with_dynamic_header_for_RfTx, analyze_RfTx_data
from csv_Semi import read_csv_with_dynamic_header_for_Semi, analyze_Semi_data
from csv_Batadc import read_csv_with_dynamic_header_for_Batadc, analyze_Batadc_data

def display_analysis_result(analysis_key, file_name, jig_col_name):
    """ session_state에 저장된 분석 결과를 Streamlit에 표시하는 함수 """
    if st.session_state.analysis_results[analysis_key] is None:
        st.error("데이터 로드에 실패했습니다. 파일 형식을 확인해주세요.")
        return

    summary_data, all_dates = st.session_state.analysis_data[analysis_key]
    df_raw = st.session_state.analysis_results[analysis_key] # 원본 데이터

    st.markdown(f"### '{file_name}' 분석 리포트")

    # --- 기본 필터링 (Jig, 날짜) ---
    st.subheader("기본 필터링")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        jig_list = sorted(df_raw[jig_col_name].dropna().unique().tolist()) if jig_col_name in df_raw.columns else []
        selected_jig = st.selectbox("PC(Jig) 선택", ["전체"] + jig_list, key=f"select_{analysis_key}")
    
    with filter_col2:
        if not all_dates:
            st.warning("분석할 데이터가 없습니다.")
            return
        min_date, max_date = min(all_dates), max(all_dates)
        selected_date = st.date_input(
            "날짜 선택",
            min_value=min_date,
            max_value=max_date,
            value=max_date,
            key=f"date_select_{analysis_key}"
        )

    filtered_dates = [d for d in all_dates if d == selected_date]
    if not filtered_dates:
        st.warning("선택된 날짜에 해당하는 데이터가 없습니다.")
        return

    st.write(f"**분석 시간**: {st.session_state.analysis_time[analysis_key]}")
    st.markdown("---")

    # --- 요약 대시보드 (테이블, 그래프) ---
    st.subheader("요약 대시보드")
    display_mode_cols = st.columns(3)
    with display_mode_cols[0]:
        if st.button("테이블", key=f"mode_table_{analysis_key}"):
            st.session_state[f'display_mode_{analysis_key}'] = 'table'
    with display_mode_cols[1]:
        if st.button("꺾은선 그래프", key=f"mode_line_{analysis_key}"):
            st.session_state[f'display_mode_{analysis_key}'] = 'line'
    with display_mode_cols[2]:
        if st.button("막대 그래프", key=f"mode_bar_{analysis_key}"):
            st.session_state[f'display_mode_{analysis_key}'] = 'bar'

    if f'display_mode_{analysis_key}' not in st.session_state:
        st.session_state[f'display_mode_{analysis_key}'] = 'table'

    jigs_to_display = jig_list if selected_jig == "전체" else [selected_jig]

    for jig in jigs_to_display:
        st.markdown(f"**PC(Jig): {jig}**")
        date_iso = selected_date.strftime('%Y-%m-%d')
        data_point = summary_data.get(jig, {}).get(date_iso)

        if not data_point:
            st.info(f"선택된 날짜({date_iso})에 '{jig}'에 대한 데이터가 없습니다.")
            continue

        if st.session_state[f'display_mode_{analysis_key}'] == 'table':
            report_data = {
                '지표': ['총 테스트 수', 'PASS', '가성불량', '진성불량', 'FAIL'],
                '값': [
                    data_point.get('total_test', 'N/A'),
                    data_point.get('pass', 'N/A'),
                    data_point.get('false_defect', 'N/A'),
                    data_point.get('true_defect', 'N/A'),
                    data_point.get('fail', 'N/A')
                ]
            }
            report_df = pd.DataFrame(report_data).set_index('지표')
            st.table(report_df)
        else:
            chart_df = pd.DataFrame({
                'PASS': [data_point.get('pass', 0)],
                '가성불량': [data_point.get('false_defect', 0)],
                '진성불량': [data_point.get('true_defect', 0)],
            }, index=[date_iso])
            
            if st.session_state[f'display_mode_{analysis_key}'] == 'line':
                st.line_chart(chart_df)
            else:
                st.bar_chart(chart_df)

        # --- 상세 내역 (펼치기/접기 기능) ---
        st.markdown("**상세 내역**")
        detail_buttons = st.columns(4)
        categories = ['pass', 'false_defect', 'true_defect', 'fail']
        labels = ['PASS 상세', '가성불량 상세', '진성불량 상세', 'FAIL 상세']

        for col, cat, label in zip(detail_buttons, categories, labels):
            with col:
                if st.button(label, key=f"{cat}_btn_{analysis_key}_{jig}"):
                    state_key = f'show_details_{analysis_key}_{jig}'
                    if state_key not in st.session_state:
                        st.session_state[state_key] = {c: False for c in categories}
                    st.session_state[state_key][cat] = not st.session_state[state_key].get(cat, False)
        
        state_key = f'show_details_{analysis_key}_{jig}'
        if state_key in st.session_state:
            for cat in categories:
                if st.session_state[state_key].get(cat, False):
                    sns_list = data_point.get(f'{cat}_sns', [])
                    count = data_point.get(cat, 0)
                    # 수정: expanded=False로 변경하여 기본적으로 접혀있도록 설정
                    with st.expander(f"{labels[categories.index(cat)]} ({selected_date.strftime('%y%m%d')}) - {count}건", expanded=False):
                        if sns_list:
                            st.text("\n".join(sns_list))
                        else:
                            st.info("해당 내역이 없습니다.")
        st.markdown("---")

    # --- DB 원본 확인 및 상세 검색 기능 ---
    st.subheader("DB 원본 상세 검색")
    search_col1, search_col2 = st.columns([1, 3])
    with search_col1:
        snumber_query = st.text_input("SNumber 검색", key=f"snumber_search_{analysis_key}")
    with search_col2:
        all_columns = df_raw.columns.tolist()
        selected_columns = st.multiselect("표시할 필드(열) 선택", all_columns, key=f"col_select_{analysis_key}")

    with st.expander("DB 원본 확인"):
        df_display = df_raw.copy()
        
        # SNumber 필터링
        if snumber_query:
            # 'SNumber' 컬럼이 문자열 타입인지 확인 후 필터링
            if 'SNumber' in df_display.columns and pd.api.types.is_string_dtype(df_display['SNumber']):
                df_display = df_display[df_display['SNumber'].str.contains(snumber_query, na=False, case=False)]
            else:
                # SNumber 컬럼이 없거나 문자열이 아닐 경우, 다른 컬럼에서 검색 시도
                 try:
                     df_display = df_display[df_display.apply(lambda row: snumber_query.lower() in str(row.values).lower(), axis=1)]
                 except Exception:
                     st.warning("SNumber 검색을 지원하지 않는 데이터 형식입니다.")

        # 필드(열) 선택 필터링
        if selected_columns:
            df_display = df_display[selected_columns]
        
        st.dataframe(df_display)

# ==============================
# 메인 실행 함수 (기존과 동일)
# ==============================
def main():
    st.set_page_config(layout="wide")
    st.title("리모컨 생산 데이터 분석 툴")
    st.markdown("---")

    # session_state 초기화
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {k: None for k in ['pcb', 'fw', 'rftx', 'semi', 'func']}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {k: None for k in ['pcb', 'fw', 'rftx', 'semi', 'func']}
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {k: None for k in ['pcb', 'fw', 'rftx', 'semi', 'func']}
    if 'analysis_time' not in st.session_state:
        st.session_state.analysis_time = {k: None for k in ['pcb', 'fw', 'rftx', 'semi', 'func']}

    tabs = st.tabs(["파일 PCB 분석", "파일 Fw 분석", "파일 RfTx 분석", "파일 Semi 분석", "파일 Func 분석"])
    tab_map = {
        'pcb': {'tab': tabs[0], 'reader': read_csv_with_dynamic_header, 'analyzer': analyze_data, 'jig_col': 'PcbMaxIrPwr'},
        'fw': {'tab': tabs[1], 'reader': read_csv_with_dynamic_header_for_Fw, 'analyzer': analyze_Fw_data, 'jig_col': 'FwPC'},
        'rftx': {'tab': tabs[2], 'reader': read_csv_with_dynamic_header_for_RfTx, 'analyzer': analyze_RfTx_data, 'jig_col': 'RfTxPC'},
        'semi': {'tab': tabs[3], 'reader': read_csv_with_dynamic_header_for_Semi, 'analyzer': analyze_Semi_data, 'jig_col': 'SemiAssyMaxSolarVolt'},
        'func': {'tab': tabs[4], 'reader': read_csv_with_dynamic_header_for_Batadc, 'analyzer': analyze_Batadc_data, 'jig_col': 'BatadcPC'}
    }

    for key, props in tab_map.items():
        with props['tab']:
            st.header(f"{key.upper()} 데이터 분석")
            st.session_state.uploaded_files[key] = st.file_uploader(f"{key.upper()} 파일을 선택하세요", type=["csv"], key=f"uploader_{key}")
            
            if st.session_state.uploaded_files[key]:
                if st.button(f"{key.upper()} 분석 실행", key=f"analyze_{key}"):
                    try:
                        df = props['reader'](st.session_state.uploaded_files[key])
                        if df is not None:
                            with st.spinner("데이터 분석 및 저장 중..."):
                                st.session_state.analysis_results[key] = df
                                st.session_state.analysis_data[key] = props['analyzer'](df)
                                st.session_state.analysis_time[key] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            st.success("분석 완료! 결과가 저장되었습니다.")
                        else:
                            st.error(f"{key.upper()} 데이터 파일을 읽을 수 없습니다. 파일 형식을 확인해주세요.")
                    except Exception as e:
                        st.error(f"분석 중 오류 발생: {e}")

                if st.session_state.analysis_results[key] is not None:
                    display_analysis_result(key, st.session_state.uploaded_files[key].name, props['jig_col'])

if __name__ == "__main__":
    main()