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
    df_raw = st.session_state.analysis_results[analysis_key]

    st.markdown(f"### '{file_name}' 분석 리포트")

    # --- 기본 필터링 (Jig, 날짜 범위) ---
    st.subheader("기본 필터링")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        jig_list = sorted(df_raw[jig_col_name].dropna().unique().tolist()) if jig_col_name in df_raw.columns else []
        selected_jig = st.selectbox("PC(Jig) 선택", ["전체"] + jig_list, key=f"select_{analysis_key}")
    
    if not all_dates:
        st.warning("분석할 데이터가 없습니다.")
        return
        
    min_date, max_date = min(all_dates), max(all_dates)
    with filter_col2:
        start_date = st.date_input("시작 날짜", min_value=min_date, max_value=max_date, value=min_date, key=f"start_date_{analysis_key}")
    with filter_col3:
        end_date = st.date_input("종료 날짜", min_value=min_date, max_value=max_date, value=max_date, key=f"end_date_{analysis_key}")

    filtered_dates = [d for d in all_dates if start_date <= d <= end_date]
    if not filtered_dates:
        st.warning("선택된 날짜 범위에 해당하는 데이터가 없습니다.")
        return

    st.write(f"**분석 시간**: {st.session_state.analysis_time[analysis_key]}")
    st.markdown("---")

    # --- 요약 대시보드 (테이블, 그래프) ---
    st.subheader("요약 대시보드 (조회 기간 합계)")
    
    jigs_to_display = jig_list if selected_jig == "전체" else [selected_jig]
    
    # 범위 내 데이터 집계
    total_summary = {'total_test': 0, 'pass': 0, 'false_defect': 0, 'true_defect': 0, 'fail': 0}
    daily_data = []

    for jig in jigs_to_display:
        for date_obj in filtered_dates:
            date_iso = date_obj.strftime('%Y-%m-%d')
            data_point = summary_data.get(jig, {}).get(date_iso)
            if data_point:
                for key in total_summary:
                    total_summary[key] += data_point.get(key, 0)
                daily_data.append({
                    'date': date_obj,
                    'PASS': data_point.get('pass', 0),
                    '가성불량': data_point.get('false_defect', 0),
                    '진성불량': data_point.get('true_defect', 0),
                })

    summary_df = pd.DataFrame({
        '지표': ['총 테스트 수', 'PASS', '가성불량', '진성불량', 'FAIL'],
        '합계': [total_summary['total_test'], total_summary['pass'], total_summary['false_defect'], total_summary['true_defect'], total_summary['fail']]
    }).set_index('지표')
    
    st.table(summary_df)

    st.subheader("일별 추이 그래프")
    if daily_data:
        chart_df = pd.DataFrame(daily_data).set_index('date')
        st.bar_chart(chart_df)
    else:
        st.info("그래프를 표시할 데이터가 없습니다.")

    st.markdown("---")

    # --- 상세 내역 (날짜별 펼치기) ---
    st.subheader("상세 내역 (일별)")
    for date_obj in filtered_dates:
        st.markdown(f"**{date_obj.strftime('%Y-%m-%d')}**")
        date_iso = date_obj.strftime('%Y-%m-%d')
        
        for jig in jigs_to_display:
            data_point = summary_data.get(jig, {}).get(date_iso)
            if not data_point:
                continue

            st.markdown(f"**PC(Jig): {jig}**")
            categories = ['pass', 'false_defect', 'true_defect', 'fail']
            labels = ['PASS', '가성불량', '진성불량', 'FAIL']
            
            for cat, label in zip(categories, labels):
                sns_list = data_point.get(f'{cat}_sns', [])
                count = data_point.get(cat, 0)
                with st.expander(f"{label} - {count}건", expanded=False):
                    if sns_list:
                        st.text("\n".join(sns_list))
                    else:
                        st.info("해당 내역이 없습니다.")
        st.markdown("---")


    # --- DB 원본 확인 및 상세 검색 기능 ---
    st.subheader("DB 원본 상세 검색")
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col1:
        snumber_query = st.text_input("SNumber 검색", key=f"snumber_search_{analysis_key}")
    with search_col2:
        all_columns = df_raw.columns.tolist()
        selected_columns = st.multiselect("표시할 필드(열) 선택", all_columns, key=f"col_select_{analysis_key}")
    with search_col3:
        st.write("") # 여백
        st.write("") # 여백
        apply_button = st.button("필터 적용", key=f"apply_filter_{analysis_key}")

    # 필터 적용 버튼 클릭 시 상태 저장
    filter_state_key = f'applied_filters_{analysis_key}'
    if apply_button:
        st.session_state[filter_state_key] = {
            'snumber': snumber_query,
            'columns': selected_columns
        }
    
    # 세션 상태에 저장된 필터 불러오기
    applied_filters = st.session_state.get(filter_state_key, {'snumber': '', 'columns': []})

    with st.expander("DB 원본 확인"):
        df_display = df_raw.copy()
        
        # SNumber 필터링 (적용된 필터 기준)
        if applied_filters['snumber']:
            query = applied_filters['snumber']
            if 'SNumber' in df_display.columns and pd.api.types.is_string_dtype(df_display['SNumber']):
                df_display = df_display[df_display['SNumber'].str.contains(query, na=False, case=False)]
            else:
                 try:
                     df_display = df_display[df_display.apply(lambda row: query.lower() in str(row.values).lower(), axis=1)]
                 except Exception:
                     st.warning("SNumber 검색을 지원하지 않는 데이터 형식입니다.")

        # 필드(열) 선택 필터링 (적용된 필터 기준)
        if applied_filters['columns']:
            df_display = df_display[applied_filters['columns']]
        
        st.dataframe(df_display)


# ==============================
# 메인 실행 함수 (기존과 동일)
# ==============================
def main():
    st.set_page_config(layout="wide")
    st.title("리모컨 생산 데이터 분석 툴")
    st.markdown("---")

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