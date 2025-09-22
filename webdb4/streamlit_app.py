import streamlit as st
import pandas as pd
from datetime import datetime, date

from util1 import get_jig_and_date_inputs, create_tabs_config
from util2 import analyze_data, display_analysis_result
from util3 import display_data_view_controls

def main():
    st.set_page_config(layout="wide")
    st.title("리모컨 생산 데이터 분석 툴")
    st.markdown("---")

    # Initialize session state variables
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
    if 'selected_cols' not in st.session_state:
        st.session_state.selected_cols = {
            'pcb': [], 'fw': [], 'rftx': [], 'semi': [], 'func': []
        }
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
    if 'show_line_chart' not in st.session_state:
        st.session_state.show_line_chart = {}
    if 'show_bar_chart' not in st.session_state:
        st.session_state.show_bar_chart = {}
        
    tabs_config = create_tabs_config()
    tab1, tab2, tab3, tab4, tab5 = st.tabs([f"파일 {key.upper()} 분석" for key in tabs_config.keys()])
    
    tabs_dict = {
        'pcb': tab1, 'fw': tab2, 'rftx': tab3, 'semi': tab4, 'func': tab5
    }

    for key, tab_content in tabs_dict.items():
        with tab_content:
            st.header(tabs_config[key]['header'])
            
            uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"], key=f"uploader_{key}")
            
            if uploaded_file:
                df_all_data = get_jig_and_date_inputs(uploaded_file, key)
                
                if df_all_data is not None and not df_all_data.empty:
                    st.success("파일이 성공적으로 로드되었습니다.")
                    st.session_state.analysis_results[key] = df_all_data.copy()
                    st.session_state.selected_cols[key] = df_all_data.columns.tolist()
                else:
                    st.warning("유효한 데이터를 불러오지 못했습니다. 올바른 형식의 파일인지 확인해주세요.")
                    st.session_state.analysis_results[key] = None

            if st.session_state.analysis_results[key] is not None:
                df_to_analyze = st.session_state.analysis_results[key].copy()
                jig_col_name = tabs_config[key]['jig_col']
                date_col_name = tabs_config[key]['date_col']
                
                if f"{date_col_name}_dt" not in df_to_analyze.columns:
                    df_to_analyze[f"{date_col_name}_dt"] = pd.to_datetime(df_to_analyze[date_col_name], errors='coerce')
                
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
                    st.success("분석 완료! 결과가 저장되었습니다.")

                # Display analysis results
                display_analysis_result(key, tabs_config[key]['header'], f"{date_col_name}_dt",
                                        selected_jig=selected_pc if selected_pc != '모든 PC' else None)
                
                # Display data view controls
                display_data_view_controls(key, tabs_config[key]['header'], tabs_config[key]['date_col'])

if __name__ == "__main__":
    main()
