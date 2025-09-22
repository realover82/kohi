import streamlit as st
import pandas as pd

def display_data_view_controls(key, header, date_col_name):
    st.markdown("---")
    st.markdown(f"#### {header.split()[1]} 데이터 조회")
    
    if st.session_state.analysis_results.get(key) is None:
        st.warning("먼저 CSV 파일을 업로드하고 분석을 실행해주세요.")
        return
    
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
                st.success(f"{header.split()[1]} 탭의 원본 데이터를 조회합니다.")
                st.session_state.original_db_view[key]['results'] = st.session_state.analysis_results[key].copy()
            else:
                st.warning("먼저 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                st.session_state.original_db_view[key]['results'] = pd.DataFrame()

    if st.session_state.snumber_search[key]['show'] and not st.session_state.snumber_search[key]['results'].empty:
        st.dataframe(st.session_state.snumber_search[key]['results'][selected_display_cols].reset_index(drop=True))

    if st.session_state.original_db_view[key]['show'] and not st.session_state.original_db_view[key]['results'].empty:
        st.dataframe(st.session_state.original_db_view[key]['results'][selected_display_cols].reset_index(drop=True))
