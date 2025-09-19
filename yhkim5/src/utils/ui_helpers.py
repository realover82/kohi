import streamlit as st
import pandas as pd
from datetime import datetime

def display_analysis_result(analysis_key, table_name, date_col_name, selected_jig=None, used_jig_col=None):
    if st.session_state.analysis_results[analysis_key].empty:
        st.warning("선택한 날짜에 해당하는 분석 데이터가 없습니다.")
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
        
        report_data = {'지표': ['총 테스트 수', 'PASS', '가성불량', '진성불량', 'FAIL']}
        
        for date_iso, date_str in zip([d.strftime('%Y-%m-%d') for d in all_dates], kor_date_cols):
            data_point = summary_data[jig].get(date_iso)
            if data_point:
                report_data[date_str] = [data_point['total_test'], data_point['pass'], data_point['false_defect'], data_point['true_defect'], data_point['fail']]
            else:
                report_data[date_str] = ['N/A'] * 5
        
        report_df = pd.DataFrame(report_data)
        st.table(report_df)
        all_reports_text += report_df.to_csv(index=False) + "\n"

        st.markdown("#### 상세 내역")
        df_filtered = st.session_state.analysis_results[analysis_key]
        
        jig_filtered_df = df_filtered[df_filtered[used_jig_col] == jig].copy()
        
        pass_sns = jig_filtered_df.groupby('SNumber')['PassStatusNorm'].apply(lambda x: 'O' in x.tolist()).loc[lambda x: x].index.tolist()
        formatted_pass_sns = [f"S/N: {s_number}, PC: {jig}" for s_number in pass_sns]
        with st.expander(f"PASS ({len(pass_sns)}건)", expanded=False):
            st.text("\n".join(formatted_pass_sns))
        
        false_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        formatted_false_sns = [f"S/N: {s_number}, PC: {jig}" for s_number in false_defect_sns]
        with st.expander(f"가성불량 ({len(false_defect_sns)}건)", expanded=False):
            st.text("\n".join(formatted_false_sns))
            
        true_defect_sns = jig_filtered_df[(jig_filtered_df['PassStatusNorm'] == 'X') & (~jig_filtered_df['SNumber'].isin(pass_sns))]['SNumber'].unique().tolist()
        formatted_true_sns = [f"S/N: {s_number}, PC: {jig}" for s_number in true_defect_sns]
        with st.expander(f"진성불량 ({len(true_defect_sns)}건)", expanded=False):
            st.text("\n".join(formatted_true_sns))

        fail_sns = jig_filtered_df['SNumber'].unique().tolist()
        all_fail_sns = list(set(fail_sns) - set(pass_sns))
        formatted_fail_sns = [f"S/N: {s_number}, PC: {jig}" for s_number in all_fail_sns]
        with st.expander(f"FAIL ({len(all_fail_sns)}건)", expanded=False):
            st.text("\n".join(formatted_fail_sns))
        
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

def display_data_views(tab_key, df_all_data):
    st.markdown("---")
    snumber_query = st.text_input("SNumber를 입력하세요", key=f"snumber_search_bar_{tab_key}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("SNumber 검색 실행", key=f"snumber_search_btn_{tab_key}"):
            if snumber_query:
                st.session_state.snumber_search[tab_key]['show'] = True
                with st.spinner("데이터베이스에서 SNumber 검색 중..."):
                    filtered_df = df_all_data[
                        df_all_data['SNumber'].fillna('').astype(str).str.contains(snumber_query, case=False, na=False)
                    ].copy()
                
                if not filtered_df.empty:
                    st.success(f"'{snumber_query}'에 대한 {len(filtered_df)}건의 검색 결과를 찾았습니다.")
                    st.session_state.snumber_search[tab_key]['results'] = filtered_df
                else:
                    st.warning(f"'{snumber_query}'에 대한 검색 결과가 없습니다.")
                    st.session_state.snumber_search[tab_key]['results'] = pd.DataFrame()
            else:
                st.warning("SNumber를 입력해주세요.")
                st.session_state.snumber_search[tab_key]['results'] = pd.DataFrame()
    with col2:
        if st.button("원본 DB 조회", key=f"view_last_db_{tab_key}"):
            st.session_state.original_db_view[tab_key]['show'] = True
            if st.session_state.analysis_results[tab_key] is not None:
                st.success(f"{tab_key.upper()} 탭의 원본 데이터를 조회합니다.")
                st.session_state.original_db_view[tab_key]['results'] = st.session_state.analysis_results[tab_key].copy()
            else:
                st.warning(f"먼저 {tab_key.upper()} 탭에서 '분석 실행' 버튼을 눌러 데이터를 분석해주세요.")
                st.session_state.original_db_view[tab_key]['results'] = pd.DataFrame()

    if st.session_state.snumber_search[tab_key]['show'] and not st.session_state.snumber_search[tab_key]['results'].empty:
        st.dataframe(st.session_state.snumber_search[tab_key]['results'].reset_index(drop=True))

    if st.session_state.original_db_view[tab_key]['show'] and not st.session_state.original_db_view[tab_key]['results'].empty:
        st.dataframe(st.session_state.original_db_view[tab_key]['results'].reset_index(drop=True))
