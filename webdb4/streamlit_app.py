import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import io

# 각 CSV 분석 모듈 불러오기 (기존 코드 유지)
from csv2 import read_csv_with_dynamic_header, analyze_data
from csv_Fw import read_csv_with_dynamic_header_for_Fw, analyze_Fw_data
from csv_RfTx import read_csv_with_dynamic_header_for_RfTx, analyze_RfTx_data
from csv_Semi import read_csv_with_dynamic_header_for_Semi, analyze_Semi_data
from csv_Batadc import read_csv_with_dynamic_header_for_Batadc, analyze_Batadc_data

def display_analysis_result(analysis_key, file_name, jig_col_name):
    """ session_state에 저장된 분석 결과를 Streamlit에 표시하는 함수 """
      # 아래 한 줄을 추가합니다.
    # st.json(st.session_state.analysis_data[analysis_key][0])

    if st.session_state.analysis_results[analysis_key] is None:
        st.error("데이터 로드에 실패했습니다. 파일 형식을 확인해주세요.")
        return

    summary_data, all_dates = st.session_state.analysis_data[analysis_key]

    st.markdown(f"### '{file_name}' 분석 리포트")

    # Jig 목록 추출
    df_filtered = st.session_state.analysis_results[analysis_key]
    jig_list = sorted(df_filtered[jig_col_name].dropna().unique().tolist()) if jig_col_name in df_filtered.columns else []

    # PC(Jig) 선택 UI
    selected_jig = st.selectbox("PC(Jig) 선택", ["전체"] + jig_list, key=f"select_{analysis_key}")

    # --- 날짜 필터링 기능 추가 ---
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    date_range_col1, date_range_col2 = st.columns(2)
    with date_range_col1:
        start_date = st.date_input("시작 날짜", min_date, min_value=min_date, max_value=max_date, key=f"start_date_{analysis_key}")
    with date_range_col2:
        end_date = st.date_input("종료 날짜", max_date, min_value=min_date, max_value=max_date, key=f"end_date_{analysis_key}")

    # 선택된 날짜 범위에 맞는 데이터 필터링
    filtered_dates = [d for d in all_dates if start_date <= d <= end_date]
    if not filtered_dates:
        st.warning("선택된 날짜 범위에 해당하는 데이터가 없습니다.")
        return

    st.write(f"**분석 시간**: {st.session_state.analysis_time[analysis_key]}")
    st.markdown("---")

    # --- 시각화 방식 선택 버튼 추가 ---
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
        
    all_reports_text = ""

    # 선택된 jig만 보여주기
    jigs_to_display = jig_list if selected_jig == "전체" else [selected_jig]

    for jig in jigs_to_display:
        st.subheader(f"구분: {jig}")

        # --- 선택된 시각화 방식에 따라 표 또는 그래프 표시 ---
        if st.session_state[f'display_mode_{analysis_key}'] == 'table':
            report_data = {
                '지표': ['총 테스트 수', 'PASS', '가성불량', '진성불량', 'FAIL']
            }
            kor_date_cols = [f"{d.strftime('%y%m%d')}" for d in filtered_dates]

            for date_iso, date_str in zip([d.strftime('%Y-%m-%d') for d in filtered_dates], kor_date_cols):
                data_point = summary_data.get(jig, {}).get(date_iso)
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
            
            report_df = pd.DataFrame(report_data).set_index('지표')
            st.table(report_df)
            all_reports_text += report_df.to_csv(index=True) + "\n"

        else: # 그래프 (꺾은선, 막대)
            chart_data = {
                '날짜': [d.strftime('%Y-%m-%d') for d in filtered_dates],
                'PASS': [],
                '가성불량': [],
                '진성불량': [],
                'FAIL': []
            }
            
            for date_iso in chart_data['날짜']:
                data_point = summary_data.get(jig, {}).get(date_iso)
                if data_point:
                    chart_data['PASS'].append(data_point['pass'])
                    chart_data['가성불량'].append(data_point['false_defect'])
                    chart_data['진성불량'].append(data_point['true_defect'])
                    chart_data['FAIL'].append(data_point['fail'])
                else:
                    chart_data['PASS'].append(0)
                    chart_data['가성불량'].append(0)
                    chart_data['진성불량'].append(0)
                    chart_data['FAIL'].append(0)
            
            chart_df = pd.DataFrame(chart_data).set_index('날짜')
            
            if st.session_state[f'display_mode_{analysis_key}'] == 'line':
                st.line_chart(chart_df)
            else: # 'bar'
                st.bar_chart(chart_df)

        st.markdown("---") # 각 지그 구분선
        
        # 상세 내역 (날짜별로 가져오기)
        st.markdown("#### 상세 내역")
        st.markdown("---")

        # 버튼 클릭 상태를 저장할 딕셔너리
        if f'show_details_{analysis_key}' not in st.session_state:
            st.session_state[f'show_details_{analysis_key}'] = {'pass': False, 'false_defect': False, 'true_defect': False, 'fail': False}

        # 버튼 생성
        expander_cols = st.columns(4)
        with expander_cols[0]:
            if st.button("PASS 상세", key=f"pass_btn_{analysis_key}"):
                st.session_state[f'show_details_{analysis_key}']['pass'] = not st.session_state[f'show_details_{analysis_key}']['pass']
        with expander_cols[1]:
            if st.button("가성불량 상세", key=f"false_defect_btn_{analysis_key}"):
                st.session_state[f'show_details_{analysis_key}']['false_defect'] = not st.session_state[f'show_details_{analysis_key}']['false_defect']
        with expander_cols[2]:
            if st.button("진성불량 상세", key=f"true_defect_btn_{analysis_key}"):
                st.session_state[f'show_details_{analysis_key}']['true_defect'] = not st.session_state[f'show_details_{analysis_key}']['true_defect']
        with expander_cols[3]:
            if st.button("FAIL 상세", key=f"fail_btn_{analysis_key}"):
                st.session_state[f'show_details_{analysis_key}']['fail'] = not st.session_state[f'show_details_{analysis_key}']['fail']
        
        # 클릭된 버튼에 따라 상세 내역 표시
        for d in filtered_dates:
            date_iso = d.strftime('%Y-%m-%d')
            data_point = summary_data.get(jig, {}).get(date_iso)
            
            if data_point:
                date_str = d.strftime('%y%m%d')
                
                # 수정: len() 대신 summary_data에서 직접 값을 가져와 사용
                if st.session_state[f'show_details_{analysis_key}']['pass']:
                    pass_sns_list = data_point.get('pass_sns', [])
                    pass_count = data_point.get('pass', 0)
                    with st.expander(f"PASS ({date_str}) - {pass_count}건", expanded=True):
                        if pass_sns_list:
                            st.text("\n".join(pass_sns_list))
                        else:
                            st.text("해당 날짜에 상세 내역이 없습니다.")
                
                if st.session_state[f'show_details_{analysis_key}']['false_defect']:
                    false_defect_sns_list = data_point.get('false_defect_sns', [])
                    false_defect_count = data_point.get('false_defect', 0)
                    with st.expander(f"가성불량 ({date_str}) - {false_defect_count}건", expanded=True):
                        if false_defect_sns_list:
                            st.text("\n".join(false_defect_sns_list))
                        else:
                            st.text("해당 날짜에 상세 내역이 없습니다.")

                if st.session_state[f'show_details_{analysis_key}']['true_defect']:
                    true_defect_sns_list = data_point.get('true_defect_sns', [])
                    true_defect_count = data_point.get('true_defect', 0)
                    with st.expander(f"진성불량 ({date_str}) - {true_defect_count}건", expanded=True):
                        if true_defect_sns_list:
                            st.text("\n".join(true_defect_sns_list))
                        else:
                            st.text("해당 날짜에 상세 내역이 없습니다.")

                if st.session_state[f'show_details_{analysis_key}']['fail']:
                    fail_sns_list = data_point.get('fail_sns', [])
                    fail_count = data_point.get('fail', 0)
                    with st.expander(f"FAIL ({date_str}) - {fail_count}건", expanded=True):
                        if fail_sns_list:
                            st.text("\n".join(fail_sns_list))
                        else:
                            st.text("해당 날짜에 상세 내역이 없습니다.")

        st.markdown("---") # 각 지그 구분선

    st.success("분석이 완료되었습니다!")

    # 다운로드 버튼
    st.download_button(
        label="분석 결과 다운로드",
        data=all_reports_text.encode('utf-8-sig'),
        file_name=f"{file_name}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ==============================
# 파일 읽기 (Cache 적용)
# ==============================
@st.cache_data
def read_pcb_data(uploaded_file):
    return read_csv_with_dynamic_header(uploaded_file)

@st.cache_data
def read_fw_data(uploaded_file):
    return read_csv_with_dynamic_header_for_Fw(uploaded_file)

@st.cache_data
def read_rftx_data(uploaded_file):
    return read_csv_with_dynamic_header_for_RfTx(uploaded_file)

@st.cache_data
def read_semi_data(uploaded_file):
    return read_csv_with_dynamic_header_for_Semi(uploaded_file)

@st.cache_data
def read_batadc_data(uploaded_file):
    return read_csv_with_dynamic_header_for_Batadc(uploaded_file)


# ==============================
# 메인 실행 함수
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["파일 PCB 분석", "파일 Fw 분석", "파일 RfTx 분석", "파일 Semi 분석", "파일 Func 분석"]
    )

    # PCB
    with tab1:
        st.header("파일 PCB (Pcb_Process)")
        st.session_state.uploaded_files['pcb'] = st.file_uploader("파일 PCB를 선택하세요", type=["csv"], key="uploader_pcb")
        if st.session_state.uploaded_files['pcb']:
            if st.button("파일 PCB 분석 실행", key="analyze_pcb"):
                df = read_pcb_data(st.session_state.uploaded_files['pcb'])
                if df is not None:
                    with st.spinner("데이터 분석 및 저장 중..."):
                        st.session_state.analysis_results['pcb'] = df
                        st.session_state.analysis_data['pcb'] = analyze_data(df)
                        st.session_state.analysis_time['pcb'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.success("분석 완료! 결과가 저장되었습니다.")
                else:
                    st.error("PCB 데이터 파일을 읽을 수 없습니다.")
        if st.session_state.analysis_results['pcb'] is not None:
            display_analysis_result('pcb', st.session_state.uploaded_files['pcb'].name, 'PcbMaxIrPwr')

    # Fw
    with tab2:
        st.header("파일 Fw (Fw_Process)")
        st.session_state.uploaded_files['fw'] = st.file_uploader("파일 Fw를 선택하세요", type=["csv"], key="uploader_fw")
        if st.session_state.uploaded_files['fw']:
            if st.button("파일 Fw 분석 실행", key="analyze_fw"):
                df = read_fw_data(st.session_state.uploaded_files['fw'])
                if df is not None:
                    with st.spinner("데이터 분석 및 저장 중..."):
                        st.session_state.analysis_results['fw'] = df
                        st.session_state.analysis_data['fw'] = analyze_Fw_data(df)
                        st.session_state.analysis_time['fw'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.success("분석 완료! 결과가 저장되었습니다.")
                else:
                    st.error("Fw 데이터 파일을 읽을 수 없습니다.")
        if st.session_state.analysis_results['fw'] is not None:
            display_analysis_result('fw', st.session_state.uploaded_files['fw'].name, 'FwPC')

    # RfTx
    with tab3:
        st.header("파일 RfTx (RfTx_Process)")
        st.session_state.uploaded_files['rftx'] = st.file_uploader("파일 RfTx를 선택하세요", type=["csv"], key="uploader_rftx")
        if st.session_state.uploaded_files['rftx']:
            if st.button("파일 RfTx 분석 실행", key="analyze_rftx"):
                df = read_rftx_data(st.session_state.uploaded_files['rftx'])
                if df is not None:
                    with st.spinner("데이터 분석 및 저장 중..."):
                        st.session_state.analysis_results['rftx'] = df
                        st.session_state.analysis_data['rftx'] = analyze_RfTx_data(df)
                        st.session_state.analysis_time['rftx'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.success("분석 완료! 결과가 저장되었습니다.")
                else:
                    st.error("RfTx 데이터 파일을 읽을 수 없습니다.")
        if st.session_state.analysis_results['rftx'] is not None:
            display_analysis_result('rftx', st.session_state.uploaded_files['rftx'].name, 'RfTxPC')

    # Semi
    with tab4:
        st.header("파일 Semi (SemiAssy_Process)")
        st.session_state.uploaded_files['semi'] = st.file_uploader("파일 Semi를 선택하세요", type=["csv"], key="uploader_semi")
        if st.session_state.uploaded_files['semi']:
            if st.button("파일 Semi 분석 실행", key="analyze_semi"):
                df = read_semi_data(st.session_state.uploaded_files['semi'])
                if df is not None:
                    with st.spinner("데이터 분석 및 저장 중..."):
                        st.session_state.analysis_results['semi'] = df
                        st.session_state.analysis_data['semi'] = analyze_Semi_data(df)
                        st.session_state.analysis_time['semi'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.success("분석 완료! 결과가 저장되었습니다.")
                else:
                    st.error("Semi 데이터 파일을 읽을 수 없습니다.")
        if st.session_state.analysis_results['semi'] is not None:
            display_analysis_result('semi', st.session_state.uploaded_files['semi'].name, 'SemiAssyMaxBatVolt')

    # Func
    with tab5:
        st.header("파일 Func (Func_Process)")
        st.session_state.uploaded_files['func'] = st.file_uploader("파일 Func를 선택하세요", type=["csv"], key="uploader_func")
        if st.session_state.uploaded_files['func']:
            if st.button("파일 Func 분석 실행", key="analyze_func"):
                df = read_batadc_data(st.session_state.uploaded_files['func'])
                if df is not None:
                    with st.spinner("데이터 분석 및 저장 중..."):
                        st.session_state.analysis_results['func'] = df
                        st.session_state.analysis_data['func'] = analyze_Batadc_data(df)
                        st.session_state.analysis_time['func'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.success("분석 완료! 결과가 저장되었습니다.")
                else:
                    st.error("Func 데이터 파일을 읽을 수 없습니다.")
        if st.session_state.analysis_results['func'] is not None:
            display_analysis_result('func', st.session_state.uploaded_files['func'].name, 'BatadcPC')


if __name__ == "__main__":
    main()
