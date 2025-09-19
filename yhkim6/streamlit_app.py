import streamlit as st
import pandas as pd
from datetime import datetime, date
import warnings
import sys
import os

# 현재 파일의 절대 경로를 기준으로 프로젝트 루트 디렉토리를 찾습니다.
project_root = os.path.dirname(os.path.abspath(__file__))

# src 폴더를 Python path에 추가
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 프로젝트 루트도 추가 (혹시 모를 경우를 대비)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 에러 처리를 위한 플래그
modules_loaded = False

# 수정된 db_utils.py에서 필요한 모듈을 import 합니다.
try:
    from db.db_utils import show_csv_uploaders
    from services.analysis_service import analyze_data
    from utils.ui_helpers import display_analysis_result, display_data_views
    modules_loaded = True
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"❌ 모듈 로드 실패: {e}")
    st.info("""
    **해결 방법:**
    1. src/ 폴더와 __init__.py 파일이 올바른 위치에 있는지 확인하세요.
    2. 필요한 모든 모듈 파일(db_utils.py, analysis_service.py 등)이 존재하는지 확인하세요.
    """)
    st.stop()

warnings.filterwarnings('ignore')

# 세션 상태 초기화
def initialize_session_state():
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {key: None for key in ['pcb', 'fw', 'rftx', 'semi', 'func']}
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {key: None for key in ['pcb', 'fw', 'rftx', 'semi', 'func']}
    if 'analysis_time' not in st.session_state:
        st.session_state.analysis_time = {key: None for key in ['pcb', 'fw', 'rftx', 'semi', 'func']}
    if 'jig_col_mapping' not in st.session_state:
        st.session_state.jig_col_mapping = {
            'pcb': 'PcbMaxIrPwr',
            'fw': 'FwPC',
            'rftx': 'RfTxPC',
            'semi': 'SemiAssyPC',
            'func': 'FwPC',
        }
    if 'show_line_chart' not in st.session_state:
        st.session_state.show_line_chart = {}
    if 'show_bar_chart' not in st.session_state:
        st.session_state.show_bar_chart = {}
    if 'analysis_status' not in st.session_state:
        st.session_state.analysis_status = {
            key: {'analyzed': False} for key in ['pcb', 'fw', 'rftx', 'semi', 'func']
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

def main():
    st.set_page_config(layout="wide")
    st.title("리모컨 생산 데이터 분석 툴")
    st.markdown("---")
    initialize_session_state()

    if not modules_loaded:
        st.stop()

    # 5개의 CSV 파일을 업로드하는 위젯을 표시하고 DataFrame 딕셔너리를 받습니다.
    uploaded_files = show_csv_uploaders()
    
    tab_keys = ['pcb', 'fw', 'rftx', 'semi', 'func']
    
    # 모든 파일이 업로드되었는지 확인합니다.
    if not all(key in uploaded_files for key in tab_keys):
        st.info("⬆️ 모든 분석을 시작하려면 5개의 파일을 모두 업로드해주세요.")
        st.stop()

    st.success("✅ 모든 파일 로드 성공!")

    # 각 탭의 정보를 정의하고, 날짜 컬럼을 변환합니다.
    tab_info = {
        'pcb': {'header': "파일 PCB (Pcb_Process)", 'date_col': 'PcbStartTime_dt', 'data': uploaded_files['pcb']},
        'fw': {'header': "파일 Fw (Fw_Process)", 'date_col': 'FwStamp_dt', 'data': uploaded_files['fw']},
        'rftx': {'header': "파일 RfTx (RfTx_Process)", 'date_col': 'RfTxStamp_dt', 'data': uploaded_files['rftx']},
        'semi': {'header': "파일 Semi (SemiAssy_Process)", 'date_col': 'SemiAssyStartTime_dt', 'data': uploaded_files['semi']},
        'func': {'header': "파일 Func (Func_Process)", 'date_col': 'BatadcStamp_dt', 'data': uploaded_files['func']}
    }

    # 각 DataFrame에 대해 날짜 컬럼을 변환합니다.
    for key in tab_keys:
        df_tab = tab_info[key]['data']
        date_col_name = tab_info[key]['date_col'].replace('_dt', '')
        
        try:
            df_tab[tab_info[key]['date_col']] = pd.to_datetime(df_tab[date_col_name], errors='coerce')
        except KeyError as e:
            st.error(f"❌ {key.upper()} 데이터에서 날짜 컬럼 '{date_col_name}'을 찾을 수 없습니다.")
            st.write(f"현재 {key.upper()} 데이터 컬럼:", list(df_tab.columns))
            st.stop()
        
    st.success("✅ 모든 날짜 컬럼 변환 완료")

    tabs = st.tabs(tab_keys)

    for i, tab_key in enumerate(tab_info.keys()):
        with tabs[i]:
            st.header(tab_info[tab_key]['header'])
            df_current_tab = tab_info[tab_key]['data']

            if df_current_tab.empty:
                st.warning("⚠️ 업로드된 파일에 데이터가 없습니다. 다른 파일을 선택해주세요.")
                continue

            try:
                jig_col_name = st.session_state.jig_col_mapping[tab_key]
                if jig_col_name not in df_current_tab.columns:
                    st.warning(f"⚠️ '{jig_col_name}' 컬럼을 찾을 수 없습니다. 'SNumber'를 사용합니다.")
                    jig_col_name = 'SNumber'
                
                unique_jigs = df_current_tab[jig_col_name].dropna().unique()
                pc_options = ['모든 PC'] + sorted(list(unique_jigs))
                selected_jig = st.selectbox("PC (Jig) 선택", pc_options, key=f"pc_select_{tab_key}")

                date_col = tab_info[tab_key]['date_col']
                
                df_dates = df_current_tab[date_col].dt.date.dropna()
                if not df_dates.empty:
                    min_date = df_dates.min()
                    max_date = df_dates.max()
                else:
                    min_date = max_date = date.today()
                
                selected_dates = st.date_input("날짜 범위 선택", value=(min_date, max_date), key=f"dates_{tab_key}")
                
                if st.button("분석 실행", key=f"analyze_{tab_key}"):
                    with st.spinner("데이터 분석 및 저장 중..."):
                        if len(selected_dates) == 2:
                            start_date, end_date = selected_dates
                            df_filtered = df_current_tab[
                                (df_current_tab[date_col].dt.date >= start_date) &
                                (df_current_tab[date_col].dt.date <= end_date)
                            ].copy()
                            if selected_jig != '모든 PC':
                                df_filtered = df_filtered[df_filtered[jig_col_name] == selected_jig].copy()
                        else:
                            st.warning("날짜 범위를 올바르게 선택해주세요.")
                            df_filtered = pd.DataFrame()
                        
                        st.session_state.analysis_results[tab_key] = df_filtered
                        st.session_state.analysis_data[tab_key] = analyze_data(df_filtered, date_col, jig_col_name)
                        st.session_state.analysis_time[tab_key] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state.analysis_status[tab_key]['analyzed'] = True
                    st.success("분석 완료! 결과가 저장되었습니다.")

                if st.session_state.analysis_status[tab_key]['analyzed']:
                    display_analysis_result(tab_key, tab_info[tab_key]['header'], date_col,
                                            selected_jig=selected_jig if selected_jig != '모든 PC' else None,
                                            used_jig_col=st.session_state.analysis_data[tab_key][2])
                
                st.markdown("---")
                st.markdown(f"#### {tab_info[tab_key]['header'].split()[1]} 데이터 조회")
                display_data_views(tab_key, df_current_tab)
                
            except Exception as e:
                st.error(f"❌ 탭 '{tab_key}' 처리 중 오류: {e}")
                st.info("이 탭은 건너뛰고 다른 탭을 사용해보세요.")

    st.markdown("---")
    st.markdown("<p style='text-align:center'>Copyright © 2024</p>", unsafe_allow_html=True)
            
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ 앱 실행 중 치명적 오류 발생: {e}")
        st.info("개발자에게 문의하거나 로그를 확인해주세요.")
