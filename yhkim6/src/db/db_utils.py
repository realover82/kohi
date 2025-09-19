import streamlit as st
import pandas as pd
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

@st.cache_data
def load_data_from_file(uploaded_file):
    """
    업로드된 CSV 또는 Excel 파일에서 데이터를 읽어 DataFrame으로 반환합니다.
    """
    if uploaded_file is not None:
        try:
            # 파일 확장자를 확인하여 CSV 또는 엑셀 파일인지 판별
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                return df
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
                return df
            else:
                st.error("❌ 지원하지 않는 파일 형식입니다. CSV 또는 Excel 파일을 업로드해주세요.")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ 파일 로드 중 오류가 발생했습니다: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def show_csv_uploaders():
    """
    Streamlit UI에 5개의 파일 업로드 위젯을 표시하고 DataFrame 딕셔너리를 반환합니다.
    """
    st.subheader("데이터 파일 업로드")
    st.info("각 분석 탭에 해당하는 파일을 업로드해주세요. 모든 파일이 필요합니다.")
    
    uploaded_files = {}
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("---")
        pcb_file = st.file_uploader("PCB 데이터 파일 (csv, xlsx)", type=["csv", "xlsx"], key="pcb_uploader")
        if pcb_file:
            uploaded_files['pcb'] = load_data_from_file(pcb_file)
            st.success("✅ PCB 파일 로드 완료")
    
        fw_file = st.file_uploader("Fw 데이터 파일 (csv, xlsx)", type=["csv", "xlsx"], key="fw_uploader")
        if fw_file:
            uploaded_files['fw'] = load_data_from_file(fw_file)
            st.success("✅ Fw 파일 로드 완료")
            
    with col2:
        st.write("---")
        rftx_file = st.file_uploader("RfTx 데이터 파일 (csv, xlsx)", type=["csv", "xlsx"], key="rftx_uploader")
        if rftx_file:
            uploaded_files['rftx'] = load_data_from_file(rftx_file)
            st.success("✅ RfTx 파일 로드 완료")

        semi_file = st.file_uploader("Semi 데이터 파일 (csv, xlsx)", type=["csv", "xlsx"], key="semi_uploader")
        if semi_file:
            uploaded_files['semi'] = load_data_from_file(semi_file)
            st.success("✅ Semi 파일 로드 완료")
            
        func_file = st.file_uploader("Func 데이터 파일 (csv, xlsx)", type=["csv", "xlsx"], key="func_uploader")
        if func_file:
            uploaded_files['func'] = load_data_from_file(func_file)
            st.success("✅ Func 파일 로드 완료")

    return uploaded_files

def show_data_info(df):
    """
    DataFrame의 기본 정보를 표시합니다.
    """
    if df is not None and not df.empty:
        st.subheader("데이터 미리보기")
        st.write(df.head())
        st.subheader("데이터 요약")
        st.write(df.describe())
        
