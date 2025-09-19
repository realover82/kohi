import streamlit as st
import pandas as pd
import os
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

@st.cache_data
def load_all_data():
    """
    지정된 CSV 파일 경로에서 모든 데이터를 읽어 DataFrame으로 반환합니다.
    """
    file_paths = {
        'pcb': './src/db/csv/Product-History-SJ_TM2360E_37W_Pcb.csv',
        'fw': './src/db/csv/Product-History-SJ_TM2360E_37W_Fw.csv',
        'rftx': './src/db/csv/Product-History-SJ_TM2360E_37W_RfTx.csv',
        'semi': './src/db/csv/Product-History-SJ_TM2360E_37W_Semi.csv',
        'func': './src/db/csv/Product-History-SJ_TM2360E_37W_Func.csv',
    }
    
    loaded_data = {}
    
    st.info("🔄 지정된 경로에서 CSV 파일을 읽어오는 중...")
    
    for key, path in file_paths.items():
        try:
            # 파일 확장자를 확인하여 CSV 또는 엑셀 파일인지 판별
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                loaded_data[key] = df
                st.success(f"✅ {path} 파일이 성공적으로 로드되었습니다.")
            elif path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
                loaded_data[key] = df
                st.success(f"✅ {path} 파일이 성공적으로 로드되었습니다.")
            else:
                st.error(f"❌ 지원하지 않는 파일 형식입니다: {path}")
                return None
        except FileNotFoundError:
            st.error(f"❌ 파일을 찾을 수 없습니다: {path}")
            st.info("파일 경로를 다시 확인해주세요.")
            return None
        except Exception as e:
            st.error(f"❌ {path} 파일 로드 중 오류가 발생했습니다: {e}")
            return None
            
    return loaded_data

def show_data_info(df):
    """
    DataFrame의 기본 정보를 표시합니다.
    """
    if df is not None and not df.empty:
        st.subheader("데이터 미리보기")
        st.write(df.head())
        st.subheader("데이터 요약")
        st.write(df.describe())


# ---

### `streamlit_app.py` 수정 사항

# `db_utils.py`의 새로운 함수인 `load_all_data()`를 호출하도록 변경하고, 파일 업로드를 기다리는 로직을 제거했습니다.


# http://googleusercontent.com/immersive_entry_chip/0