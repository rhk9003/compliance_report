import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from io import StringIO

# --- 設定頁面配置 ---
st.set_page_config(
    page_title="AI 文案法規合規性檢測助手 (GitHub版)",
    page_icon="⚖️",
    layout="wide"
)

# --- 側邊欄：設定與 API Key ---
with st.sidebar:
    st.header("⚙️ 系統設定")
    
    load_dotenv()
    # 優先從 Streamlit Secrets 讀取 (部署時用)，其次從環境變數，最後才是輸入框
    # 如果您部署到 Streamlit Cloud，API Key 會設在 Secrets 裡
    env_api_key = os.getenv("GOOGLE_API_KEY")
    if "GOOGLE_API_KEY" in st.secrets:
        env_api_key = st.secrets["GOOGLE_API_KEY"]

    api_key = st.text_input(
        "輸入 Google Gemini API Key",
        value=env_api_key if env_api_key else "",
        type="password",
        help="請輸入您的 Gemini API Key 以啟動服務"
    )

    st.markdown("---")
    st.info(
        "**核心模型：**\n"
        "優先：Gemini 3 Pro Preview\n"
        "備援：Gemini 2.5 Pro"
    )

# --- 核心功能函式 ---

def read_pdf(file_path_or_buffer):
    """讀取 PDF 內容 (支援路徑或上傳的緩衝區)"""
    try:
        reader = PyPDF2.PdfReader(file_path_or_buffer)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def load_default_database():
    """從專案目錄中自動讀取預設的資料庫檔案"""
    # 設定您的檔案路徑 (請確保檔案名稱與此處一致)
    default_db_path = os.path.join("data", "violation_db.pdf") 
    
    if os.path.exists(default_db_path):
        return read_pdf(default_db_path), True
    else:
        return "", False

def extract_text_from_uploaded_file(uploaded_file):
    """從使用者手動上傳的檔案中提取文字"""
    if uploaded_file is None:
        return ""
    
    try:
        if uploaded_file.type == "application/pdf":
            return read_pdf(uploaded_file)
        
        elif uploaded_file.type == "text/plain":
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            return stringio.read()
        
        else:
            return ""
    except Exception as e:
        st.error(f"檔案讀取失敗: {e}")
        return ""

def analyze_compliance(api_key, ad_copy, reference_data):
    """呼叫 Gemini 進行法規比對"""
    if not api_key:
        st.error("請先輸入 API Key")
        return None

    genai.configure(api_key=api_key)
    model_name = "gemini-3-pro-preview" 
    
    try:
        system_instruction = """
        你是一位精通台灣法規的「首席合規長 (Chief Compliance Officer)」。
        你的核心任務是保護使用者免於因廣告違規而受罰。
        
        你必須依據以下兩大準則進行嚴格審查：
        1. **內部黃金準則（最高優先級）**：參考資料庫中的「違規案例」與「避雷指南」。若文案包含類似的詞彙、誇張邏輯或暗示（例如：資料庫說「小腹橡皮擦」違規，則「腰間肉立可白」也應視為高風險）。
        2. **台灣現行法規**：食安法第28條（誇大不實、醫療效能）、健康食品管理法。

        你必須對「療效」、「保證」、「快速瘦身」、「再生」、「回春」、「抗炎」等敏感概念保持極度警戒。
        """

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        prompt = f"""
        請針對以下【待審核文案】，進行深度合規性分析。

        ### 1. 黃金比對資料庫 (包含過往違規判例與指南)：
        {reference_data}

        ### 2. 待審核文案：
        {ad_copy}

        ---
        ### 請輸出分析報告（Markdown 格式）：

        1.  **總體風險評級**：(安全 / 低風險 / 中風險 / 高 / 極高-必罰)
        2.  **關鍵違規熱點分析**：
            * **違規詞句**：列出文案中的具體問題句子。
            * **比對結果**：明確指出違反了【資料庫】中的哪一類邏輯或哪個具體案例。
            * **風險解釋**：為什麼這樣寫不行？
        3.  **合規修改建議**：
            * 針對每一個違規點，提供「安全替代詞彙」或「寫法」。
            * *關鍵挑戰*：請嘗試保留行銷吸引力，將「療效宣稱」轉化為「營養補給」或「生理機能調節」的合規敘述。
        4.  **行銷邏輯檢視**：
            * 修改後的文案是否仍具備吸引力？
        """

        generation_config = genai.types.GenerationConfig(
            temperature=0.1, 
            top_p=0.8,
            top_k=40
        )

        with st.spinner(f"正在使用 {model_name} 對照【內部資料庫】進行推理..."):
            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text

    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():
            try:
                fallback_model = "gemini-2.5-pro"
                st.warning(f"{model_name} 無法存取，切換至 {fallback_model} 進行分析...")
                model = genai.GenerativeModel(fallback_model, system_instruction=system_instruction)
                response = model.generate_content(prompt, generation_config=generation_config)
                return response.text
            except Exception as e2:
                st.error(f"分析發生錯誤: {str(e2)}")
                return None
        else:
            st.error(f"分析發生錯誤: {str(e)}")
            return None

# --- 主介面 ---

st.title("🛡️ 台灣行銷文案法規快篩系統 (Auto-Load)")
st.markdown("利用 **Gemini 3 Pro**，自動比對 **GitHub 資料庫** 與您的文案。")

# --- 自動載入預設資料庫 ---
default_text, is_loaded = load_default_database()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 知識庫狀態")
    
    # 顯示預設資料庫狀態
    if is_loaded:
        st.success(f"✅ 已自動載入 GitHub/本地 資料庫 ({len(default_text)} 字)")
        with st.expander("預覽核心資料庫內容"):
            st.text(default_text[:1000] + "...")
    else:
        st.error("❌ 未偵測到 `data/violation_db.pdf`，請確認檔案已上傳至 GitHub 或本地目錄。")

    st.markdown("---")
    st.write("**補充資料 (選填)**：")
    st.caption("如果有最新的法規或這次專案的特殊規範，可以在此額外上傳。")
    supplementary_file = st.file_uploader("上傳補充文件 (TXT/PDF)", type=["txt", "pdf"], key="supp_file")
    
    supplementary_text = ""
    if supplementary_file:
        supplementary_text = extract_text_from_uploaded_file(supplementary_file)
        st.info(f"➕ 已加入補充資料 ({len(supplementary_text)} 字)")

with col2:
    st.subheader("2. 待審核文案")
    
    input_method = st.radio("輸入方式", ["直接貼上文字", "上傳文案檔案"], horizontal=True)
    
    ad_copy_text = ""
    
    if input_method == "直接貼上文字":
        ad_copy_text = st.text_area("請在此貼上文案內容", height=300, placeholder="例如：這款產品能讓你的小腹像橡皮擦一樣消失，保證3天見效...")
    else:
        ad_file = st.file_uploader("上傳文案 (TXT/PDF)", type=["txt", "pdf"], key="ad_file")
        if ad_file:
            ad_copy_text = extract_text_from_uploaded_file(ad_file)
            st.success(f"已讀取文案，長度：{len(ad_copy_text)} 字")

# --- 執行按鈕 ---
st.markdown("---")
if st.button("🚀 開始法規合規性分析", type="primary", use_container_width=True):
    if not ad_copy_text:
        st.warning("⚠️ 請務必提供「待審核文案」")
    elif not is_loaded and not supplementary_text:
        st.warning("⚠️ 警告：沒有載入任何法規資料庫，分析可能不準確。建議檢查 data 資料夾或上傳檔案。")
        # 即使沒有資料庫，若使用者堅持也可以跑，但給予警告
        full_reference = ""
        result = analyze_compliance(api_key, ad_copy_text, full_reference)
        if result:
            st.markdown(result)
            
    else:
        # 合併 預設資料庫 + 補充資料
        full_reference = f"{default_text}\n\n=== 以下為補充資料 ===\n{supplementary_text}"
        
        result = analyze_compliance(api_key, ad_copy_text, full_reference)
        if result:
            st.markdown("## 📋 分析報告")
            st.markdown(result)
            
            st.download_button(
                label="📥 下載分析報告 (Markdown)",
                data=result,
                file_name="compliance_report.md",
                mime="text/markdown"
            )
