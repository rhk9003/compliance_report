import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from io import StringIO

# --- 設定頁面配置 ---
st.set_page_config(
    page_title="AI 文案法規合規性檢測助手 (下載版)",
    page_icon="⚖️",
    layout="wide"
)

# --- 側邊欄：設定與 API Key ---
with st.sidebar:
    st.header("⚙️ 系統設定")
    
    load_dotenv()
    # 優先從 Streamlit Secrets 讀取 (部署時用)，其次從環境變數
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
    
    # === 新增功能：提供資料庫下載 ===
    st.header("📂 資料庫資源")
    st.info("若您手邊沒有違規案例資料，請先下載此份標準檔案，再上傳至右側分析區。")
    
    # 設定檔案路徑
    db_file_path = os.path.join("data", "violation_db.pdf")
    
    # 檢查檔案是否存在，若存在則顯示下載按鈕
    if os.path.exists(db_file_path):
        with open(db_file_path, "rb") as f:
            st.download_button(
                label="📥 下載「標準違規案例資料庫」",
                data=f,
                file_name="standard_violation_db.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.warning("⚠️ 系統提示：未在 data/ 目錄下找到 violation_db.pdf，無法提供下載。")

    st.markdown("---")
    st.caption("Core Model: Gemini 3 Pro Preview")

# --- 核心功能函式 ---

def extract_text_from_uploaded_file(uploaded_file):
    """從使用者手動上傳的檔案中提取文字"""
    if uploaded_file is None:
        return ""
    
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        
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
        1. **使用者提供的【違規案例資料庫】（最高優先級）**：你必須比對文案是否包含與資料庫中「違規情節」相似的詞彙、邏輯或暗示（例如：若資料庫中有「小腹橡皮擦」違規，則「腰間肉橡皮擦」也應視為高風險）。
        2. **台灣現行法規**：包含《食品安全衛生管理法》第28條（誇大不實、醫療效能）、《健康食品管理法》。

        你必須對「療效」、「保證」、「快速瘦身」、「再生」、「回春」、「抗炎」等敏感概念保持極度警戒。
        """

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        
        prompt = f"""
        請針對以下【待審核文案】，進行深度合規性分析。

        ### 1. 參考法規與判例資料庫：
        {reference_data if reference_data else "使用者未上傳資料庫，請依據台灣一般法規常識判斷。"}

        ### 2. 待審核文案：
        {ad_copy}

        ---
        ### 請輸出分析報告（請使用 Markdown 格式）：

        1.  **總體風險評級**：(安全 / 低風險 / 中風險 / 高風險 / 極高風險-必罰)
        2.  **關鍵違規熱點分析** (請詳細列出)：
            * **違規詞句**：列出文案中的具體問題句子。
            * **比對結果**：明確指出違反了【資料庫】中的哪一類邏輯或哪個具體案例（若有）。
            * **風險解釋**：為什麼這樣寫不行？（例如：涉及改變身體外觀、涉及醫療效能）。
        3.  **合規修改建議**：
            * 針對每一個違規點，提供「安全替代詞彙」或「寫法」。
            * *關鍵挑戰*：請嘗試保留行銷吸引力，將「療效宣稱」轉化為「營養補給」或「生理機能調節」的合規敘述。
        """

        generation_config = genai.types.GenerationConfig(
            temperature=0.1, 
            top_p=0.8,
            top_k=40
        )

        with st.spinner(f"正在使用 {model_name} 對照【上傳的資料庫】進行推理..."):
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

st.title("🛡️ 台灣行銷文案法規快篩系統")
st.markdown("利用 **Gemini 3 Pro**，比對您上傳的 **違規案例資料庫** 與文案。")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 匯入知識庫 (必要的判斷依據)")
    st.markdown("請上傳法規資料庫 (PDF/TXT)。若無檔案，可由左側選單下載標準範本。")
    
    ref_file = st.file_uploader("上傳資料庫", type=["txt", "pdf"], key="ref_file")
    
    ref_text = ""
    if ref_file:
        ref_text = extract_text_from_uploaded_file(ref_file)
        st.success(f"✅ 資料庫載入成功！包含 {len(ref_text)} 字的判例。")
    else:
        st.warning("⚠️ 等待上傳資料庫... (若未上傳，AI 僅能憑內建知識判斷，準確度較低)")

with col2:
    st.subheader("2. 待審核文案")
    
    input_method = st.radio("輸入方式", ["直接貼上文字", "上傳文案檔案"], horizontal=True)
    
    ad_copy_text = ""
    
    if input_method == "直接貼上文字":
        ad_copy_text = st.text_area("請在此貼上文案內容", height=300, placeholder="例如：這款酵素能讓你躺著就瘦，7天保證見效...")
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
    elif not api_key:
        st.warning("⚠️ 請在側邊欄輸入 API Key")
    else:
        # 如果使用者沒有上傳參考資料，給予最後提示，但仍允許執行
        if not ref_text:
            st.toast("⚠️ 注意：您未上傳資料庫，AI 將僅依據內建法規進行分析。", icon="⚠️")
        
        result = analyze_compliance(api_key, ad_copy_text, ref_text)
        if result:
            st.markdown("## 📋 分析報告")
            st.markdown(result)
            
            st.download_button(
                label="📥 下載分析報告 (Markdown)",
                data=result,
                file_name="compliance_report.md",
                mime="text/markdown"
            )
