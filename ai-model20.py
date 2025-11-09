#
# Deepseek 주제별 prompt 챗봇
# 

import streamlit as st
import openai
import requests
import json
from datetime import datetime

# 페이지 설정
st.set_page_config(
    page_title="다중 주제 챗봇 by Kevin",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "topic" not in st.session_state:
    st.session_state.topic = ""

# 사이드바 - 설정
with st.sidebar:
    st.title("챗봇 설정")
    
    # API 키 입력
    # openai_api_key = st.text_input("OpenAI API 키", type="password")
    openai_api_key = st.secrets["api_keys"]["my_api_key"]
    # 모델 선택
    model_choice = st.radio(
        "사용할 모델 선택:",
        ["OpenAI GPT-3.5", "Free LLM (HuggingFace)"]
    )
    
    # 주제 선택
    topic = st.selectbox(
        "주제 선택:",
        ["종교 (성경해설)", "심리학 (고민상담)", "의학 (질병)", "영어 (회화, 해설)", "기타"]
    )
    
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.topic = topic
        st.rerun()

# 주제별 프롬프트 생성 함수
def get_topic_prompt(topic, user_input):
    base_prompts = {
        "종교 (성경해설)": f"""
        당신은 기독교 성경 전문 해설가입니다. 사용자의 질문에 대해 성경 구절을 인용하고,
        그 의미를 현대적 관점에서 쉽게 설명해주세요. 
        답변은 항상 사랑과 긍정의 메시지를 담아야 합니다.
        
        사용자 질문: {user_input}
        
        답변 형식:
        1. 관련 성경 구절 인용 (장:절)
        2. 구절의 역사적/문화적 배경 설명
        3. 현대 생활에 적용할 수 있는 교훈
        4. 격려의 말씀
        
        답변:
        """,
        
        "심리학 (고민상담)": f"""
        당신은 전문 상담 심리학자입니다. 사용자의 고민에 공감하며,
        과학적으로 입증된 심리학적 지식을 바탕으로 조언을 제공해주세요.
        위기 상황에서는 전문가 상담을 권유해야 합니다.
        
        사용자 고민: {user_input}
        
        답변 형식:
        1. 공감과 이해 표현
        2. 관련 심리학 개념 설명
        3. 실용적인 조언과 해결책 제시
        4. 필요한 경우 전문가 상담 권유
        
        답변:
        """,
        
        "의학 (질병)": f"""
        당신은 의학 정보를 제공하는 조수입니다. 사용자의 건강 관련 질문에 대해
        일반적인 정보를 제공하되, 진단이나 치료법을 제시하지는 마세요.
        항상 전문 의료진의 상담을 받을 것을 강조하세요.
        
        사용자 질문: {user_input}
        
        답변 형식:
        1. 질문에 대한 일반적인 의학 정보 제공
        2. 가능한 원인과 증상 설명
        3. 예방법이나 관리 팁
        4. 반드시 전문 의료진 상담 권고
        
        답변:
        """,
        
        "영어 (회화, 해설)": f"""
        당신은 영어 교육 전문가입니다. 사용자의 영어 관련 질문에 대해
        문법, 표현, 발음 등 다양한 측면에서 설명해주세요.
        한국어와 영어를 적절히 혼용하여 설명하되, 예문은 반드시 영어로 제공하세요.
        
        사용자 질문: {user_input}
        
        답변 형식:
        1. 질문의 핵심 개념 설명 (한국어)
        2. 관련 문법/표현 상세 설명
        3. 예문 제시 (영어 + 한국어 해석)
        4. 실전 활용 팁
        
        답변:
        """,
        
        "기타": f"""
        사용자의 질문에 대해 전문적이고 정확한 정보를 제공해주세요.
        특정 주제에 속하지 않는 일반적인 질문에도 친절하게 답변해주세요.
        
        사용자 질문: {user_input}
        
        답변:
        """
    }
    
    return base_prompts.get(topic, base_prompts["기타"])

# OpenAI API 호출 함수
def call_openai(prompt, api_key):
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 다양한 주제에 대해 전문적인 지식을 가진 조수입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API 호출 중 오류 발생: {str(e)}"

# 무료 LLM API 호출 함수 (HuggingFace)
def call_free_llm(prompt):
    try:
        # HuggingFace Inference API 사용 (무료 모델)
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        headers = {"Authorization": "Bearer hf_YourTokenHere"}  # 실제 토큰으로 교체 필요
        
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '응답을 생성할 수 없습니다.')
        else:
            return "무료 모델 응답을 받지 못했습니다. 잠시 후 다시 시도해주세요."
            
    except Exception as e:
        return f"무료 모델 호출 중 오류 발생: {str(e)}"

# 메인 화면
st.title("🤖 다중 주제 챗봇   by Kevin")
st.markdown(f"현재 선택된 주제: **{topic}**")

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 주제별 프롬프트 생성
    enhanced_prompt = get_topic_prompt(topic, prompt)
    
    # 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            if model_choice == "OpenAI GPT-3.5" and openai_api_key:
                response = call_openai(enhanced_prompt, openai_api_key)
            else:
                response = call_free_llm(enhanced_prompt)
            
            st.markdown(response)
    
    # 어시스턴트 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": response})

# 주제별 설명
st.sidebar.markdown("---")
st.sidebar.subheader("주제 설명")

topic_descriptions = {
    "종교 (성경해설)": "성경 구절 해석과 기독교 교리 관련 질문",
    "심리학 (고민상담)": "심리적 고민과 일상 문제 상담",
    "의학 (질병)": "질병 증상과 건강 관리 일반 정보",
    "영어 (회화, 해설)": "영어 학습과 회화 관련 질문",
    "기타": "기타 다양한 주제의 질문"
}

st.sidebar.info(topic_descriptions[topic])

# 주의사항
st.sidebar.markdown("---")
st.sidebar.caption("""
**주의사항:**
- 의학/심리학 상담은 전문가 상담을 대체하지 않습니다
- 중요한 결정은 여러 정보원을 참고하세요
- API 키는 안전하게 관리해주세요
""")