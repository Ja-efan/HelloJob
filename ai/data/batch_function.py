import openai
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from pydantic import BaseModel

def get_business_categories_batch(df, api_key, batch_size=10):
    """
    데이터프레임의 기관 정보를 batch_size만큼 묶어서 한 번에 OpenAI API에 요청하여 사업 종목을 반환하는 함수
    
    Args:
        df (DataFrame): 기관 정보가 담긴 데이터프레임
        api_key (str): OpenAI API 키
        batch_size (int): 한 번에 처리할 기관의 수
        
    Returns:
        list: 사업 종목 정보가 담긴 객체 리스트
    """
    
    system_prompt = """당신은 공공기관의 정보를 분석하여 해당 공기업의 '사업 종목'을 간결하게 추출하는 전문가입니다.
여러 기관 정보를 한 번에 받고 각 기관별로 사업 종목을 추출해야 합니다.
몇가지 예시를 제공하겠습니다. 예시를 참고하여 공기업들의 '사업 종목'을 추출해주세요.

중요: **'사업 종목'은 'xx업' 으로 끝나야 합니다.**

# 예시
한국가스안전공사: 에너지안전관리업
한국교통안전공단: 교통안전관리업
한국산업안전보건공단: 산업안전보건업
한국지역난방공사: 에너지공급업
한국토지주택공사: 토지주택사업"""

    class BusinessCategory(BaseModel):
        institution_name: str  
        business_category: str

    class BusinessCategoryList(BaseModel):
        categories: list[BusinessCategory]
        
    # API 키 설정
    openai.api_key = api_key
    
    # 결과를 저장할 리스트
    business_categories = []
    
    client = openai.OpenAI(api_key=api_key)
    
    # 기관명 컬럼 확인 (일반적으로 '기관명'이지만, 없을 경우 0번째 컬럼 사용)
    institution_name_col = '기관명' if '기관명' in df.columns else df.columns[0]
    
    # 배치 단위로 처리
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}")
        
        # 배치에 있는 모든 기관 정보를 하나의 문자열로 구성
        institution_batch_info = ""
        for idx, row in batch.iterrows():
            # 기관 정보 헤더 추가
            institution_info = f"----- 기관 {idx} -----\n"
            
            # 데이터프레임의 모든 컬럼 순회하며 정보 추가
            for col in df.columns:
                # 값이 있는 경우에만 추가
                if pd.notna(row[col]) and isinstance(row[col], (str, int, float)):
                    institution_info += f"{col}: {row[col]}\n"
            
            institution_batch_info += f"\n{institution_info}\n"
        
        try:
            # OpenAI API 요청 - 배치 전체를 한 번에 처리
            response = client.beta.chat.completions.parse(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"다음 공공기관들의 정보를 바탕으로 각 기관의 주요 사업 종목을 간결하게 추출하세요. 각 기관마다 기관명과 사업 종목을 쌍으로 응답해주세요.\n\n{institution_batch_info}"}
                ],
                temperature=0.5,
                max_tokens=1500,
                response_format=BusinessCategoryList
            )
            
            # 응답에서 사업 종목 리스트 추출
            category_list = response.choices[0].message.parsed.categories
            business_categories.extend(category_list)
            
            # API 요청 제한 방지를 위한 딜레이
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing batch starting with index {i}: {e}")
            # 에러 발생 시 기관별로 개별 처리
            batch_results = []
            for _, row in batch.iterrows():
                batch_results.append(BusinessCategory(
                    institution_name=row[institution_name_col], 
                    business_category="Error: 사업 종목 정보 추출 실패"
                ))
            business_categories.extend(batch_results)
    
    return business_categories

# 사용 예시
# df = pd.read_excel('공공기관_리스트.xlsx', header=1)
# df = df.drop(index=0).reset_index(drop=True)
# api_key = "your_api_key"
# business_categories = get_business_categories_batch(df, api_key, batch_size=10)

# # 결과를 DataFrame의 새 열에 추가
# df['사업 종목'] = [cat.business_category for cat in business_categories]
# df.to_excel('공공기관_사업종목.xlsx', index=False) 