from pydantic import BaseModel
from app.schemas.cover_letter import Experience, Project
from typing import List

class CoverLetterContent(BaseModel):
    cover_letter_content_number: int  # 자기소개서 문항 번호 
    cover_letter_content_question: str  # 자기소개서 항목 (질문)
    cover_letter_content_detail: str  # 자기소개서 내용
    
class CoverLetter(BaseModel):
    cover_letter_id: int  # 자기소개서 ID
    cover_letter_contents: List[CoverLetterContent]  # 자기소개서 내용 리스트
    
class CreateQuestionRequest(BaseModel):
    cover_letter: CoverLetter
    experiences: List[Experience]
    projects: List[Project]
    
class CreateQuestionResponse(BaseModel):
    cover_letter_id: int
    expected_questions: List[str]
    
class QuestionAnswerPair(BaseModel): 
    """면접 질문과 답변 쌍"""
    interview_answer_id: int  # 면접 답변 ID
    interview_question: str  # 면접 질문
    interview_answer: str  # 면접 답변
    interview_question_category: str  # 질문 카테고리 ["인성면접", "자기소개서면접", "네트워크", "운영체제", "컴퓨터구조", "데이터베이스", "알고리즘", "보안", "자료구조", "기타"]
    
class FeedbackInterviewRequest(BaseModel):
    interview_question_answer_pairs: List[QuestionAnswerPair]  # 면접 질문과 답변 쌍의 리스트
    cover_letter_contents: List[CoverLetterContent]  # 자기소개서 내용 리스트
    
class SingleFeedback(BaseModel):
    interview_answer_id: int  # 면접 답변 ID
    feedback: str  # 피드백 
    follow_up_questions: List[str]  # 예상 꼬리 질문
    
class FeedbackInterviewResponse(BaseModel):
    single_feedbacks: List[SingleFeedback]  # 피드백 리스트
    overall_feedback: str  # 전반적인 피드백