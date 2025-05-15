from fastapi import APIRouter
from app.schemas import interview
from app.services import interview_service
from typing import List

router = APIRouter(prefix="/interview", tags=["interview"])

@router.post("/question/cover-letter")
async def create_interview_question_from_cover_letter(request: interview.CreateQuestionRequest)->interview.CreateQuestionResponse:

    # 면접 질문 생성 서비스 호출
    questions = await interview_service.create_interview_questions_from_cover_letter(request)
    
    return interview.CreateQuestionResponse(
        cover_letter_id=request.cover_letter.cover_letter_id,
        expected_questions=questions
    )

@router.post("/feedback")
async def feedback_interview(request: interview.FeedbackInterviewRequest)->interview.FeedbackInterviewResponse:
    
    result = await interview_service.feedback_interview(request)
    
    return result