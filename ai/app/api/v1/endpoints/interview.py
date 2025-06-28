from typing import List
from fastapi import APIRouter
from app.schemas import interview
from app.services import interview_service
from app.services.gms import interview_service as gms_interview_service
from app.core.logger import app_logger

logger = app_logger

router = APIRouter(prefix="/interview", tags=["interview"])


####################################################### original interview service ########################################################

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


# ######################################################## gms interview service ########################################################

# @router.post("/question/cover-letter")
# async def create_interview_question_from_cover_letter_gms(request: interview.CreateQuestionRequest)->interview.CreateQuestionResponse:

#     questions = await gms_interview_service.create_interview_questions_from_cover_letter(request)
    
#     response = interview.CreateQuestionResponse(
#         cover_letter_id=request.cover_letter.cover_letter_id,
#         expected_questions=questions
#     )
#     logger.info(f"CreateInterviewQuestionFromCoverLetterGMSResponse: {response}")
#     return response

# @router.post("/feedback")
# async def feedback_interview_gms(request: interview.FeedbackInterviewRequest)->interview.FeedbackInterviewResponse:
    
#     result = await gms_interview_service.feedback_interview(request)
#     logger.info(f"FeedbackInterviewGMSResponse: {result}")
#     return result