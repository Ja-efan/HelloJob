from fastapi import APIRouter
from app.schemas.cover_letter import CreateCoverLetterRequest, EditCoverLetterRequest, CreateCoverLetterResponse, EditCoverLetterResponse, ChatCoverLetterRequest, ChatCoverLetterResponse
from app.services.cover_letter_service import create_cover_letter_all, edit_cover_letter_service, chat_with_cover_letter_service
from app.core.logger import app_logger
from app.services.gms import cover_letter_service as gms_cover_letter_service

logger = app_logger

router = APIRouter(prefix="/cover-letter", tags=["cover-letter"])

# ####################################################### original cover letter service ########################################################

# @router.post("")
# async def create_cover_letter(request: CreateCoverLetterRequest):
#     """자기소개서 초안을 생성합니다."""
#     cover_letters_result = await create_cover_letter_all(request)
#     logger.info(f"CreateCoverLetterResponse: {cover_letters_result}")
#     return CreateCoverLetterResponse(cover_letters=cover_letters_result)


# # @router.post("/edit")
# # async def edit_cover_letter(request: EditCoverLetterRequest):
# #     """자기소개서 수정 방향을 제시합니다."""
# #     edit_suggestions_result = await edit_cover_letter_service(request)
# #     logger.info(f"EditCoverLetterResponse: {edit_suggestions_result}")
# #     return EditCoverLetterResponse(
# #         user_message=request.edit_content.user_message,
# #         ai_message=edit_suggestions_result
# #     )


# @router.post("/chat")
# async def chat_with_cover_letter(request: ChatCoverLetterRequest):
#     """자기소개서 관련 채팅 기능을 제공합니다."""
#     response = await chat_with_cover_letter_service(request)
#     # logger.info(f"ChatCoverLetterResponse: {response}")
#     if response["status"] == "success":
#         response = ChatCoverLetterResponse(
#             status=response["status"],
#             user_message=request.user_message,
#             ai_message=response["content"]
#         )
#         logger.info(f"ChatCoverLetterResponse: {response}")
#         return response
#     else:
#         response = ChatCoverLetterResponse(
#             status=response["status"],
#             user_message=request.user_message,
#             ai_message=response["content"]
#         )
#         logger.info(f"ChatCoverLetterResponse: {response}")
#         return response
    
    
######################################################## gms cover letter service ########################################################

@router.post("")
async def create_cover_letter_gms(request: CreateCoverLetterRequest):
    """GMS를 사용한 자기소개서 초안을 생성합니다."""
    cover_letters_result = await gms_cover_letter_service.create_cover_letter_all(request)
    logger.info(f"CreateCoverLetterResponse: {cover_letters_result}")
    return CreateCoverLetterResponse(cover_letters=cover_letters_result)

@router.post("/chat")
async def chat_with_cover_letter_gms(request: ChatCoverLetterRequest):
    """GMS를 사용한 자기소개서 관련 채팅 기능을 제공합니다."""
    response = await gms_cover_letter_service.chat_with_cover_letter_service(request)
    # logger.info(f"ChatCoverLetterResponse: {response}")
    if response["status"] == "success":
        response = ChatCoverLetterResponse(
            status=response["status"],
            user_message=request.user_message,
            ai_message=response["content"]
        )
        logger.info(f"ChatCoverLetterResponse: {response}")
        return response
    else:
        response = ChatCoverLetterResponse(
            status=response["status"],
            user_message=request.user_message,
            ai_message=response["content"]
        )
        logger.info(f"ChatCoverLetterResponse: {response}")
        return response