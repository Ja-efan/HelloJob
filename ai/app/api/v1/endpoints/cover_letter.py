from fastapi import APIRouter
from app.schemas.cover_letter import CreateCoverLetterRequest, EditCoverLetterRequest, CreateCoverLetterResponse, EditCoverLetterResponse, ChatCoverLetterRequest, ChatCoverLetterResponse
from app.services.cover_letter_service import create_cover_letter_all, edit_cover_letter_service, chat_with_cover_letter_service


router = APIRouter(prefix="/cover-letter", tags=["cover-letter"])

@router.post("")
async def create_cover_letter(request: CreateCoverLetterRequest):
    """자기소개서 초안을 생성합니다."""
    cover_letters_result = await create_cover_letter_all(request)
    
    return CreateCoverLetterResponse(cover_letters=cover_letters_result)


@router.post("/edit")
async def edit_cover_letter(request: EditCoverLetterRequest):
    """자기소개서 수정 방향을 제시합니다."""
    edit_suggestions_result = await edit_cover_letter_service(request)
    
    return EditCoverLetterResponse(
        user_message=request.edit_content.user_message,
        ai_message=edit_suggestions_result
    )


@router.post("/chat")
async def chat_with_cover_letter(request: ChatCoverLetterRequest):
    """자기소개서 관련 채팅 기능을 제공합니다."""
    response = await chat_with_cover_letter_service(request)
    
    if response["status"] == "success":
        return ChatCoverLetterResponse(
            status=response["status"],
            user_message=request.user_message,
            ai_message=response["content"]
        )
    else:
        return ChatCoverLetterResponse(
            status=response["status"],
            user_message=request.user_message,
            ai_message=response["content"]
        )