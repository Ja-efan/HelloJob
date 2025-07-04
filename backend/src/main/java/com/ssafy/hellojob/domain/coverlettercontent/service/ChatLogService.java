package com.ssafy.hellojob.domain.coverlettercontent.service;

import com.ssafy.hellojob.domain.coverlettercontent.dto.ai.request.AIChatRequestDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.ai.response.AIChatResponseDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ChatMessageDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ChatResponseDto;
import com.ssafy.hellojob.domain.coverlettercontent.entity.ChatLog;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContentStatus;
import com.ssafy.hellojob.domain.coverlettercontent.repository.ChatLogRepository;
import com.ssafy.hellojob.global.common.client.FastApiClientService;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import com.ssafy.hellojob.global.util.JsonUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class ChatLogService {

    private final ChatLogRepository chatLogRepository;
    private final FastApiClientService fastApiClientService;
    private final JsonUtil jsonUtil;

    public List<ChatMessageDto> getContentChatLog(Integer contentId) {
        log.debug("🌞 지금 GetContentChatLog 들어옴");
        String chatLogString = chatLogRepository.findChatLogContentById(contentId);
        log.debug("🌞 ChatLogString: {}", chatLogString);

        if (chatLogString == null || chatLogString.isBlank()) return new ArrayList<>();

        return jsonUtil.parseMessage(chatLogString);
    }

    @Transactional
    public ChatResponseDto sendChat(CoverLetterContent content, AIChatRequestDto aiChatRequestDto) {
        // FastAPI 요청
        AIChatResponseDto response = fastApiClientService.sendChatToFastApi(aiChatRequestDto);

        if (response.getStatus().equals("error") || response.getAi_message() == null || response.getAi_message().isBlank()) {
            throw new BaseException(ErrorCode.FAST_API_RESPONSE_ERROR);
        }

        ChatMessageDto userMessage = ChatMessageDto.builder()
                .sender("user")
                .message(aiChatRequestDto.getUser_message())
                .build();

        ChatMessageDto aiMessage = ChatMessageDto.builder()
                .sender("ai")
                .message(response.getAi_message())
                .build();

        // 본문 내용 저장
        String contentDetail = aiChatRequestDto.getCover_letter().getCover_letter();
        content.updateCoverLetterContentWithChat(contentDetail);

        // 채팅 저장
        saveNewChatLog(content, userMessage, aiMessage);

        updateContentStatus(content);

        return ChatResponseDto.builder()
                .aiMessage(aiMessage.getMessage())
                .contentStatus(content.getContentStatus())
                .build();
    }

    public void saveNewChatLog(CoverLetterContent content, ChatMessageDto userMessage, ChatMessageDto aiMessage) {
        List<ChatMessageDto> newChats = new ArrayList<>();

        Optional<ChatLog> chatLogOpt = chatLogRepository.findById(content.getContentId());

        if (chatLogOpt.isEmpty()) {
            // 기존 로그 없으면 새로 생성
            newChats.add(userMessage);
            newChats.add(aiMessage);

            ChatLog newChat = ChatLog.builder()
                    .coverLetterContent(content)
                    .chatLogContent(jsonUtil.messageToJson(newChats))
                    .updatedCount(1)
                    .build();

            chatLogRepository.save(newChat);
        } else {
            // 있으면 기존 로그를 String으로 바꿔서 추가한 후 다시 JSON형태로 변경
            ChatLog existingLog = chatLogOpt.get();

            newChats = jsonUtil.parseMessage(existingLog.getChatLogContent());

            newChats.add(userMessage);
            newChats.add(aiMessage);

            existingLog.updateChatLog(jsonUtil.messageToJson(newChats));
        }
    }

    public void updateContentStatus(CoverLetterContent content) {
        // 작성 중이 아니라면 작성 중으로 상태 변경
        if (content.getContentStatus() != CoverLetterContentStatus.IN_PROGRESS) {
            content.updateContentStatus(CoverLetterContentStatus.IN_PROGRESS);
        }
    }
}
