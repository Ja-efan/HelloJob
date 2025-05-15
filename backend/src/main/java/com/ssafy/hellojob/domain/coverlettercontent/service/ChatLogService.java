package com.ssafy.hellojob.domain.coverlettercontent.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.hellojob.domain.coverlettercontent.dto.request.ChatRequestDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ChatMessageDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ChatResponseDto;
import com.ssafy.hellojob.domain.coverlettercontent.entity.ChatLog;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContentStatus;
import com.ssafy.hellojob.domain.coverlettercontent.repository.ChatLogRepository;
import com.ssafy.hellojob.domain.coverlettercontent.repository.CoverLetterContentRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
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
    private final CoverLetterContentRepository coverLetterContentRepository;

    // JSON을 자바 객체로 바꾸거나 자바 객체를 JSON으로 바꿔줌
    private final ObjectMapper mapper = new ObjectMapper();

    public List<ChatMessageDto> getContentChatLog(Integer contentId) {
        log.debug("🌞 지금 GetContentChatLog 들어옴");
        String chatLogString = chatLogRepository.findChatLogContentById(contentId);
        log.debug("🌞 ChatLogString: {}", chatLogString);

        if (chatLogString == null || chatLogString.isBlank()) return new ArrayList<>();

        List<ChatMessageDto> chatLog;

        try {
            chatLog = mapper.readValue(chatLogString, new TypeReference<>() {
            });
        } catch (JsonProcessingException e) {
            log.error("🌞 채팅 로그 파싱 실패: {}", chatLogString);
            throw new RuntimeException("채팅 로그 JSON 파싱 실패", e);
        }

        log.debug("🌞 chatLog {}", chatLog.toArray().toString());

        return chatLog;
    }

    @Transactional
    public ChatResponseDto sendChat(Integer contentId, ChatRequestDto requestDto) {
        // 사용자 메시지를 받은 다음 AI에 전달
        // AI로부터 메시지를 받으면 DB에 저장 & contentStatus 확인 후 IN_PROGRESS로 변경
        // AI 메시지를 받는 로직 들어가야 함
        ChatMessageDto userMessages = ChatMessageDto.builder()
                .sender("user")
                .message(requestDto.getUserMessage())
                .build();

        ChatMessageDto aiMessage = ChatMessageDto.builder()
                .sender("ai")
                .message("아직 AI 연결 안됐지롱")
                .build();

        // 본문 내용 저장
        String contentDetail = requestDto.getContentDetail();

        CoverLetterContent content = coverLetterContentRepository.findById(contentId)
                .orElseThrow(() -> new BaseException(ErrorCode.COVER_LETTER_CONTENT_NOT_FOUND));

        content.updateCoverLetterContentWithChat(contentDetail);

        // 새로운 채팅 배열
        List<ChatMessageDto> newChats = new ArrayList<>();

        Optional<ChatLog> chatLogOpt = chatLogRepository.findById(contentId);
        
        if (chatLogOpt.isEmpty()) {
            // 기존 로그 없으면 새로 생성
            newChats.add(userMessages);
            newChats.add(aiMessage);

            ChatLog newChat = ChatLog.builder()
                    .coverLetterContent(content)
                    .chatLogContent(toJson(newChats))
                    .updatedCount(1)
                    .build();

            chatLogRepository.save(newChat);
        } else {
            // 있으면 기존 로그를 String으로 바꿔서 추가한 후 다시 JSON형태로 변경
            ChatLog existingLog = chatLogOpt.get();

            newChats = parseJson(existingLog.getChatLogContent());

            newChats.add(userMessages);
            newChats.add(aiMessage);

            existingLog.updateChatLog(toJson(newChats));
        }

        // 작성 중이 아니라면 작성 중으로 상태 변경
        if (content.getContentStatus() != CoverLetterContentStatus.IN_PROGRESS) {
            content.updateContentStatus(CoverLetterContentStatus.IN_PROGRESS);
        }

        return ChatResponseDto.builder()
                .aiMessage(aiMessage.getMessage())
                .contentStatus(content.getContentStatus())
                .build();
    }

    // JSON 형태로 파싱
    private List<ChatMessageDto> parseJson(String json) {
        if (json == null || json.isBlank()) return new ArrayList<>();
        try {
            return mapper.readValue(json, new TypeReference<>() {});
        } catch (JsonProcessingException e) {
            throw new RuntimeException("채팅 로그 파싱 실패", e);
        }
    }

    // String 형태로 직렬화
    private String toJson(List<ChatMessageDto> messages) {
        try {
            return mapper.writeValueAsString(messages);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("채팅 JSON 직렬화 실패", e);
        }
    }


}
