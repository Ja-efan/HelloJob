package com.ssafy.hellojob.domain.coverletter.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.hellojob.domain.coverletter.dto.response.ChatMessageDto;
import com.ssafy.hellojob.domain.coverletter.entity.ChatLog;
import com.ssafy.hellojob.domain.coverletter.repository.ChatLogRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class ChatLogService {

    private final ChatLogRepository chatLogRepository;
    // JSON을 자바 객체로 바꾸거나 자바 객체를 JSON으로 바꿔주는 줌
    private final ObjectMapper mapper = new ObjectMapper();

    public List<ChatMessageDto> getContentChatLog(Integer contentId) {
        String chatLogString = chatLogRepository.findChatLogContentById(contentId);

        if (chatLogString == null || chatLogString.isBlank()) return new ArrayList<>();

        List<ChatMessageDto> chatLog;
        try {
            chatLog = mapper.readValue(chatLogString, new TypeReference<>() {
            });
        } catch (JsonProcessingException e) {
            log.error("🌞 채팅 로그 파싱 실패: {}", chatLogString);
            throw new RuntimeException("채팅 로그 JSON 파싱 실패", e);
        }

        return chatLog;
    }
}
