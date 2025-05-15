package com.ssafy.hellojob.domain.coverlettercontent.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class ChatMessageDto {
    private String sender;
    private String message;

    @Builder
    public ChatMessageDto(String sender, String message) {
        this.sender = sender;
        this.message = message;
    }
}
