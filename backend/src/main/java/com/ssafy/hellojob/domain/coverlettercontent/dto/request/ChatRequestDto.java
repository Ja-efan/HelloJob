package com.ssafy.hellojob.domain.coverlettercontent.dto.request;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class ChatRequestDto {
    private String userMessage;
    private String contentDetail;
}
