package com.ssafy.hellojob.domain.coverlettercontent.dto.response;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContentStatus;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ChatResponseDto {
    private String aiMessage;
    private CoverLetterContentStatus contentStatus;

    @Builder
    public ChatResponseDto(String aiMessage, CoverLetterContentStatus contentStatus) {
        this.aiMessage = aiMessage;
        this.contentStatus = contentStatus;
    }
}
