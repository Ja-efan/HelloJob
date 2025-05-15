package com.ssafy.hellojob.domain.coverlettercontent.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@NoArgsConstructor
public class CoverLetterContentDto {
    private Integer contentId;
    private String contentQuestion;
    private int contentNumber;
    private int contentLength;
    private String contentDetail;
    private List<Integer> contentExperienceIds;
    private List<Integer> contentProjectIds;
    private String contentFirstPrompt;
    private List<ChatMessageDto> contentChatLog;
    private LocalDateTime contentUpdatedAt;

    @Builder
    public CoverLetterContentDto(Integer contentId, String contentQuestion, int contentNumber, int contentLength, String contentDetail, List<Integer> contentExperienceIds, List<Integer> contentProjectIds, String contentFirstPrompt, List<ChatMessageDto> contentChatLog, LocalDateTime contentUpdatedAt) {
        this.contentId = contentId;
        this.contentQuestion = contentQuestion;
        this.contentNumber = contentNumber;
        this.contentLength = contentLength;
        this.contentDetail = contentDetail;
        this.contentExperienceIds = contentExperienceIds;
        this.contentProjectIds = contentProjectIds;
        this.contentFirstPrompt = contentFirstPrompt;
        this.contentChatLog = contentChatLog;
        this.contentUpdatedAt = contentUpdatedAt;
    }
}
