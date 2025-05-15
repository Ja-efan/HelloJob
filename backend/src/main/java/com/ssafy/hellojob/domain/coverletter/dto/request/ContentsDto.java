package com.ssafy.hellojob.domain.coverletter.dto.request;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class ContentsDto {
    private String contentQuestion;
    private int contentNumber;
    private List<Integer> contentExperienceIds;
    private List<Integer> contentProjectIds;
    private int contentLength;
    private String contentFirstPrompt;

    @Builder
    public ContentsDto(String contentQuestion, int contentNumber, List<Integer> contentExperienceIds, List<Integer> contentProjectIds, int contentLength, String contentFirstPrompt) {
        this.contentQuestion = contentQuestion;
        this.contentNumber = contentNumber;
        this.contentExperienceIds = contentExperienceIds;
        this.contentProjectIds = contentProjectIds;
        this.contentLength = contentLength;
        this.contentFirstPrompt = contentFirstPrompt;
    }
}
