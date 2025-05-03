package com.ssafy.hellojob.domain.coverlettercontent.dto;

import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ContentQuestionStatusDto;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@NoArgsConstructor
public class CoverLetterStatuses {
    private Integer coverLetterId;
    private int totalContentQuestionCount;
    private List<ContentQuestionStatusDto> contentQuestionStatuses;
    private LocalDateTime updatedAt;

    @Builder
    public CoverLetterStatuses(Integer coverLetterId, int totalContentQuestionCount, List<ContentQuestionStatusDto> contentQuestionStatuses, LocalDateTime updatedAt) {
        this.coverLetterId = coverLetterId;
        this.totalContentQuestionCount = totalContentQuestionCount;
        this.contentQuestionStatuses = contentQuestionStatuses;
        this.updatedAt = updatedAt;
    }
}
