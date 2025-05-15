package com.ssafy.hellojob.domain.coverlettercontent.dto.response;

import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContentStatus;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class ContentQuestionStatusDto {
    private Integer contentId;
    private Integer contentNumber;
    private CoverLetterContentStatus contentStatus;
}
