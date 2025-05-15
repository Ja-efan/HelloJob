package com.ssafy.hellojob.domain.coverletter.dto.request;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class CoverLetterRequestDto {
    private String coverLetterTitle;
    private Integer companyAnalysisId;
    private Integer jobRoleAnalysisId;
    private List<ContentsDto> contents;
}
