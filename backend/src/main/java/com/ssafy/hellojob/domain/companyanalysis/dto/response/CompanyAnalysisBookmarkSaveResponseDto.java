package com.ssafy.hellojob.domain.companyanalysis.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class CompanyAnalysisBookmarkSaveResponseDto {

    private Integer companyAnalysisBookmarkId;
    private Integer companyAnalysisId;

}
