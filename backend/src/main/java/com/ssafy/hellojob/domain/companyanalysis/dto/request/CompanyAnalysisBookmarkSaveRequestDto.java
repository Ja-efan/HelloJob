package com.ssafy.hellojob.domain.companyanalysis.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class CompanyAnalysisBookmarkSaveRequestDto {

    @NotNull(message = ValidationMessage.COMPANY_ANALYSIS_BOOKMARK_COMPANY_ANALYSIS_ID_NOT_EMPTY)
    private Integer companyAnalysisId;

}
