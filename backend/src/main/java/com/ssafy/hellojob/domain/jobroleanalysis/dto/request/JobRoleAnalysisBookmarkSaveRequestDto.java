package com.ssafy.hellojob.domain.jobroleanalysis.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class JobRoleAnalysisBookmarkSaveRequestDto {

    @NotNull(message = ValidationMessage.JOB_ROLE_ANALYSIS_JOB_ROLE_ANALYSIS_ID_NOT_EMPTY)
    private Integer jobRoleAnalysisId;

}
