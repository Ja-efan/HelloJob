package com.ssafy.hellojob.domain.jobroleanalysis.dto.response;

import lombok.*;

@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class JobRoleAnalysisSaveResponseDto {

    private Integer jobRoleAnalysisId;

    @Builder
    public JobRoleAnalysisSaveResponseDto(Integer jobRoleAnalysisId){
        this.jobRoleAnalysisId = jobRoleAnalysisId;
    }

}
