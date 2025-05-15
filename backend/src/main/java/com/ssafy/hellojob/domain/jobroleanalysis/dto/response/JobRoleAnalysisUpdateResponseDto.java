package com.ssafy.hellojob.domain.jobroleanalysis.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class JobRoleAnalysisUpdateResponseDto {

    private Integer jobRoleAnalysisId;

    @Builder
    public JobRoleAnalysisUpdateResponseDto(Integer jobRoleAnalysisId){
        this.jobRoleAnalysisId = jobRoleAnalysisId;
    }


}
