package com.ssafy.hellojob.domain.jobroleanalysis.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class JobRoleAnalysisBookmarkSaveResponseDto {

    private Integer jobRoleAnalysisBookmarkId;
    private Integer jobRoleAnalysisId;

    @Builder
    public JobRoleAnalysisBookmarkSaveResponseDto(Integer jobRoleAnalysisBookmarkId, Integer jobRoleAnalysisId){
        this.jobRoleAnalysisBookmarkId = jobRoleAnalysisBookmarkId;
        this.jobRoleAnalysisId = jobRoleAnalysisId;
    }


}
