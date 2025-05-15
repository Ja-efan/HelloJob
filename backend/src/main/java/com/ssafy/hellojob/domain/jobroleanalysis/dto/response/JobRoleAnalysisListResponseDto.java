package com.ssafy.hellojob.domain.jobroleanalysis.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
@AllArgsConstructor
public class JobRoleAnalysisListResponseDto {

    private Integer jobRoleAnalysisBookmarkId;
    private Integer jobRoleAnalysisId;
    private String companyName;
    private String jobRoleName;
    private String jobRoleAnalysisTitle;
    private String jobRoleCategory;
    private boolean isPublic;
    private Integer jobRoleViewCount;
    private Integer jobRoleBookmarkCount;
    private boolean bookmark;
    private LocalDateTime updatedAt;
    private LocalDateTime createdAt;

}
