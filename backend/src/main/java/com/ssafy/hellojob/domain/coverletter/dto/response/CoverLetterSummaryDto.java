package com.ssafy.hellojob.domain.coverletter.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class CoverLetterSummaryDto {
    private String coverLetterTitle;
    private List<Integer> contentIds;
    private Integer companyAnalysisId;
    private Integer jobRoleSnapshotId;

    @Builder
    public CoverLetterSummaryDto(String coverLetterTitle, List<Integer> contentIds, Integer companyAnalysisId, Integer jobRoleSnapshotId) {
        this.coverLetterTitle = coverLetterTitle;
        this.contentIds = contentIds;
        this.companyAnalysisId = companyAnalysisId;
        this.jobRoleSnapshotId = jobRoleSnapshotId;
    }
}
