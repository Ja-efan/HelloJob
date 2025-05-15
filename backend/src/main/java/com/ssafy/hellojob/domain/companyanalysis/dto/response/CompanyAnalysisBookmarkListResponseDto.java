package com.ssafy.hellojob.domain.companyanalysis.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.List;

@Getter
@Builder
@AllArgsConstructor
public class CompanyAnalysisBookmarkListResponseDto {

    private Integer companyAnalysisBookmarkId;
    private Integer companyAnalysisId;
    private String companyName; // 기업명
    private LocalDateTime createdAt;
    private Integer companyViewCount;
    private String companyLocation; // 기업 소재지
    private String companySize; // 기업 규모
    private String companyIndustry; // 기업 업종명
    private Integer companyAnalysisBookmarkCount;
    private boolean bookmark; // 북마크 여부
    private boolean isPublic;
    private List<String> dartCategory;

}
