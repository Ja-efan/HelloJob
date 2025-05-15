package com.ssafy.hellojob.domain.companyanalysis.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.sql.Date;
import java.time.LocalDateTime;
import java.util.List;

@Getter
@Builder
@AllArgsConstructor
public class CompanyAnalysisDetailResponseDto {

    private Integer companyAnalysisId;
    private String companyName;
    private LocalDateTime createdAt;
    private Integer companyViewCount;
    private String companyLocation;
    private String companySize;
    private String companyIndustry;
    private Integer companyAnalysisBookmarkCount;
    private boolean bookmark;
    private boolean isPublic;
    private String newsAnalysisData;
    private Date newsAnalysisDate;
    private List<String> newsAnalysisUrl;
    private String dartBrand;
    private String dartCompanyAnalysis;
    private String dartVision;
    private String dartFinancialSummery;
    private List<String> dartCategory;

}
