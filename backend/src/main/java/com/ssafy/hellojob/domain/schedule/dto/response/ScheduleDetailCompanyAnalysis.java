package com.ssafy.hellojob.domain.schedule.dto.response;

import lombok.*;

import java.sql.Date;
import java.time.LocalDateTime;
import java.util.List;

@Getter
@Builder
@AllArgsConstructor
public class ScheduleDetailCompanyAnalysis {

    private Integer companyAnalysisId;
    private String companyName;
    private String newsAnalysisData;
    private List<String> newsAnalysisUrl;
    private String dartBrand;
    private String dartCompanyAnalysis;
    private String dartVision;
    private String dartFinancialSummery;
    private List<String> dartCategory;

}
