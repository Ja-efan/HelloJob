package com.ssafy.hellojob.domain.companyanalysis.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.sql.Date;
import java.util.List;

@Getter
@Builder
@AllArgsConstructor
public class CompanyAnalysisFastApiResponseDto {

    private String company_name;
    private Date analysis_date;
    private String company_brand;
    private String company_analysis;
    private String company_vision;
    private String company_finance;
    private String news_summary;
    private List<String> news_urls;

}
