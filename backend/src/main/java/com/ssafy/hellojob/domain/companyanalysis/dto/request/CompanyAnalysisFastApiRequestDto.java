package com.ssafy.hellojob.domain.companyanalysis.dto.request;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
@AllArgsConstructor
public class CompanyAnalysisFastApiRequestDto {

    private String company_name;
    private boolean base;
    private boolean plus;
    private boolean fin;

}