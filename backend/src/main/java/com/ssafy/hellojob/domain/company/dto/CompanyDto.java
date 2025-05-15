package com.ssafy.hellojob.domain.company.dto;

import com.ssafy.hellojob.domain.company.entity.CompanySize;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
public class CompanyDto {

    private Integer id;

    private String companyName;

    private String companyLocation;

    private CompanySize companySize;

    private String companyIndustry;

    private LocalDateTime updatedAt;

}
