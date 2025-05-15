package com.ssafy.hellojob.domain.company.dto;

import com.ssafy.hellojob.domain.company.entity.CompanySize;
import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class CompanyListDto {

    private Integer id;

    private String companyName;

    private String companyLocation;

    private CompanySize companySize;

    private String companyIndustry;
    private LocalDateTime updatedAt;

}
