package com.ssafy.hellojob.domain.company.service;

import com.ssafy.hellojob.domain.company.dto.CompanyDto;
import com.ssafy.hellojob.domain.company.dto.CompanyListDto;
import com.ssafy.hellojob.domain.company.entity.Company;
import com.ssafy.hellojob.domain.company.repository.CompanyRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
@RequiredArgsConstructor
public class CompanyService {

    private final CompanyRepository companyRepository;

    public List<CompanyListDto> getAllCompany(){
        return companyRepository.getAllCompany();
    }

    public List<CompanyListDto> getCompanyByCompanyName(String companyName){
        return companyRepository.getCompanyByCompanyName(companyName);
    }

    public String getCompanyNameByCompanyId(Integer companyId){
        String result =  companyRepository.getCompanyNameByCompanyId(companyId)
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_NOT_FOUND));

        return result;
    }

    public CompanyDto getCompanyByCompanyId(Integer companyId){

        Company company = companyRepository.findByCompanyId(companyId)
                .orElseThrow(() -> new BaseException(ErrorCode.COMPANY_NOT_FOUND));

        CompanyDto result = CompanyDto.builder()
                .id(companyId)
                .companyName(company.getCompanyName())
                .companyIndustry(company.getCompanyIndustry())
                .companyLocation(company.getCompanyLocation())
                .companySize(company.getCompanySize())
                .updatedAt(company.getUpdatedAt())
                .build();

        return result;
    }

}
