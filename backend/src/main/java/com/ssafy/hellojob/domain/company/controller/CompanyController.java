package com.ssafy.hellojob.domain.company.controller;

import com.ssafy.hellojob.domain.company.dto.CompanyDto;
import com.ssafy.hellojob.domain.company.dto.CompanyListDto;
import com.ssafy.hellojob.domain.company.service.CompanyService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/company")
public class CompanyController {

    private final CompanyService companyService;

    @GetMapping("/search")
    public List<CompanyListDto> getCompanyName(@RequestParam(value = "companyName", required = false) String companyName){
        List<CompanyListDto> result;

        if (companyName != null && !companyName.isEmpty()) {
            result = companyService.getCompanyByCompanyName(companyName);
        } else {
            result = companyService.getAllCompany();
        }

        return result;

    }

    @GetMapping("/{companyId}")
    public CompanyDto getCompanyDetail(@PathVariable("companyId") Integer companyId,
                                       @AuthenticationPrincipal UserPrincipal userPrincipal){

        CompanyDto result = companyService.getCompanyByCompanyId(companyId);
        return result;
    }



}
