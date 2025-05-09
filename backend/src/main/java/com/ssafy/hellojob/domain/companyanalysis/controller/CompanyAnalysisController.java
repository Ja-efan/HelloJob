package com.ssafy.hellojob.domain.companyanalysis.controller;

import com.ssafy.hellojob.domain.company.service.CompanyService;
import com.ssafy.hellojob.domain.companyanalysis.dto.request.CompanyAnalysisBookmarkSaveRequestDto;
import com.ssafy.hellojob.domain.companyanalysis.dto.request.CompanyAnalysisFastApiRequestDto;
import com.ssafy.hellojob.domain.companyanalysis.dto.request.CompanyAnalysisRequestDto;
import com.ssafy.hellojob.domain.companyanalysis.dto.response.*;
import com.ssafy.hellojob.domain.companyanalysis.service.CompanyAnalysisService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import com.ssafy.hellojob.global.common.client.FastApiClientService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/company-analysis")
public class CompanyAnalysisController {

    private final FastApiClientService fastApiClientService;

    private final CompanyAnalysisService companyAnalysisService;
    private final CompanyService companyService;

    // 기업 분석 전체 목록 조회
    @GetMapping("/all-analysis")
    public List<CompanyAnalysisListResponseDto> CompanyAnalysisAll(@AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();
        List<CompanyAnalysisListResponseDto> result = companyAnalysisService.searchAllCompanyAnalysis(userId);

        return result;
    }

    // 기업 분석 상세 조회
    @GetMapping("/{companyAnalysisId}")
    public CompanyAnalysisDetailResponseDto CompanyAnalysisDetail(@PathVariable("companyAnalysisId") Integer companyAnalysisId,
                                                                  @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();
        CompanyAnalysisDetailResponseDto result = companyAnalysisService.detailCompanyAnalysis(userId, companyAnalysisId);

        return result;
    }

    // 테스트용 post 요청 코드
    @PostMapping("/test")
    public CompanyAnalysisBookmarkSaveRequestDto CompanyAnalysisRequest(@RequestBody CompanyAnalysisFastApiResponseDto responseDto,
                                                                        @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();

        CompanyAnalysisRequestDto requestDto = CompanyAnalysisRequestDto.builder()
                .companyId(1)
                .isPublic(true)
                .basic(true)
                .plus(false)
                .financial(true)
                .build();

        CompanyAnalysisBookmarkSaveRequestDto result = companyAnalysisService.createCompanyAnalysis(userId, requestDto, responseDto);

        return result;

    }

    // fast API 구현 코드
    @PostMapping()
    public CompanyAnalysisBookmarkSaveRequestDto CompanyAnalysisRequest(@RequestBody CompanyAnalysisRequestDto requestDto,
                                                                        @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();
        String companyName = companyService.getCompanyNameByCompanyId(requestDto.getCompanyId());
        
        log.debug("프론트에서 기업 분석 요청 들어옴");
        log.debug("기업명: {}", companyName);
        log.debug("기업ID: {}", requestDto.getCompanyId());
        log.debug("isPublic: {}", requestDto.isPublic());
        log.debug("isBasic: {}", requestDto.isBasic());
        log.debug("isPlus: {}", requestDto.isPlus());
        log.debug("isFinancial: {}", requestDto.isFinancial());

        CompanyAnalysisFastApiRequestDto fastApiRequestDto = CompanyAnalysisFastApiRequestDto.builder()
                .company_name(companyName)
                .base(requestDto.isBasic())
                .plus(requestDto.isPlus())
                .fin(requestDto.isFinancial())
                .build();

        log.debug("fast API로 요청 보냄 !!! ");

        CompanyAnalysisFastApiResponseDto responseDto = fastApiClientService.sendJobAnalysisToFastApi(fastApiRequestDto);

        log.debug("fast API에서 응답 받음 !!! ");
        log.debug("비전 : {}", responseDto.getCompany_vision());

        CompanyAnalysisBookmarkSaveRequestDto result = companyAnalysisService.createCompanyAnalysis(userId, requestDto, responseDto);
        return result;

    }

    // 기업 분석 검색
    @GetMapping("/search/{companyId}")
    public List<CompanyAnalysisListResponseDto> CompanyAnalysisSearch(@PathVariable("companyId") Integer companyId,
                                                   @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();
        List<CompanyAnalysisListResponseDto> result = companyAnalysisService.searchByCompanyIdCompanyAnalysis(companyId, userId);

        return result;
    }

    // 기업 분석 북마크 추가
    @PostMapping("/bookmark")
    public CompanyAnalysisBookmarkSaveResponseDto CompanyAnalysisBookmarkSave(@Valid @RequestBody CompanyAnalysisBookmarkSaveRequestDto requestDto,
                                                                              @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();

        CompanyAnalysisBookmarkSaveResponseDto responseDto = companyAnalysisService.addCompanyAnalysisBookmark(userId, requestDto);
        return responseDto;
    }

    // 기업 분석 북마크 해제
    @DeleteMapping("/bookmark/{companyAnalysisId}")
    public void CompanyAnalysisBookmarkDelete(@PathVariable("companyAnalysisId") Integer companyAnalysisId,
                                              @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();
        companyAnalysisService.deleteCompanyAnalysisBookmark(companyAnalysisId, userId);
    }

    // 기업 분석 북마크 목록 조회
    @GetMapping("/bookmark")
    public List<CompanyAnalysisBookmarkListResponseDto> CompanyAnalysisBookmarkList(@RequestParam(value = "companyId", required = false) Integer companyId,
                                                                                    @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();
        List<CompanyAnalysisBookmarkListResponseDto> result;

        if (companyId == null) {
            result = companyAnalysisService.searchCompanyAnalysisBookmarkList(userId);
        } else {
            result = companyAnalysisService.searchCompanyAnalysisBookmarkListWithCompanyId(userId, companyId);
        }

        return result;

    }

}
