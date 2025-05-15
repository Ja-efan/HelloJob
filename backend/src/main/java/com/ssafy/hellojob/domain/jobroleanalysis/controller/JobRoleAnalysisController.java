package com.ssafy.hellojob.domain.jobroleanalysis.controller;

import com.ssafy.hellojob.domain.jobroleanalysis.dto.request.JobRoleAnalysisBookmarkSaveRequestDto;
import com.ssafy.hellojob.domain.jobroleanalysis.dto.request.JobRoleAnalysisSaveRequestDto;
import com.ssafy.hellojob.domain.jobroleanalysis.dto.request.JobRoleAnalysisSearchCondition;
import com.ssafy.hellojob.domain.jobroleanalysis.dto.request.JobRoleAnalysisUpdateRequestDto;
import com.ssafy.hellojob.domain.jobroleanalysis.dto.response.*;
import com.ssafy.hellojob.domain.jobroleanalysis.service.JobRoleAnalysisService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/job-role-analysis")
public class JobRoleAnalysisController {

    private final JobRoleAnalysisService jobRoleAnalysisService;

    // 직무 분석 등록
    @PostMapping()
    public JobRoleAnalysisSaveResponseDto jobRoleAnalysisSave(@Valid @RequestBody JobRoleAnalysisSaveRequestDto requestDto,
                                                              @AuthenticationPrincipal UserPrincipal userPrincipal){

        if (userPrincipal == null) {
            throw new BaseException(ErrorCode.AUTH_NOT_FOUND);
        }

        Integer userId = userPrincipal.getUserId();

        JobRoleAnalysisSaveResponseDto responseDto = jobRoleAnalysisService.createJobRoleAnalysis(userId, requestDto);

        return responseDto;
    }

    // 직무 분석 상세 조회
    @GetMapping("/{jobRoleAnalysisId}")
    public JobRoleAnalysisDetailResponseDto jobRoleAnalysisDetail(@PathVariable("jobRoleAnalysisId") Integer jobRoleAnalysisId,
                                                                  @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();

        return jobRoleAnalysisService.searchJobRoleAnalysis(userId, jobRoleAnalysisId);
    }

    // 직무 분석 북마크 추가
    @PostMapping("/bookmark")
    public JobRoleAnalysisBookmarkSaveResponseDto jobRoleAnalysisBookmarkSave(@Valid @RequestBody JobRoleAnalysisBookmarkSaveRequestDto requestDto,
                                                                              @AuthenticationPrincipal UserPrincipal userPrincipal){

        Integer userId = userPrincipal.getUserId();

        JobRoleAnalysisBookmarkSaveResponseDto responseDto = jobRoleAnalysisService.addJobRoleBookmark(userId, requestDto);
        return responseDto;
    }

    // 직무 분석 북마크 삭제
    @DeleteMapping("/bookmark/{jobRoleAnalysisBookmarkId}")
    public void JobRoleAnalysisBookmarkDelete (@PathVariable("jobRoleAnalysisBookmarkId") Integer jobRoleAnalysisBookmarkId,
                                               @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();

        jobRoleAnalysisService.deleteJobRoleBookmark(jobRoleAnalysisBookmarkId, userId);
    }

    // 직무 분석 북마크 목록 조회
    @GetMapping("/bookmark")
    public List<JobRoleAnalysisListResponseDto> JobRoleAnalysisBookmarkList(@RequestParam(value = "companyId", required = false) Integer companyId,
                                                                            @AuthenticationPrincipal UserPrincipal userPrincipal) {
        Integer userId = userPrincipal.getUserId();
        List<JobRoleAnalysisListResponseDto> result;

        if (companyId == null) {
            result = jobRoleAnalysisService.searchJobRoleAnalysisBookmarkList(userId);
        } else {
            result = jobRoleAnalysisService.searchJobRoleAnalysisBookmarkListWithCompanyId(userId, companyId);
        }

        return result;

    }

    // 직무 분석 검색(기본값: companyId, 검색 조건: 직무명, 직무 분석 제목, 직무 카테고리)
    @GetMapping("/{companyId}/search")
    public List<JobRoleAnalysisSearchListResponseDto> JobRoleAnalysisSearch(@PathVariable Integer companyId,
                                                                            @ModelAttribute JobRoleAnalysisSearchCondition condition,
                                                                            @AuthenticationPrincipal UserPrincipal userPrincipal) {

        Integer userId = userPrincipal.getUserId();
        List<JobRoleAnalysisSearchListResponseDto> result = jobRoleAnalysisService.searchJobRoleAnalysis(userId, companyId, condition);

        return result;
    }

    // 직무 분석 삭제
    @DeleteMapping("/analysis/{jobRoleAnalysisId}")
    public void JobRoleAnalysisDelete(@PathVariable Integer jobRoleAnalysisId,
                                      @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();
        jobRoleAnalysisService.deleteJobRoleAnalysis(userId, jobRoleAnalysisId);
    }

    // 직무 분석 수정
    @PutMapping()
    public JobRoleAnalysisUpdateResponseDto JobRoleAnalysisUpdate(@Valid @RequestBody JobRoleAnalysisUpdateRequestDto requestDto, @AuthenticationPrincipal UserPrincipal userPrincipal){
        Integer userId = userPrincipal.getUserId();

        JobRoleAnalysisUpdateResponseDto result = jobRoleAnalysisService.updateJobRoleAnalysis(requestDto, userId);
        return result;
    }

}
