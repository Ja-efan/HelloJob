package com.ssafy.hellojob.domain.user.controller;

import com.ssafy.hellojob.domain.coverletter.dto.response.MyPageCoverLetterDto;
import com.ssafy.hellojob.domain.coverletter.service.CoverLetterService;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperiencesResponseDto;
import com.ssafy.hellojob.domain.exprience.service.ExperienceService;
import com.ssafy.hellojob.domain.jobroleanalysis.dto.response.JobRoleAnalysisSearchListResponseDto;
import com.ssafy.hellojob.domain.jobroleanalysis.service.JobRoleAnalysisService;
import com.ssafy.hellojob.domain.project.dto.response.ProjectsResponseDto;
import com.ssafy.hellojob.domain.project.service.ProjectService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/mypage")
public class MypageController {

    private final JobRoleAnalysisService jobRoleAnalysisService;
    private final CoverLetterService coverLetterService;
    private final ExperienceService experienceService;
    private final ProjectService projectService;

    @GetMapping("/job-role-analysis")
    public List<JobRoleAnalysisSearchListResponseDto> JobRoleAnalysisListSearchByUser(@AuthenticationPrincipal UserPrincipal userPrincipal) {

        Integer userId = userPrincipal.getUserId();
        List<JobRoleAnalysisSearchListResponseDto> result = jobRoleAnalysisService.searchJobRoleAnalysisByUserId(userId);

        return result;
    }

    @GetMapping("/cover-letter")
    public ResponseEntity<?> getCoverLetterForMyPage(
            @AuthenticationPrincipal UserPrincipal principal,
            @PageableDefault(size = 10, sort = "updatedAt", direction = Sort.Direction.DESC) Pageable pageable) {
        Integer userId = principal.getUserId();
        Page<MyPageCoverLetterDto> page = coverLetterService.getCoverLettersForMaPage(userId, pageable);

        return page.isEmpty()
                ? ResponseEntity.noContent().build()
                : ResponseEntity.ok(page);
    }

    // 마이페이지 경험 목록 Pageable
    @GetMapping("/experience")
    public ResponseEntity<?> getExperiencesPage(
            @AuthenticationPrincipal UserPrincipal principal,
            @PageableDefault(size = 10, sort = "updatedAt", direction = Sort.Direction.DESC) Pageable pageable) {
        Integer userId = principal.getUserId();

        Page<ExperiencesResponseDto> experiences = experienceService.getExperiencesPage(userId, pageable);

        return experiences.isEmpty()
                ? ResponseEntity.noContent().build()
                : ResponseEntity.ok(experiences);
    }

    // 마이페이지 프로젝트 목록 Pageable
    @GetMapping("/project")
    public ResponseEntity<?> getProjectsPage(
            @AuthenticationPrincipal UserPrincipal principal,
            @PageableDefault(size = 10, sort = "updatedAt", direction = Sort.Direction.DESC) Pageable pageable) {
        Integer userId = principal.getUserId();
        Page<ProjectsResponseDto> list = projectService.getProjectsPage(userId, pageable);
        return list.isEmpty()
                ? ResponseEntity.noContent().build()
                : ResponseEntity.ok(list);
    }
}
