package com.ssafy.hellojob.domain.coverletter.controller;

import com.ssafy.hellojob.domain.coverletter.dto.request.CoverLetterRequestDto;
import com.ssafy.hellojob.domain.coverletter.dto.response.CoverLetterCreateResponseDto;
import com.ssafy.hellojob.domain.coverletter.dto.response.CoverLetterStatusesDto;
import com.ssafy.hellojob.domain.coverletter.dto.response.CoverLetterSummaryDto;
import com.ssafy.hellojob.domain.coverletter.service.CoverLetterService;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("api/v1/cover-letter")
public class CoverLetterController {

    private final CoverLetterService coverLetterService;

    @PostMapping
    public CoverLetterCreateResponseDto createCoverLetter(
            @AuthenticationPrincipal UserPrincipal principal,
            @RequestBody CoverLetterRequestDto requestDto
    ) {
        User user = principal.getUser();

        return coverLetterService.createCoverLetter(user, requestDto);
    }

    @GetMapping("/status/{coverLetterId}")
    public CoverLetterStatusesDto getCoverLetterStatuses(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer coverLetterId
    ) {
        User user = principal.getUser();
        return coverLetterService.getCoverLetterStatuses(user, coverLetterId);
    }

    @GetMapping("/{coverLetterId}")
    public CoverLetterSummaryDto getCoverLetterSummary(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer coverLetterId
    ) {
        User user = principal.getUser();
        return coverLetterService.getCoverLetterSummary(user, coverLetterId);
    }

    @PatchMapping("/{coverLetterId}")
    public Map<String, String> saveAll(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer coverLetterId
    ) {
        User user = principal.getUser();
        return coverLetterService.saveAll(user, coverLetterId);
    }

    @DeleteMapping("/{coverLetterId}")
    public Map<String, String> deleteCoverLetter(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer coverLetterId
    ) {
        User user = principal.getUser();
        return coverLetterService.deleteCoverLetter(user, coverLetterId);
    }
}
