package com.ssafy.hellojob.domain.coverlettercontent.controller;

import com.ssafy.hellojob.domain.coverlettercontent.dto.request.ChatRequestDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.request.CoverLetterUpdateRequestDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.ChatResponseDto;
import com.ssafy.hellojob.domain.coverlettercontent.dto.response.CoverLetterContentDto;
import com.ssafy.hellojob.domain.coverlettercontent.service.ChatLogService;
import com.ssafy.hellojob.domain.coverlettercontent.service.CoverLetterContentService;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import lombok.RequiredArgsConstructor;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("api/v1/cover-letter-content")
@RequiredArgsConstructor
public class CoverLetterContentController {

    private final CoverLetterContentService coverLetterContentService;
    private final ChatLogService chatLogService;

    @GetMapping("/{contentId}")
    public CoverLetterContentDto getCoverLetterContent(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer contentId) {
        User user = principal.getUser();
        CoverLetterContentDto response = coverLetterContentService.getCoverLetterContent(user, contentId);
        return response;
    }

    @PatchMapping("/{contentId}")
    public Map<String, String> updateCoverLetterContent(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer contentId,
            @RequestBody CoverLetterUpdateRequestDto requestDto
    ) {
        User user = principal.getUser();
        return coverLetterContentService.updateCoverLetterContent(user, contentId, requestDto);
    }

    @PostMapping("/{contentId}/chat")
    public ChatResponseDto sendChat(
            @PathVariable Integer contentId,
            @RequestBody ChatRequestDto requestDto
            ) {
        return chatLogService.sendChat(contentId, requestDto);
    }
}
