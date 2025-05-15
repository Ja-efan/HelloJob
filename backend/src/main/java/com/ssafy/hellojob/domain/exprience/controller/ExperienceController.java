package com.ssafy.hellojob.domain.exprience.controller;

import com.ssafy.hellojob.domain.exprience.dto.request.ExperienceRequestDto;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperienceCreateResponseDto;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperienceResponseDto;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperiencesResponseDto;
import com.ssafy.hellojob.domain.exprience.service.ExperienceService;
import com.ssafy.hellojob.global.auth.token.UserPrincipal;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@Slf4j
@RestController
@RequiredArgsConstructor
@RequestMapping("/api/v1/experience")
public class ExperienceController {

    private final ExperienceService experienceService;

    @PostMapping
    public ExperienceCreateResponseDto createExperience(
            @AuthenticationPrincipal UserPrincipal principal,
            @Valid @RequestBody ExperienceRequestDto requestDto) {
        Integer userId = principal.getUserId();

        return experienceService.createExperience(userId, requestDto);
    }

    @GetMapping
    public ResponseEntity<?> getExperiences(
            @AuthenticationPrincipal UserPrincipal principal) {
        Integer userId = principal.getUserId();

        List<ExperiencesResponseDto> experiences = experienceService.getExperiences(userId);

        return experiences.isEmpty()
                ? ResponseEntity.noContent().build()
                : ResponseEntity.ok(experiences);
    }

    @GetMapping("/{experienceId}")
    public ExperienceResponseDto getExperience(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer experienceId) {
        Integer userId = principal.getUserId();

        return experienceService.getExperience(userId, experienceId);
    }

    @PutMapping("/{experienceId}")
    public Map<String, String> updateExperience(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer experienceId,
            @Valid @RequestBody ExperienceRequestDto experienceRequestDto) {
        Integer userId = principal.getUserId();
        experienceService.updateExperience(userId, experienceId, experienceRequestDto);

        return Map.of("message", "경험이 수정되었습니다.");
    }

    @DeleteMapping("/{experienceId}")
    public Map<String, String> deleteExperience(
            @AuthenticationPrincipal UserPrincipal principal,
            @PathVariable Integer experienceId
    ) {
        Integer userId = principal.getUserId();
        experienceService.deleteExperience(userId, experienceId);

        return Map.of("message", "경험이 삭제되었습니다.");
    }
}
