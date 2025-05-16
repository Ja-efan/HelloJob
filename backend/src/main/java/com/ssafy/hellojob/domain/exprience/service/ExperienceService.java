package com.ssafy.hellojob.domain.exprience.service;

import com.ssafy.hellojob.domain.exprience.dto.request.ExperienceRequestDto;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperienceCreateResponseDto;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperienceResponseDto;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperiencesResponseDto;
import com.ssafy.hellojob.domain.exprience.entity.Experience;
import com.ssafy.hellojob.domain.exprience.repository.ExperienceRepository;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.service.UserReadService;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@Transactional
@RequiredArgsConstructor
public class ExperienceService {

    private final ExperienceRepository experienceRepository;
    private final UserReadService userReadService;
    private final ExperienceReadService experienceReadService;

    public ExperienceCreateResponseDto createExperience(Integer userId, ExperienceRequestDto experienceRequestDto) {
        User user = userReadService.findUserByIdOrElseThrow(userId);

        if (experienceRequestDto.getExperienceStartDate().isAfter(experienceRequestDto.getExperienceEndDate())) {
            log.debug("🌞 경험 시작 날짜: " + experienceRequestDto.getExperienceStartDate() + " 경험 종료 날짜: " + experienceRequestDto.getExperienceEndDate());
            throw new BaseException(ErrorCode.EXPERIENCE_DATE_NOT_VALID);
        }

        Experience experience = Experience.builder()
                .user(user)
                .experienceName(experienceRequestDto.getExperienceName())
                .experienceDetail(experienceRequestDto.getExperienceDetail())
                .experienceClient(experienceRequestDto.getExperienceClient())
                .experienceRole(experienceRequestDto.getExperienceRole())
                .experienceStartDate(experienceRequestDto.getExperienceStartDate())
                .experienceEndDate(experienceRequestDto.getExperienceEndDate())
                .build();

        experienceRepository.save(experience);
        Integer experienceId = experience.getExperienceId();

        return ExperienceCreateResponseDto.builder()
                .experienceId(experienceId)
                .build();
    }

    public List<ExperiencesResponseDto> getExperiences(Integer userId) {
        userReadService.findUserByIdOrElseThrow(userId);
        List<ExperiencesResponseDto> experiences = experienceRepository.findExperiencesByUserId(userId);
        return experiences;
    }

    public Page<ExperiencesResponseDto> getExperiencesPage(Integer userId, Pageable pageable) {
        userReadService.findUserByIdOrElseThrow(userId);
        Page<ExperiencesResponseDto> page = experienceRepository.findExperiencesPageByUserId(userId, pageable);
        return page;
    }

    public ExperienceResponseDto getExperience(Integer userId, Integer experienceId) {

        userReadService.findUserByIdOrElseThrow(userId);
        Experience experience = experienceReadService.findExperienceByIdOrElseThrow(experienceId);
        experienceReadService.checkExperienceValidation(userId, experience);

        return ExperienceResponseDto.builder()
                .experienceId(experience.getExperienceId())
                .experienceName(experience.getExperienceName())
                .experienceRole(experience.getExperienceRole())
                .experienceDetail(experience.getExperienceDetail())
                .experienceClient(experience.getExperienceClient())
                .experienceStartDate(experience.getExperienceStartDate())
                .experienceEndDate(experience.getExperienceEndDate())
                .updatedAt(experience.getUpdatedAt())
                .build();
    }

    public void updateExperience(Integer userId, Integer experienceId, ExperienceRequestDto experienceRequestDto) {

        userReadService.findUserByIdOrElseThrow(userId);
        Experience experience = experienceReadService.findExperienceByIdOrElseThrow(experienceId);
        experienceReadService.checkExperienceValidation(userId, experience);

        if (experienceRequestDto.getExperienceStartDate().isAfter(experienceRequestDto.getExperienceEndDate())) {
            log.debug("🌞 경험 시작 날짜: " + experienceRequestDto.getExperienceStartDate() + " 경험 종료 날짜: " + experienceRequestDto.getExperienceEndDate());
            throw new BaseException(ErrorCode.EXPERIENCE_DATE_NOT_VALID);
        }

        experience.updateExperience(experienceRequestDto);
    }

    public void deleteExperience(Integer userId, Integer experienceId) {

        userReadService.findUserByIdOrElseThrow(userId);
        Experience experience = experienceReadService.findExperienceByIdOrElseThrow(experienceId);
        experienceReadService.checkExperienceValidation(userId, experience);

        experienceRepository.deleteById(experienceId);
    }
}
