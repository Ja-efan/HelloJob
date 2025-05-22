package com.ssafy.hellojob.domain.coverlettercontent.service;

import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterExperience;
import com.ssafy.hellojob.domain.coverlettercontent.repository.CoverLetterExperienceRepository;
import com.ssafy.hellojob.domain.exprience.entity.Experience;
import com.ssafy.hellojob.domain.exprience.repository.ExperienceRepository;
import com.ssafy.hellojob.domain.project.entity.Project;
import com.ssafy.hellojob.domain.project.repository.ProjectRepository;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class CoverLetterExperienceService {

    private final ExperienceRepository experienceRepository;
    private final ProjectRepository projectRepository;
    private final CoverLetterExperienceRepository coverLetterExperienceRepository;

    @Transactional
    public void saveCoverLetterExperience(List<Integer> experienceIds, User user, CoverLetterContent content) {
        for (Integer experienceId : experienceIds) {

            Experience experience = experienceRepository.findById(experienceId)
                    .orElseThrow(() -> new BaseException(ErrorCode.EXPERIENCE_NOT_FOUND));

            if (!experience.getUser().getUserId().equals(user.getUserId()))
                throw new BaseException(ErrorCode.EXPERIENCE_MISMATCH);

            CoverLetterExperience coverLetterExperience = CoverLetterExperience.builder()
                    .coverLetterContent(content)
                    .experience(experience)
                    .build();

            content.getExperiences().add(coverLetterExperience);

            coverLetterExperienceRepository.save(coverLetterExperience);
        }
    }

    @Transactional
    public void saveCoverLetterProject(List<Integer> projectIds, User user, CoverLetterContent content) {
        for (Integer projectId : projectIds) {

            Project project = projectRepository.findById(projectId)
                    .orElseThrow(() -> new BaseException(ErrorCode.EXPERIENCE_NOT_FOUND));

            if (!project.getUser().getUserId().equals(user.getUserId()))
                throw new BaseException(ErrorCode.EXPERIENCE_MISMATCH);

            CoverLetterExperience coverLetterExperience = CoverLetterExperience.builder()
                    .coverLetterContent(content)
                    .project(project)
                    .build();

            content.getExperiences().add(coverLetterExperience);

            coverLetterExperienceRepository.save(coverLetterExperience);
        }
    }

    @Transactional(readOnly = true)
    public List<Integer> getCoverLetterExperienceIds(Integer contentId) {
        return coverLetterExperienceRepository.findExperiencesByContentId(contentId);
    }

    @Transactional(readOnly = true)
    public List<Integer> getCoverLetterProjectIds(Integer contentId) {
        return coverLetterExperienceRepository.findProjectsByContentId(contentId);
    }
}
